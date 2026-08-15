"""Append-only result persistence, locking, and resume state."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import portalocker
from pydantic import BaseModel, ConfigDict, Field

from llm_future_affinity.domain import GameRecord, RunStatus, SubmissionType, Track


class OutputRow(BaseModel):
    """One attempted model/game/track conversation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    attempt_id: str
    attempt_number: int
    supersedes_attempt_id: str | None = None
    game_id: int
    model_key: str
    model_family: str
    model_id: str
    requested_provider: str
    observed_provider: str | None = None
    requested_endpoint: str
    observed_endpoint: str | None = None
    requested_quantization: str | None = None
    observed_quantization: str | None = None
    track: Track
    i2_identity: str
    started_at: str
    finished_at: str
    duration_ms: int
    trace_id: str | None = None
    span_id: str | None = None
    hidden_code: str
    code_length: int
    symbol_set_size: int
    initial_query_credits: int
    queries_used: int
    credits_remaining: int
    final_answer: str | None = None
    positions_correct: int | None = None
    final_score: float | None = None
    solved: bool | None = None
    submission_type: SubmissionType | None = None
    num_model_calls: int
    num_http_attempts: int
    num_transport_retries: int
    num_invalid_responses: int
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cache_write_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    token_totals_complete: bool
    cost_total_complete: bool
    cache_hit_detected: bool
    prompt_hash: str
    inference_settings: dict[str, Any]
    routing_settings: dict[str, Any]
    run_status: RunStatus
    analysis_eligible: bool
    exclusion_reasons: list[str] = Field(default_factory=list)
    error_category: str | None = None
    error_message: str | None = None
    parse_error: str | None = None
    interaction_trace: list[dict[str, Any]] = Field(default_factory=list)


FIELDNAMES = list(OutputRow.model_fields)
JSON_FIELDS = {"inference_settings", "routing_settings", "exclusion_reasons", "interaction_trace"}
BOOL_FIELDS = {
    "solved",
    "token_totals_complete",
    "cost_total_complete",
    "cache_hit_detected",
    "analysis_eligible",
}

type IntendedRunKey = tuple[str, int, Track]


def intended_key(model_key: str, game_id: int, track: Track) -> IntendedRunKey:
    return model_key, game_id, track


def serialize_row(row: OutputRow) -> dict[str, str]:
    raw = row.model_dump(mode="json")
    serialized: dict[str, str] = {}
    for field_name in FIELDNAMES:
        value = raw[field_name]
        if value is None:
            serialized[field_name] = ""
        elif field_name in JSON_FIELDS:
            serialized[field_name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(value, bool):
            serialized[field_name] = "true" if value else "false"
        else:
            serialized[field_name] = str(value)
    return serialized


def deserialize_row(raw: dict[str, str], line_number: int | None = None) -> OutputRow:
    converted: dict[str, Any] = dict(raw)
    for key, value in tuple(converted.items()):
        if value == "":
            converted[key] = None
        elif key in JSON_FIELDS:
            converted[key] = json.loads(value)
        elif key in BOOL_FIELDS:
            if value not in {"true", "false"}:
                raise ValueError(f"invalid boolean value {value!r} for {key}")
            converted[key] = value == "true"
    try:
        return OutputRow.model_validate(converted)
    except Exception as error:
        location = f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"invalid output row{location}: {error}") from error


def read_output(path: Path) -> list[OutputRow]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError("output CSV header does not match the current schema")
        return [deserialize_row(row, line_number=index) for index, row in enumerate(reader, start=2)]


class ResumeState:
    def __init__(self, rows: Iterable[OutputRow], model_key: str) -> None:
        self.model_key = model_key
        all_rows = list(rows)
        foreign_models = sorted({row.model_key for row in all_rows if row.model_key != model_key})
        if foreign_models:
            raise ValueError(f"output contains rows for other models: {foreign_models}")
        self.rows = all_rows
        counts = Counter(
            intended_key(row.model_key, row.game_id, row.track)
            for row in self.rows
            if row.run_status is RunStatus.COMPLETED
        )
        duplicates = [key for key, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"multiple completed rows found for intended runs: {duplicates}")
        self.completed = set(counts)

    def is_complete(self, game_id: int, track: Track) -> bool:
        return intended_key(self.model_key, game_id, track) in self.completed

    def select_batch(self, games: Iterable[GameRecord], batch_size: int | None) -> list[tuple[GameRecord, list[Track]]]:
        selected: list[tuple[GameRecord, list[Track]]] = []
        for game in sorted(games, key=lambda item: item.game_id):
            pending = [track for track in Track if not self.is_complete(game.game_id, track)]
            if not pending:
                continue
            selected.append((game, pending))
            if batch_size is not None and len(selected) >= batch_size:
                break
        return selected

    def attempt_identity(self, game_id: int, track: Track) -> tuple[int, str | None]:
        matching = [row for row in self.rows if row.game_id == game_id and row.track is track]
        if not matching:
            return 1, None
        latest = max(matching, key=lambda row: row.attempt_number)
        return latest.attempt_number + 1, latest.attempt_id


class ModelOutputLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock: portalocker.Lock | None = None

    def __enter__(self) -> ModelOutputLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = portalocker.Lock(str(self.path), mode="a", timeout=0)
        try:
            self._lock.acquire()
        except portalocker.exceptions.LockException as error:
            raise RuntimeError(f"output is locked by another process: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None


type WriteItem = tuple[OutputRow, asyncio.Future[None]] | None


class AsyncCsvWriter:
    """One append task owns the CSV file for the lifetime of an execution."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._queue: asyncio.Queue[WriteItem] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._handle: TextIO | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("CSV writer is already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists() and self.path.stat().st_size > 0
        if existing:
            read_output(self.path)
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        if not existing:
            csv.DictWriter(self._handle, fieldnames=FIELDNAMES).writeheader()
            self._flush()
        self._task = asyncio.create_task(self._run())

    async def write(self, row: OutputRow) -> None:
        if self._task is None:
            raise RuntimeError("CSV writer is not started")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((row, future))
        await future

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        self._task = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    async def _run(self) -> None:
        assert self._handle is not None
        writer = csv.DictWriter(self._handle, fieldnames=FIELDNAMES)
        while True:
            item = await self._queue.get()
            if item is None:
                break
            row, future = item
            try:
                writer.writerow(serialize_row(row))
                self._flush()
            except Exception as error:
                future.set_exception(error)
            else:
                future.set_result(None)

    def _flush(self) -> None:
        assert self._handle is not None
        self._handle.flush()
        os.fsync(self._handle.fileno())
