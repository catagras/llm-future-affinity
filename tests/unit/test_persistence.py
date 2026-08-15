from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

import pytest

from llm_future_affinity.domain import GameRecord, RunStatus, SubmissionType, Track
from llm_future_affinity.persistence import (
    FIELDNAMES,
    AsyncCsvWriter,
    ModelOutputLock,
    OutputRow,
    ResumeState,
    deserialize_row,
    read_output,
    serialize_row,
)


def make_row(**overrides: Any) -> OutputRow:
    values: dict[str, Any] = {
        "run_id": "attempt-1",
        "attempt_id": "attempt-1",
        "attempt_number": 1,
        "game_id": 1,
        "model_key": "model",
        "model_family": "family",
        "model_id": "author/model",
        "requested_provider": "provider",
        "observed_provider": "provider",
        "requested_endpoint": "provider/exact",
        "observed_endpoint": "provider/exact",
        "track": "A",
        "i2_identity": "continuation",
        "started_at": "2026-01-01T00:00:00.000Z",
        "finished_at": "2026-01-01T00:00:01.000Z",
        "duration_ms": 1000,
        "hidden_code": "ABCD",
        "code_length": 4,
        "symbol_set_size": 4,
        "initial_query_credits": 2,
        "queries_used": 1,
        "credits_remaining": 1,
        "final_answer": "ABCA",
        "positions_correct": 3,
        "final_score": 0.75,
        "solved": False,
        "submission_type": SubmissionType.PARTIAL,
        "num_model_calls": 2,
        "num_http_attempts": 2,
        "num_transport_retries": 0,
        "num_invalid_responses": 0,
        "total_input_tokens": 10,
        "total_output_tokens": 2,
        "total_tokens": 12,
        "total_cost_usd": 0.01,
        "token_totals_complete": True,
        "cost_total_complete": True,
        "cache_hit_detected": False,
        "prompt_hash": "hash",
        "inference_settings": {"temperature": 0},
        "routing_settings": {"endpoint_slug": "provider/exact"},
        "run_status": RunStatus.COMPLETED,
        "analysis_eligible": True,
        "exclusion_reasons": [],
        "interaction_trace": [{"step": 1}],
    }
    values.update(overrides)
    return OutputRow.model_validate(values)


def test_row_round_trip() -> None:
    row = make_row()
    serialized = serialize_row(row)
    assert list(serialized) == FIELDNAMES
    assert serialized["analysis_eligible"] == "true"
    assert serialized["inference_settings"] == '{"temperature":0}'
    assert deserialize_row(serialized) == row


def test_deserialize_rejects_bad_boolean() -> None:
    raw = serialize_row(make_row())
    raw["solved"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        deserialize_row(raw)


def test_read_output_missing_and_bad_header(tmp_path: Path) -> None:
    assert read_output(tmp_path / "missing.csv") == []
    path = tmp_path / "bad.csv"
    path.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_output(path)


async def test_async_writer_appends_and_flushes(tmp_path: Path) -> None:
    path = tmp_path / "output.csv"
    writer = AsyncCsvWriter(path)
    await writer.start()
    await asyncio.gather(writer.write(make_row()), writer.write(make_row(attempt_id="attempt-2", run_id="attempt-2")))
    await writer.close()
    assert len(read_output(path)) == 2
    with path.open("r", encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == FIELDNAMES


async def test_writer_lifecycle_errors(tmp_path: Path) -> None:
    writer = AsyncCsvWriter(tmp_path / "output.csv")
    with pytest.raises(RuntimeError, match="not started"):
        await writer.write(make_row())
    await writer.start()
    with pytest.raises(RuntimeError, match="already"):
        await writer.start()
    await writer.close()
    await writer.close()


def test_resume_selects_partial_games_and_attempts() -> None:
    rows = [
        make_row(game_id=1, track=Track.A),
        make_row(game_id=1, track=Track.B, run_id="b", attempt_id="b", run_status=RunStatus.API_ERROR),
        make_row(game_id=2, track=Track.A, run_id="c", attempt_id="c", run_status=RunStatus.FORCE_RERUN),
    ]
    state = ResumeState(rows, "model")
    selected = state.select_batch([GameRecord(2, "ABCD"), GameRecord(1, "ABCD")], batch_size=1)
    assert selected[0][0].game_id == 1
    assert selected[0][1] == [Track.B, Track.C, Track.D]
    assert state.attempt_identity(1, Track.B) == (2, "b")
    assert state.attempt_identity(1, Track.C) == (1, None)


def test_resume_rejects_duplicate_completed() -> None:
    with pytest.raises(ValueError, match="multiple completed"):
        ResumeState([make_row(), make_row(run_id="two", attempt_id="two", attempt_number=2)], "model")


def test_resume_rejects_rows_from_another_model() -> None:
    with pytest.raises(ValueError, match="other models"):
        ResumeState([make_row(model_key="other")], "model")


def test_model_output_lock_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "model.lock"
    with ModelOutputLock(path), pytest.raises(RuntimeError, match="locked"), ModelOutputLock(path):
        pass
