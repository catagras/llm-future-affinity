"""Top-level dry-run planning and real execution orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry.trace import Span
from tqdm import tqdm

from llm_future_affinity.config import LoadedConfig
from llm_future_affinity.credentials import openrouter_api_key
from llm_future_affinity.debug import DebugWriter
from llm_future_affinity.domain import GameRecord, Track
from llm_future_affinity.game import load_games
from llm_future_affinity.manifest import build_manifest, ensure_manifest, utc_now, validate_manifest
from llm_future_affinity.openrouter import OpenRouterClient
from llm_future_affinity.persistence import AsyncCsvWriter, ModelOutputLock, OutputRow, ResumeState, read_output
from llm_future_affinity.prompting import render_all_prompts
from llm_future_affinity.runner import ConversationSession
from llm_future_affinity.scheduler import InterruptHandler, Scheduler, SchedulerResult
from llm_future_affinity.telemetry import NullTelemetry, OtelTelemetry, Telemetry


@dataclass(frozen=True, slots=True)
class RunPaths:
    games: Path
    output: Path
    manifest: Path
    lock: Path
    debug_dir: Path


@dataclass(frozen=True, slots=True)
class RunPlan:
    paths: RunPaths
    games: list[GameRecord]
    existing_rows: list[OutputRow]
    selected: list[tuple[GameRecord, list[Track]]]
    completed_count: int
    batch_size: int | None
    manifest: dict[str, object]


def paths_for(loaded: LoadedConfig, model_key: str) -> RunPaths:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", model_key)
    if not safe_key:
        raise ValueError("model key does not contain any filename-safe characters")
    output_dir = loaded.resolve(loaded.config.experiment.output_dir)
    return RunPaths(
        games=loaded.resolve(loaded.config.experiment.games_file),
        output=output_dir / f"{safe_key}.csv",
        manifest=output_dir / f"{safe_key}.manifest.json",
        lock=output_dir / f"{safe_key}.lock",
        debug_dir=loaded.resolve(loaded.config.experiment.debug_dir),
    )


def create_plan(loaded: LoadedConfig, model_key: str, batch_size: int | None) -> RunPlan:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch size must be positive")
    loaded.config.model_for(model_key)
    paths = paths_for(loaded, model_key)
    games = load_games(paths.games, loaded.config.game)
    rows = read_output(paths.output)
    state = ResumeState(rows, model_key)
    selected = state.select_batch(games, batch_size)
    expected_manifest = build_manifest(loaded, model_key, paths.games, len(games), paths.output)
    if paths.manifest.exists():
        validate_manifest(paths.manifest, expected_manifest)
    return RunPlan(
        paths=paths,
        games=games,
        existing_rows=rows,
        selected=selected,
        completed_count=len(state.completed),
        batch_size=batch_size,
        manifest=expected_manifest,
    )


async def execute_plan(
    loaded: LoadedConfig,
    model_key: str,
    plan: RunPlan,
    *,
    disable_otel: bool,
    debug: bool,
    show_progress: bool = True,
    api_key: str | None = None,
) -> SchedulerResult:
    telemetry: Telemetry = NullTelemetry()
    debug_writer: DebugWriter | None = None
    client: OpenRouterClient | None = None
    writer: AsyncCsvWriter | None = None
    invocation_span: Span | None = None
    progress: tqdm[Any] | None = None
    result: SchedulerResult | None = None

    with ModelOutputLock(plan.paths.lock):
        current_rows = read_output(plan.paths.output)
        state = ResumeState(current_rows, model_key)
        selected = state.select_batch(plan.games, plan.batch_size)
        if not selected:
            return SchedulerResult(rows=[], admission_stopped=False, forced=False, reason=None)

        key = api_key or openrouter_api_key()
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required for --execute")
        if not disable_otel and loaded.config.observability.enabled_for_execute:
            telemetry = OtelTelemetry(loaded.config.observability)

        try:
            await telemetry.preflight()
            model = loaded.config.model_for(model_key)
            client = OpenRouterClient(key, model, loaded.config.execution, telemetry, debug=debug)
            await client.preflight()
            ensure_manifest(plan.paths.manifest, plan.manifest)

            writer = AsyncCsvWriter(plan.paths.output)
            await writer.start()
            if debug:
                timestamp = utc_now().replace("-", "").replace(":", "").replace(".", "")
                debug_writer = DebugWriter(plan.paths.debug_dir / f"{timestamp}_{_safe_key(model_key)}.jsonl")
                debug_writer.start()

            invocation_span = telemetry.start_span(
                "experiment.invocation",
                attributes={
                    "experiment.name": loaded.config.experiment.name,
                    "experiment.model_key": model_key,
                    "experiment.batch_games": len(selected),
                },
            )
            sessions = [
                ConversationSession(
                    loaded.config,
                    model_key,
                    game,
                    track,
                    *state.attempt_identity(game.game_id, track),
                    client,
                    telemetry,
                    invocation_span,
                )
                for game, tracks in selected
                for track in tracks
            ]

            if show_progress:
                progress = tqdm(
                    total=len(plan.games) * len(Track),
                    initial=len(state.completed),
                    unit="track",
                    desc=model_key,
                )

            accumulated_cost = sum(row.total_cost_usd or 0.0 for row in current_rows)
            accumulated_tokens = sum(row.total_tokens or 0 for row in current_rows)

            def update_progress(row: OutputRow, active: int) -> None:
                nonlocal accumulated_cost, accumulated_tokens
                accumulated_cost += row.total_cost_usd or 0.0
                accumulated_tokens += row.total_tokens or 0
                if progress is None:
                    return
                if row.run_status.value == "completed":
                    progress.update(1)
                progress.set_postfix(
                    active=active,
                    cost=f"${accumulated_cost:.4f}",
                    status=row.run_status.value,
                    tokens=accumulated_tokens,
                )

            scheduler = Scheduler(
                sessions,
                writer,
                telemetry,
                loaded.config.execution.max_in_flight_calls,
                loaded.config.execution.limits,
                debug_writer=debug_writer,
                progress=update_progress,
            )
            with InterruptHandler(scheduler):
                result = await scheduler.run()
        finally:
            if invocation_span is not None:
                telemetry.end_span(invocation_span)
            if progress is not None:
                progress.close()
            if writer is not None:
                await writer.close()
            if debug_writer is not None:
                debug_writer.close()
            if client is not None:
                await client.close()
            await telemetry.shutdown()

    assert result is not None
    if telemetry.failed:
        return SchedulerResult(
            rows=result.rows,
            admission_stopped=True,
            forced=result.forced,
            reason=result.reason or "otel_export_failure",
        )
    return result


def rendered_prompts(loaded: LoadedConfig) -> dict[Track, str]:
    return render_all_prompts(loaded.config.game, loaded.config.prompt)


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)
