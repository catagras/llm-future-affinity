from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.trace import Span

from llm_future_affinity.config import AppConfig
from llm_future_affinity.domain import GameRecord, ModelReply, RunStatus, Track
from llm_future_affinity.openrouter import OpenRouterClient, RetryExhausted
from llm_future_affinity.persistence import AsyncCsvWriter, read_output
from llm_future_affinity.runner import ConversationSession
from llm_future_affinity.scheduler import Scheduler
from llm_future_affinity.telemetry import NullTelemetry
from tests.unit.test_runner import attempt, reply


class CoordinatedClient:
    def __init__(self, results: dict[tuple[int, str], list[ModelReply | Exception]], delay: float = 0.001) -> None:
        self.results = results
        self.delay = delay
        self.current = 0
        self.maximum = 0

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        parent_span: Span,
        audit_attributes: Mapping[str, Any],
    ) -> ModelReply:
        del messages, parent_span
        key = int(audit_attributes["experiment.game_id"]), str(audit_attributes["experiment.track"])
        self.current += 1
        self.maximum = max(self.maximum, self.current)
        try:
            await asyncio.sleep(self.delay)
            result = self.results[key].pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self.current -= 1


def sessions(app_config: AppConfig, client: CoordinatedClient, count: int) -> list[ConversationSession]:
    telemetry = NullTelemetry()
    return [
        ConversationSession(
            app_config,
            "test-model",
            GameRecord(game_id, "ABCD"),
            Track.A,
            1,
            None,
            cast(OpenRouterClient, client),
            telemetry,
            trace.INVALID_SPAN,
        )
        for game_id in range(1, count + 1)
    ]


async def test_scheduler_honors_concurrency_and_persists(tmp_path: Path, app_config: AppConfig) -> None:
    client = CoordinatedClient({(game_id, "A"): [reply("SUBMIT ABCD")] for game_id in range(1, 5)})
    writer = AsyncCsvWriter(tmp_path / "results.csv")
    await writer.start()
    scheduler = Scheduler(
        sessions(app_config, client, 4),
        writer,
        NullTelemetry(),
        2,
        app_config.execution.limits,
        shuffle=False,
    )
    result = await scheduler.run()
    await writer.close()
    assert result.successful
    assert client.maximum == 2
    assert len(read_output(tmp_path / "results.csv")) == 4


async def test_breaker_stops_unstarted_but_finishes_started(tmp_path: Path, app_config: AppConfig) -> None:
    failure = RetryExhausted("provider down", [attempt(timeout=True)])
    client = CoordinatedClient(
        {
            (1, "A"): [failure],
            (2, "A"): [reply("QUERY AABC"), reply("SUBMIT ABCD")],
            (3, "A"): [reply("SUBMIT ABCD")],
            (4, "A"): [reply("SUBMIT ABCD")],
        },
        delay=0.01,
    )
    writer = AsyncCsvWriter(tmp_path / "results.csv")
    await writer.start()
    scheduler = Scheduler(
        sessions(app_config, client, 4),
        writer,
        NullTelemetry(),
        2,
        app_config.execution.limits,
        shuffle=False,
    )
    result = await scheduler.run()
    await writer.close()
    assert result.admission_stopped
    assert [row.game_id for row in result.rows] == [1, 2]
    assert {row.run_status for row in result.rows} == {RunStatus.API_ERROR, RunStatus.COMPLETED}
    assert client.results[(3, "A")]  # never started


async def test_force_stop_before_start_writes_no_attempts(tmp_path: Path, app_config: AppConfig) -> None:
    client = CoordinatedClient({(1, "A"): [reply("SUBMIT ABCD")]})
    writer = AsyncCsvWriter(tmp_path / "results.csv")
    await writer.start()
    scheduler = Scheduler(
        sessions(app_config, client, 1),
        writer,
        NullTelemetry(),
        1,
        app_config.execution.limits,
        shuffle=False,
    )
    scheduler.request_stop(force=True)
    result = await scheduler.run()
    await writer.close()
    assert result.forced
    assert result.rows == []


async def test_progress_receives_terminal_rows(tmp_path: Path, app_config: AppConfig) -> None:
    client = CoordinatedClient({(1, "A"): [reply("SUBMIT ABCD")]})
    writer = AsyncCsvWriter(tmp_path / "results.csv")
    await writer.start()
    progress: list[RunStatus] = []
    scheduler = Scheduler(
        sessions(app_config, client, 1),
        writer,
        NullTelemetry(),
        1,
        app_config.execution.limits,
        progress=lambda row, active: progress.append(row.run_status),
        shuffle=False,
    )
    await scheduler.run()
    await writer.close()
    assert progress == [RunStatus.COMPLETED]


async def test_http_attempt_limit_does_not_schedule_an_extra_turn(tmp_path: Path, app_config: AppConfig) -> None:
    app_config.execution.limits.max_total_http_attempts = 1
    client = CoordinatedClient({(1, "A"): [reply("QUERY AABC"), reply("SUBMIT ABCD")]})
    writer = AsyncCsvWriter(tmp_path / "results.csv")
    await writer.start()
    scheduler = Scheduler(
        sessions(app_config, client, 1),
        writer,
        NullTelemetry(),
        1,
        app_config.execution.limits,
        shuffle=False,
    )
    result = await scheduler.run()
    await writer.close()
    assert result.reason == "max_http_attempts"
    assert result.rows[0].run_status is RunStatus.INTERRUPTED
    assert result.rows[0].num_model_calls == 1
    assert len(client.results[(1, "A")]) == 1
