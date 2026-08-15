"""Work-conserving conversation scheduler and admission breaker."""

from __future__ import annotations

import asyncio
import random
import signal
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import FrameType

from llm_future_affinity.config import LimitsConfig
from llm_future_affinity.debug import DebugWriter
from llm_future_affinity.domain import RunStatus
from llm_future_affinity.persistence import AsyncCsvWriter, OutputRow
from llm_future_affinity.runner import ConversationSession, StepOutcome
from llm_future_affinity.telemetry import Telemetry


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    rows: list[OutputRow]
    admission_stopped: bool
    forced: bool
    reason: str | None

    @property
    def successful(self) -> bool:
        return not self.admission_stopped and all(row.run_status is RunStatus.COMPLETED for row in self.rows)


ProgressCallback = Callable[[OutputRow, int], None]


class Scheduler:
    def __init__(
        self,
        sessions: Iterable[ConversationSession],
        writer: AsyncCsvWriter,
        telemetry: Telemetry,
        max_in_flight: int,
        limits: LimitsConfig,
        *,
        debug_writer: DebugWriter | None = None,
        progress: ProgressCallback | None = None,
        shuffle: bool = True,
    ) -> None:
        session_list = list(sessions)
        if shuffle:
            random.SystemRandom().shuffle(session_list)
        self._sessions = session_list
        self.ready = deque(session_list)
        self.writer = writer
        self.telemetry = telemetry
        self.max_in_flight = max_in_flight
        self.limits = limits
        self.debug_writer = debug_writer
        self.progress = progress
        self.admission_stopped = False
        self.force_stop = False
        self.stop_reason: str | None = None
        self.rows: list[OutputRow] = []
        self._active: dict[asyncio.Task[StepOutcome], ConversationSession] = {}
        self._started_clock = time.monotonic()
        self._consecutive_failures = 0

    def request_stop(self, force: bool = False, reason: str = "operator_interrupt") -> None:
        self.admission_stopped = True
        self.stop_reason = self.stop_reason or reason
        if force:
            self.force_stop = True
            for task in tuple(self._active):
                task.cancel()

    async def run(self) -> SchedulerResult:
        while self.ready or self._active:
            self._check_limits()
            if self.force_stop:
                await self._cancel_ready_started(
                    RunStatus.CANCELLED,
                    "forced_interrupt",
                    "conversation cancelled by second interrupt",
                )
            elif self._hard_limit_reached():
                await self._cancel_ready_started(
                    RunStatus.INTERRUPTED,
                    "execution_limit",
                    self.stop_reason or "execution stopped",
                )
            self._fill_slots()
            if not self._active:
                break

            done, _ = await asyncio.wait(self._active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                session = self._active.pop(task)
                if task.cancelled():
                    session.cancel(
                        RunStatus.CANCELLED, "forced_interrupt", "conversation cancelled by second interrupt"
                    )
                    await self._persist(session)
                    continue
                try:
                    outcome = task.result()
                except Exception as error:  # unexpected bugs must still preserve the attempted run
                    session.cancel(RunStatus.API_ERROR, "internal_error", str(error))
                    outcome = StepOutcome(terminal=True, open_breaker=True)

                self._check_limits()
                if outcome.open_breaker:
                    self.request_stop(reason=session.error_category or "conversation_failure")
                if outcome.terminal:
                    await self._persist(session)
                elif self._hard_limit_reached() or self.force_stop:
                    session.cancel(RunStatus.INTERRUPTED, "execution_limit", self.stop_reason or "execution stopped")
                    await self._persist(session)
                else:
                    self.ready.append(session)

                if self.telemetry.failed:
                    self.request_stop(reason="otel_export_failure")

        return SchedulerResult(
            rows=self.rows,
            admission_stopped=self.admission_stopped,
            forced=self.force_stop,
            reason=self.stop_reason,
        )

    def _fill_slots(self) -> None:
        while len(self._active) < self.max_in_flight:
            session = self._next_eligible()
            if session is None:
                return
            task = asyncio.create_task(session.step())
            self._active[task] = session

    def _next_eligible(self) -> ConversationSession | None:
        if not self.ready:
            return None
        if not self.admission_stopped:
            return self.ready.popleft()
        if self._hard_limit_reached():
            return None
        for _ in range(len(self.ready)):
            session = self.ready.popleft()
            if session.started:
                return session
        return None

    async def _persist(self, session: ConversationSession) -> None:
        row = session.to_output_row()
        await self.writer.write(row)
        if self.debug_writer is not None:
            await self.debug_writer.write_conversation(session.debug_record())
        await self.telemetry.force_flush()
        self.rows.append(row)
        if row.run_status is RunStatus.COMPLETED:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        if self.progress is not None:
            self.progress(row, len(self._active))

    async def _cancel_ready_started(self, status: RunStatus, category: str, message: str) -> None:
        remaining: deque[ConversationSession] = deque()
        while self.ready:
            session = self.ready.popleft()
            if session.started:
                session.cancel(status, category, message)
                await self._persist(session)
            else:
                remaining.append(session)
        self.ready = remaining

    def _check_limits(self) -> None:
        reason = self._limit_reason()
        if reason is not None:
            self.admission_stopped = True
            self.stop_reason = self.stop_reason or reason

    def _hard_limit_reached(self) -> bool:
        return self.stop_reason in {"max_runtime", "max_http_attempts", "max_cost", "max_consecutive_failures"}

    def _limit_reason(self) -> str | None:
        if (
            self.limits.max_runtime_seconds is not None
            and time.monotonic() - self._started_clock >= self.limits.max_runtime_seconds
        ):
            return "max_runtime"
        attempts = sum(len(session.attempts) for session in self._all_known_sessions())
        if self.limits.max_total_http_attempts is not None and attempts >= self.limits.max_total_http_attempts:
            return "max_http_attempts"
        cost = sum(session.total_usage.cost_usd or 0.0 for session in self._all_known_sessions())
        if self.limits.max_total_cost_usd is not None and cost >= self.limits.max_total_cost_usd:
            return "max_cost"
        if (
            self.limits.max_consecutive_failed_conversations is not None
            and self._consecutive_failures >= self.limits.max_consecutive_failed_conversations
        ):
            return "max_consecutive_failures"
        return None

    def _all_known_sessions(self) -> set[ConversationSession]:
        return set(self._sessions)


class InterruptHandler:
    """First Ctrl-C is graceful; the second cancels active work immediately."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self._count = 0
        self._previous: signal._HANDLER | None = None

    def __enter__(self) -> InterruptHandler:
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *_: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del signum, frame
        self._count += 1
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(self.scheduler.request_stop, self._count >= 2, "operator_interrupt")
