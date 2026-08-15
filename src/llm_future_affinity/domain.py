"""Shared domain types for the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


class Track(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Action(StrEnum):
    QUERY = "QUERY"
    SUBMIT = "SUBMIT"


class SubmissionType(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    INVALID_SUBMISSION = "invalid_submission"
    API_ERROR = "api_error"
    PROVIDER_MISMATCH = "provider_mismatch"
    CACHE_HIT = "cache_hit"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    FORCE_RERUN = "force_rerun"


@dataclass(frozen=True, slots=True)
class GameRecord:
    game_id: int
    hidden_code: str


@dataclass(frozen=True, slots=True)
class Feedback:
    exact: int
    misplaced: int


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    valid: bool
    action: Action | None = None
    value: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    def add(self, other: Usage) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "total_tokens",
        ):
            current = getattr(self, name)
            incoming = getattr(other, name)
            if incoming is not None:
                setattr(self, name, (current or 0) + incoming)
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd


@dataclass(slots=True)
class HttpAttempt:
    attempt_number: int
    started_at: str
    finished_at: str
    duration_ms: int
    status_code: int | None
    error_category: str | None
    error_message: str | None
    usage: Usage = field(default_factory=Usage)
    generation_id: str | None = None
    cache_status: str = "UNKNOWN"
    observed_provider: str | None = None
    observed_endpoint: str | None = None
    observed_quantization: str | None = None
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelReply:
    content: str
    assistant_message: dict[str, Any]
    attempts: list[HttpAttempt]
    usage: Usage
    generation_id: str | None
    cache_status: str
    observed_provider: str | None
    observed_endpoint: str | None
    observed_quantization: str | None
