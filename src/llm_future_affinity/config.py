"""Typed experiment configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, StringConstraints, model_validator

from llm_future_affinity.domain import Track


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExperimentConfig(StrictModel):
    name: NonEmptyString
    games_file: Path
    output_dir: Path = Path("outputs")
    debug_dir: Path = Path("debug")


class GameConfig(StrictModel):
    code_length: PositiveInt
    symbols: list[str]
    allow_repeated_symbols: bool = True
    initial_query_credits: PositiveInt

    @model_validator(mode="after")
    def validate_symbols(self) -> Self:
        if not self.symbols:
            raise ValueError("game.symbols must not be empty")
        if any(len(symbol) != 1 or symbol.isspace() for symbol in self.symbols):
            raise ValueError("every game symbol must be exactly one non-whitespace character")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("game.symbols must be unique")
        if not self.allow_repeated_symbols and self.code_length > len(self.symbols):
            raise ValueError("code_length exceeds the symbol count while repetition is disabled")
        return self


class TrackPromptConfig(StrictModel):
    i2_identity: NonEmptyString
    beneficiary_clause: NonEmptyString


class PromptConfig(StrictModel):
    base_template: NonEmptyString
    invalid_response_template: NonEmptyString
    tracks: dict[Track, TrackPromptConfig]

    @model_validator(mode="after")
    def validate_tracks(self) -> Self:
        if set(self.tracks) != set(Track):
            raise ValueError("prompt.tracks must contain exactly A, B, C, and D")
        return self


class RoutingConfig(StrictModel):
    endpoint_slug: NonEmptyString
    quantizations: list[str] | None = None
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True


class ReasoningConfig(StrictModel):
    enabled: bool | None
    effort: str | None
    max_tokens: PositiveInt | None
    exclude: bool | None

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("reasoning.effort and reasoning.max_tokens are mutually exclusive")
        return self


class ThinkingConfig(StrictModel):
    display: Literal["summarized", "omitted"]


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    max_tokens: PositiveInt
    temperature: float | None = Field(ge=0)
    top_p: float | None = Field(ge=0, le=1)
    top_k: int | None = Field(ge=0)
    min_p: float | None = Field(ge=0, le=1)
    seed: int | None
    reasoning: ReasoningConfig | None
    thinking: ThinkingConfig | None


class ModelConfig(StrictModel):
    model_family: NonEmptyString
    model_id: NonEmptyString
    routing: RoutingConfig
    inference: InferenceConfig


class RetryConfig(StrictModel):
    max_attempts: Literal[4] = 4
    initial_delay_seconds: PositiveFloat = 1.0
    max_delay_seconds: PositiveFloat = 20.0
    jitter_ratio: float = Field(default=0.25, ge=0, le=1)


class LimitsConfig(StrictModel):
    max_model_calls_per_conversation: PositiveInt | None = None
    max_total_http_attempts: PositiveInt | None = None
    max_total_cost_usd: PositiveFloat | None = None
    max_runtime_seconds: PositiveFloat | None = None
    max_consecutive_failed_conversations: PositiveInt | None = None


class ExecutionConfig(StrictModel):
    max_in_flight_calls: PositiveInt = 8
    request_timeout_seconds: PositiveFloat = 120
    metadata_timeout_seconds: PositiveFloat = 30
    retry: RetryConfig = Field(default_factory=RetryConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)


class ObservabilityConfig(StrictModel):
    enabled_for_execute: bool = True
    otlp_endpoint: str = "http://localhost:4318"
    health_endpoint: str = "http://localhost:13133/ready"
    service_name: NonEmptyString = "llm-future-affinity"
    flush_timeout_seconds: PositiveFloat = 15


class AppConfig(StrictModel):
    experiment: ExperimentConfig
    game: GameConfig
    prompt: PromptConfig
    models: dict[str, ModelConfig]
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def model_for(self, key: str) -> ModelConfig:
        try:
            return self.models[key]
        except KeyError as error:
            available = ", ".join(sorted(self.models)) or "<none>"
            raise ValueError(f"unknown model key {key!r}; available keys: {available}") from error


class LoadedConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: AppConfig
    source_path: Path

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent

    def resolve(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.base_dir / value).resolve()


def load_config(path: Path) -> LoadedConfig:
    source_path = path.resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    config = AppConfig.model_validate(raw)
    validate_prompt_placeholders(config.prompt)
    return LoadedConfig(config=config, source_path=source_path)


def validate_prompt_placeholders(prompt: PromptConfig) -> None:
    base_allowed = {
        "code_length",
        "symbol_count",
        "symbols",
        "initial_query_credits",
        "beneficiary_clause",
        "repetition_rule",
        "code_space",
    }
    invalid_allowed = {"code_length", "credits_remaining"}
    _validate_template_fields(
        prompt.base_template,
        base_allowed,
        "prompt.base_template",
        required={"code_length", "symbol_count", "symbols", "initial_query_credits", "beneficiary_clause"},
    )
    _validate_template_fields(
        prompt.invalid_response_template,
        invalid_allowed,
        "prompt.invalid_response_template",
        required=invalid_allowed,
    )


def _validate_template_fields(template: str, allowed: set[str], label: str, required: set[str]) -> None:
    fields = {name for _, name, _, _ in Formatter().parse(template) if name}
    unknown = fields - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown placeholders: {', '.join(sorted(unknown))}")
    missing = required - fields
    if missing:
        raise ValueError(f"{label} is missing required placeholders: {', '.join(sorted(missing))}")


def without_none(value: Any) -> Any:
    """Recursively remove null mapping values while preserving list positions."""
    if isinstance(value, dict):
        return {key: without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_none(item) for item in value]
    return value
