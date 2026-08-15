"""One independent model/game/track conversation state machine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span

from llm_future_affinity.config import AppConfig, ModelConfig
from llm_future_affinity.domain import Action, GameRecord, HttpAttempt, RunStatus, SubmissionType, Track, Usage
from llm_future_affinity.game import calculate_feedback, score_submission
from llm_future_affinity.manifest import utc_now
from llm_future_affinity.openrouter import OpenRouterClient, OpenRouterError
from llm_future_affinity.persistence import OutputRow
from llm_future_affinity.prompting import prompt_hash, render_initial_prompt
from llm_future_affinity.protocol import parse_command, render_feedback, render_invalid_correction
from llm_future_affinity.telemetry import Telemetry

INVALID_RESPONSE_LIMIT = 3


@dataclass(frozen=True, slots=True)
class StepOutcome:
    terminal: bool
    open_breaker: bool = False


class ConversationSession:
    def __init__(
        self,
        config: AppConfig,
        model_key: str,
        game: GameRecord,
        track: Track,
        attempt_number: int,
        supersedes_attempt_id: str | None,
        client: OpenRouterClient,
        telemetry: Telemetry,
        invocation_span: Span,
    ) -> None:
        self.config = config
        self.model_key = model_key
        self.model = config.model_for(model_key)
        self.game = game
        self.track = track
        self.attempt_number = attempt_number
        self.supersedes_attempt_id = supersedes_attempt_id
        self.client = client
        self.telemetry = telemetry
        self.invocation_span = invocation_span

        self.attempt_id = str(uuid.uuid4())
        self.initial_prompt = render_initial_prompt(config.game, config.prompt, track)
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": self.initial_prompt}]
        self.started = False
        self.terminal = False
        self.started_at = ""
        self.finished_at = ""
        self._started_clock = 0.0
        self.conversation_span: Span | None = None

        self.queries_used = 0
        self.credits_remaining = config.game.initial_query_credits
        self.consecutive_invalid = 0
        self.num_invalid_responses = 0
        self.num_model_calls = 0
        self.attempts: list[HttpAttempt] = []
        self.total_usage = Usage()
        self.interaction_trace: list[dict[str, Any]] = []

        self.final_answer: str | None = None
        self.positions_correct: int | None = None
        self.final_score: float | None = None
        self.solved: bool | None = None
        self.submission_type: SubmissionType | None = None
        self.run_status: RunStatus | None = None
        self.error_category: str | None = None
        self.error_message: str | None = None
        self.parse_error: str | None = None
        self.cache_hit_detected = False
        self.observed_provider: str | None = None
        self.observed_endpoint: str | None = None
        self.observed_quantization: str | None = None

    async def step(self) -> StepOutcome:
        if self.terminal:
            raise RuntimeError("cannot advance a terminal conversation")
        self._start_if_needed()
        max_calls = self.config.execution.limits.max_model_calls_per_conversation
        if max_calls is not None and self.num_model_calls >= max_calls:
            self.finish(RunStatus.CANCELLED, "model_call_limit", "conversation model-call limit reached")
            return StepOutcome(terminal=True, open_breaker=True)

        self.num_model_calls += 1
        attributes = self._audit_attributes()
        try:
            reply = await self.client.complete(self.messages, self._span(), attributes)
        except OpenRouterError as error:
            self._absorb_attempts(error.attempts)
            category = error.attempts[-1].error_category if error.attempts else None
            self.finish(RunStatus.API_ERROR, category or type(error).__name__, str(error))
            return StepOutcome(terminal=True, open_breaker=True)

        self._absorb_attempts(reply.attempts)
        self.observed_provider = reply.observed_provider or self.observed_provider
        self.observed_endpoint = reply.observed_endpoint or self.observed_endpoint
        self.observed_quantization = reply.observed_quantization or self.observed_quantization

        trace_entry = self._trace_entry(reply.attempts)
        if reply.cache_status == "HIT":
            self.cache_hit_detected = True
            trace_entry["response_cache_status"] = "HIT"
            self.interaction_trace.append(trace_entry)
            self.finish(RunStatus.CACHE_HIT, "response_cache_hit", "OpenRouter returned a response-cache hit")
            return StepOutcome(terminal=True)

        mismatch = self._routing_mismatch()
        if mismatch is not None:
            self.interaction_trace.append(trace_entry)
            self.finish(RunStatus.PROVIDER_MISMATCH, "routing_mismatch", mismatch)
            return StepOutcome(terminal=True, open_breaker=True)

        assistant = _assistant_history(reply.assistant_message, reply.content)
        self.messages.append(assistant)
        parsed = parse_command(reply.content, self.config.game)
        trace_entry["raw_action_valid"] = parsed.valid
        if not parsed.valid:
            self.consecutive_invalid += 1
            self.num_invalid_responses += 1
            self.parse_error = parsed.error_message
            trace_entry.update(
                {
                    "action": None,
                    "parse_error": parsed.error_message,
                    "consecutive_invalid_responses": self.consecutive_invalid,
                }
            )
            self.interaction_trace.append(trace_entry)
            if self.consecutive_invalid >= INVALID_RESPONSE_LIMIT:
                self.finish(RunStatus.INVALID_SUBMISSION, "invalid_format", parsed.error_message)
                return StepOutcome(terminal=True)
            self.messages.append(
                {
                    "role": "user",
                    "content": render_invalid_correction(
                        self.config.prompt,
                        self.config.game,
                        self.credits_remaining,
                    ),
                }
            )
            return StepOutcome(terminal=False)

        assert parsed.action is not None and parsed.value is not None
        self.consecutive_invalid = 0
        trace_entry["action_received"] = parsed.action.value
        trace_entry["value"] = parsed.value

        if parsed.action is Action.QUERY and self.credits_remaining > 0:
            self.queries_used += 1
            self.credits_remaining -= 1
            feedback = calculate_feedback(self.game.hidden_code, parsed.value)
            trace_entry.update(
                {
                    "action_applied": Action.QUERY.value,
                    "feedback": {"exact": feedback.exact, "misplaced": feedback.misplaced},
                    "credits_remaining": self.credits_remaining,
                }
            )
            self.interaction_trace.append(trace_entry)
            self.messages.append(
                {
                    "role": "user",
                    "content": render_feedback(self.queries_used, feedback, self.credits_remaining),
                }
            )
            return StepOutcome(terminal=False)

        trace_entry["action_applied"] = Action.SUBMIT.value
        trace_entry["coerced_at_zero_credits"] = parsed.action is Action.QUERY
        trace_entry["credits_remaining"] = self.credits_remaining
        self.interaction_trace.append(trace_entry)
        self._apply_submission(parsed.value)
        self.finish(RunStatus.COMPLETED)
        return StepOutcome(terminal=True)

    def cancel(self, status: RunStatus, category: str, message: str) -> None:
        if not self.terminal:
            self._start_if_needed()
            self.finish(status, category, message)

    def finish(
        self,
        status: RunStatus,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if self.terminal:
            return
        self.terminal = True
        self.run_status = status
        self.error_category = error_category
        self.error_message = error_message
        self.finished_at = utc_now()
        if self.conversation_span is not None:
            self.telemetry.end_span(
                self.conversation_span,
                RuntimeError(error_message) if error_message and status is not RunStatus.COMPLETED else None,
            )

    def to_output_row(self) -> OutputRow:
        if not self.terminal or self.run_status is None:
            raise RuntimeError("conversation must be terminal before creating an output row")
        trace_id, span_id = self.telemetry.span_ids(self._span())
        status = self.run_status
        eligible = status is RunStatus.COMPLETED
        exclusion_reasons = [] if eligible else [status.value]
        token_complete, cost_complete = self._accounting_completeness()
        requested_quantization = self.model.routing.quantizations
        return OutputRow(
            run_id=self.attempt_id,
            attempt_id=self.attempt_id,
            attempt_number=self.attempt_number,
            supersedes_attempt_id=self.supersedes_attempt_id,
            game_id=self.game.game_id,
            model_key=self.model_key,
            model_family=self.model.model_family,
            model_id=self.model.model_id,
            requested_provider=self.model.routing.endpoint_slug.split("/", 1)[0],
            observed_provider=self.observed_provider,
            requested_endpoint=self.model.routing.endpoint_slug,
            observed_endpoint=self.observed_endpoint,
            requested_quantization=",".join(requested_quantization) if requested_quantization else None,
            observed_quantization=self.observed_quantization,
            track=self.track,
            i2_identity=self.config.prompt.tracks[self.track].i2_identity,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_ms=round((time.perf_counter() - self._started_clock) * 1000),
            trace_id=trace_id,
            span_id=span_id,
            hidden_code=self.game.hidden_code,
            code_length=self.config.game.code_length,
            symbol_set_size=len(self.config.game.symbols),
            initial_query_credits=self.config.game.initial_query_credits,
            queries_used=self.queries_used,
            credits_remaining=self.credits_remaining,
            final_answer=self.final_answer,
            positions_correct=self.positions_correct,
            final_score=self.final_score,
            solved=self.solved,
            submission_type=self.submission_type,
            num_model_calls=self.num_model_calls,
            num_http_attempts=len(self.attempts),
            num_transport_retries=sum(attempt.attempt_number > 1 for attempt in self.attempts),
            num_invalid_responses=self.num_invalid_responses,
            total_input_tokens=self.total_usage.input_tokens,
            total_output_tokens=self.total_usage.output_tokens,
            total_reasoning_tokens=self.total_usage.reasoning_tokens,
            total_cached_tokens=self.total_usage.cached_tokens,
            total_cache_write_tokens=self.total_usage.cache_write_tokens,
            total_tokens=self.total_usage.total_tokens,
            total_cost_usd=self.total_usage.cost_usd,
            token_totals_complete=token_complete,
            cost_total_complete=cost_complete,
            cache_hit_detected=self.cache_hit_detected,
            prompt_hash=prompt_hash(self.initial_prompt),
            inference_settings=_inference_settings(self.model),
            routing_settings=self.model.routing.model_dump(mode="json"),
            run_status=status,
            analysis_eligible=eligible,
            exclusion_reasons=exclusion_reasons,
            error_category=self.error_category,
            error_message=self.error_message,
            parse_error=self.parse_error,
            interaction_trace=self.interaction_trace,
        )

    def debug_record(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "model_key": self.model_key,
            "game_id": self.game.game_id,
            "track": self.track.value,
            "run_status": self.run_status.value if self.run_status else None,
            "http_attempts": [
                {
                    "attempt_number": attempt.attempt_number,
                    "request": attempt.raw_request,
                    "response": attempt.raw_response,
                    "error_category": attempt.error_category,
                    "error_message": attempt.error_message,
                }
                for attempt in self.attempts
            ],
        }

    def _start_if_needed(self) -> None:
        if self.started:
            return
        self.started = True
        self.started_at = utc_now()
        self._started_clock = time.perf_counter()
        self.conversation_span = self.telemetry.start_span(
            "experiment.conversation",
            self.invocation_span,
            self._audit_attributes(),
        )

    def _span(self) -> Span:
        return self.conversation_span or self.invocation_span

    def _audit_attributes(self) -> dict[str, Any]:
        return {
            "experiment.attempt_id": self.attempt_id,
            "experiment.model_key": self.model_key,
            "experiment.game_id": self.game.game_id,
            "experiment.track": self.track.value,
            "experiment.model_call": self.num_model_calls,
        }

    def _absorb_attempts(self, attempts: list[HttpAttempt]) -> None:
        self.attempts.extend(attempts)
        for attempt in attempts:
            self.total_usage.add(attempt.usage)

    def _trace_entry(self, attempts: list[HttpAttempt]) -> dict[str, Any]:
        return {
            "step": self.num_model_calls,
            "credits_before": self.credits_remaining,
            "http_attempts": [_attempt_summary(attempt) for attempt in attempts],
            "response_cache_status": attempts[-1].cache_status if attempts else "UNKNOWN",
        }

    def _routing_mismatch(self) -> str | None:
        expected_endpoint = self.model.routing.endpoint_slug
        expected_provider = expected_endpoint.split("/", 1)[0]
        if self.observed_endpoint is not None and self.observed_endpoint != expected_endpoint:
            return f"expected endpoint {expected_endpoint!r}, observed {self.observed_endpoint!r}"
        if (
            self.observed_endpoint is None
            and self.observed_provider is not None
            and _routing_token(self.observed_provider.split("/", 1)[0]) != _routing_token(expected_provider)
        ):
            return f"expected provider {expected_provider!r}, observed {self.observed_provider!r}"
        allowed = self.model.routing.quantizations
        if allowed and self.observed_quantization is not None and self.observed_quantization not in allowed:
            return f"expected quantization in {allowed!r}, observed {self.observed_quantization!r}"
        return None

    def _apply_submission(self, answer: str) -> None:
        positions, score, solved, submission_type = score_submission(self.game.hidden_code, answer)
        self.final_answer = answer
        self.positions_correct = positions
        self.final_score = score
        self.solved = solved
        self.submission_type = submission_type

    def _accounting_completeness(self) -> tuple[bool, bool]:
        unknown_transport = any(attempt.error_category in {"timeout", "transport_error"} for attempt in self.attempts)
        successful = [
            attempt for attempt in self.attempts if attempt.status_code is not None and attempt.status_code < 400
        ]
        tokens_complete = not unknown_transport and all(
            attempt.usage.total_tokens is not None for attempt in successful
        )
        cost_complete = not unknown_transport and all(attempt.usage.cost_usd is not None for attempt in successful)
        return tokens_complete, cost_complete


def _assistant_history(message: dict[str, Any], content: str) -> dict[str, Any]:
    history: dict[str, Any] = {"role": "assistant", "content": content}
    for field in ("reasoning_details", "thought_signature", "thought_signatures"):
        if field in message:
            history[field] = message[field]
    return history


def _attempt_summary(attempt: HttpAttempt) -> dict[str, Any]:
    return {
        "attempt_number": attempt.attempt_number,
        "started_at": attempt.started_at,
        "finished_at": attempt.finished_at,
        "duration_ms": attempt.duration_ms,
        "status_code": attempt.status_code,
        "error_category": attempt.error_category,
        "error_message": attempt.error_message,
        "usage": {
            "input_tokens": attempt.usage.input_tokens,
            "output_tokens": attempt.usage.output_tokens,
            "reasoning_tokens": attempt.usage.reasoning_tokens,
            "cached_tokens": attempt.usage.cached_tokens,
            "cache_write_tokens": attempt.usage.cache_write_tokens,
            "total_tokens": attempt.usage.total_tokens,
        },
        "cost_usd": attempt.usage.cost_usd,
        "generation_id": attempt.generation_id,
        "response_cache_status": attempt.cache_status,
        "observed_provider": attempt.observed_provider,
        "observed_endpoint": attempt.observed_endpoint,
        "observed_quantization": attempt.observed_quantization,
    }


def _routing_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _inference_settings(model: ModelConfig) -> dict[str, Any]:
    settings = model.inference.model_dump(mode="json")
    if model.custom_options:
        settings["custom_options"] = model.custom_options
    return settings
