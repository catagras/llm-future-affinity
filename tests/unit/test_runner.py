from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from opentelemetry import trace
from opentelemetry.trace import Span

from llm_future_affinity.config import AppConfig
from llm_future_affinity.domain import GameRecord, HttpAttempt, ModelReply, RunStatus, Track, Usage
from llm_future_affinity.openrouter import OpenRouterClient, RetryExhausted
from llm_future_affinity.runner import ConversationSession
from llm_future_affinity.telemetry import NullTelemetry


def attempt(number: int = 1, *, timeout: bool = False) -> HttpAttempt:
    return HttpAttempt(
        attempt_number=number,
        started_at="2026-01-01T00:00:00.000Z",
        finished_at="2026-01-01T00:00:00.100Z",
        duration_ms=100,
        status_code=None if timeout else 200,
        error_category="timeout" if timeout else None,
        error_message="late" if timeout else None,
        usage=Usage() if timeout else Usage(input_tokens=10, output_tokens=2, total_tokens=12, cost_usd=0.01),
        generation_id=None if timeout else f"gen-{number}",
        cache_status="MISS",
        observed_provider="test-provider",
        observed_endpoint="test-provider/exact",
    )


def reply(content: str, *, cache: str = "MISS", endpoint: str = "test-provider/exact") -> ModelReply:
    item = attempt()
    item.cache_status = cache
    item.observed_endpoint = endpoint
    return ModelReply(
        content=content,
        assistant_message={
            "role": "assistant",
            "content": content,
            "reasoning": "do not persist",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}],
            "thought_signature": "signed",
        },
        attempts=[item],
        usage=item.usage,
        generation_id=item.generation_id,
        cache_status=cache,
        observed_provider=item.observed_provider,
        observed_endpoint=endpoint,
        observed_quantization=None,
    )


class FakeClient:
    def __init__(self, replies: list[ModelReply | Exception]) -> None:
        self.replies = replies
        self.messages: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        parent_span: Span,
        audit_attributes: Mapping[str, Any],
    ) -> ModelReply:
        del parent_span, audit_attributes
        self.messages.append([dict(message) for message in messages])
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def session(app_config: AppConfig, client: FakeClient, track: Track = Track.A) -> ConversationSession:
    telemetry = NullTelemetry()
    return ConversationSession(
        app_config,
        "test-model",
        GameRecord(1, "ABCD"),
        track,
        1,
        None,
        cast(OpenRouterClient, client),
        telemetry,
        trace.INVALID_SPAN,
    )


async def run_to_terminal(value: ConversationSession) -> None:
    while not value.terminal:
        await value.step()


async def test_query_then_partial_submission(app_config: AppConfig) -> None:
    value = session(app_config, FakeClient([reply("QUERY AABC"), reply("SUBMIT ABCA")]))
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.run_status is RunStatus.COMPLETED
    assert row.analysis_eligible
    assert row.submission_type == "partial"
    assert row.final_score == 0.75
    assert row.queries_used == 1
    assert row.credits_remaining == 1
    assert row.interaction_trace[0]["feedback"] == {"exact": 1, "misplaced": 2}


async def test_exact_zero_query_submission(app_config: AppConfig) -> None:
    value = session(app_config, FakeClient([reply("SUBMIT ABCD")]))
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.submission_type == "exact"
    assert row.solved is True
    assert row.final_score == 1
    assert row.queries_used == 0


async def test_invalid_counter_resets_after_valid_query(app_config: AppConfig) -> None:
    client = FakeClient(
        [
            reply("bad"),
            reply("still bad"),
            reply("QUERY AABC"),
            reply("bad"),
            reply("bad"),
            reply("bad"),
        ]
    )
    value = session(app_config, client)
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.run_status is RunStatus.INVALID_SUBMISSION
    assert row.final_answer is None
    assert row.submission_type is None
    assert row.num_invalid_responses == 5
    assert "Credits remaining: 2" in client.messages[1][-1]["content"]
    assert "Credits remaining: 1" in client.messages[4][-1]["content"]


async def test_zero_credit_query_is_coerced_to_submit(app_config: AppConfig) -> None:
    app_config.game.initial_query_credits = 1
    value = session(app_config, FakeClient([reply("QUERY AABC"), reply("QUERY DCBA")]))
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.run_status is RunStatus.COMPLETED
    assert row.final_answer == "DCBA"
    assert row.queries_used == 1
    assert row.interaction_trace[-1]["action_received"] == "QUERY"
    assert row.interaction_trace[-1]["action_applied"] == "SUBMIT"
    assert row.interaction_trace[-1]["coerced_at_zero_credits"] is True


async def test_cache_hit_is_excluded(app_config: AppConfig) -> None:
    value = session(app_config, FakeClient([reply("SUBMIT ABCD", cache="HIT")]))
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.run_status is RunStatus.CACHE_HIT
    assert row.cache_hit_detected
    assert not row.analysis_eligible


async def test_endpoint_mismatch_opens_breaker(app_config: AppConfig) -> None:
    value = session(app_config, FakeClient([reply("SUBMIT ABCD", endpoint="other")]))
    outcome = await value.step()
    assert outcome.open_breaker
    assert value.to_output_row().run_status is RunStatus.PROVIDER_MISMATCH


async def test_retry_error_is_saved_and_opens_breaker(app_config: AppConfig) -> None:
    error = RetryExhausted("down", [attempt(timeout=True)])
    value = session(app_config, FakeClient([error]))
    outcome = await value.step()
    row = value.to_output_row()
    assert outcome.open_breaker
    assert row.run_status is RunStatus.API_ERROR
    assert row.num_http_attempts == 1
    assert row.error_category == "timeout"
    assert not row.token_totals_complete
    assert not row.cost_total_complete


async def test_transport_retry_count_counts_retried_attempts_once(app_config: AppConfig) -> None:
    final_reply = reply("SUBMIT ABCD")
    final_reply.attempts = [attempt(1, timeout=True), attempt(2, timeout=True), attempt(3)]
    value = session(app_config, FakeClient([final_reply]))
    await run_to_terminal(value)
    row = value.to_output_row()
    assert row.num_http_attempts == 3
    assert row.num_transport_retries == 2


async def test_reasoning_details_are_preserved_but_plaintext_is_not(app_config: AppConfig) -> None:
    client = FakeClient([reply("QUERY AABC"), reply("SUBMIT ABCD")])
    value = session(app_config, client)
    await run_to_terminal(value)
    assistant = client.messages[1][1]
    assert assistant["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "opaque"}]
    assert assistant["thought_signature"] == "signed"
    assert "reasoning" not in assistant


async def test_model_call_safety_limit(app_config: AppConfig) -> None:
    app_config.execution.limits.max_model_calls_per_conversation = 1
    value = session(app_config, FakeClient([reply("QUERY AABC")]))
    assert not (await value.step()).terminal
    outcome = await value.step()
    assert outcome.open_breaker
    assert value.to_output_row().error_category == "model_call_limit"


def test_nonterminal_row_is_rejected(app_config: AppConfig) -> None:
    value = session(app_config, FakeClient([]))
    with pytest.raises(RuntimeError, match="terminal"):
        value.to_output_row()
