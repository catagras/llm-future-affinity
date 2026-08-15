from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
import respx
from opentelemetry import trace
from opentelemetry.trace import Span

from llm_future_affinity.config import AppConfig
from llm_future_affinity.domain import Usage
from llm_future_affinity.openrouter import (
    ApiError,
    OpenRouterClient,
    RetryExhausted,
    merge_usage,
    normalize_usage,
)
from llm_future_affinity.telemetry import NullTelemetry, sanitize_for_audit


class RecordingTelemetry(NullTelemetry):
    def __init__(self) -> None:
        self.audits: list[dict[str, Any]] = []

    def record_audit(self, span: Span, body: dict[str, Any], attributes: Mapping[str, Any]) -> None:
        del span, attributes
        self.audits.append(body)


def success_response(content: str = "SUBMIT ABCD") -> dict[str, Any]:
    return {
        "id": "gen-1",
        "provider": "test-provider",
        "provider_endpoint": "test-provider/exact",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "cost": 0.01,
            "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


def make_client(
    app_config: AppConfig,
    telemetry: RecordingTelemetry | None = None,
    sleep: Any = None,
    debug: bool = False,
    clock: Any = None,
) -> OpenRouterClient:
    async def no_sleep(_: float) -> None:
        return None

    return OpenRouterClient(
        "secret",
        app_config.model_for("test-model"),
        app_config.execution,
        telemetry or RecordingTelemetry(),
        sleep=sleep or no_sleep,
        base_url="https://openrouter.test/api/v1",
        debug=debug,
        clock=clock or time.monotonic,
    )


@respx.mock
async def test_preflight_validates_model_and_exact_endpoint(app_config: AppConfig) -> None:
    respx.get("https://openrouter.test/api/v1/model/test/model").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"supported_parameters": ["max_tokens", "temperature", "top_p", "reasoning"]}},
        )
    )
    respx.get("https://openrouter.test/api/v1/models/test/model/endpoints").mock(
        return_value=httpx.Response(200, json={"data": {"endpoints": [{"provider_slug": "test-provider/exact"}]}})
    )
    client = make_client(app_config)
    await client.preflight()
    await client.close()


@respx.mock
async def test_preflight_rejects_unsupported_parameter(app_config: AppConfig) -> None:
    respx.get("https://openrouter.test/api/v1/model/test/model").mock(
        return_value=httpx.Response(200, json={"data": {"supported_parameters": ["max_tokens"]}})
    )
    client = make_client(app_config)
    with pytest.raises(ValueError, match="temperature"):
        await client.preflight()
    await client.close()


@respx.mock
async def test_preflight_rejects_missing_endpoint(app_config: AppConfig) -> None:
    respx.get("https://openrouter.test/api/v1/model/test/model").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"supported_parameters": ["max_tokens", "temperature", "top_p", "reasoning"]}},
        )
    )
    respx.get("https://openrouter.test/api/v1/models/test/model/endpoints").mock(
        return_value=httpx.Response(200, json={"data": {"endpoints": [{"provider_slug": "other"}]}})
    )
    client = make_client(app_config)
    with pytest.raises(ValueError, match="not available"):
        await client.preflight()
    await client.close()


@respx.mock
async def test_preflight_rejects_endpoint_parameter_and_quantization_mismatch(app_config: AppConfig) -> None:
    app_config.models["test-model"].routing.quantizations = ["fp8"]
    respx.get("https://openrouter.test/api/v1/model/test/model").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"supported_parameters": ["max_tokens", "temperature", "top_p", "reasoning"]}},
        )
    )
    endpoint_route = respx.get("https://openrouter.test/api/v1/models/test/model/endpoints")
    endpoint_route.mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "provider_slug": "test-provider/exact",
                            "supported_parameters": ["max_tokens"],
                            "quantization": "fp16",
                        }
                    ]
                }
            },
        )
    )
    client = make_client(app_config)
    with pytest.raises(ValueError, match="endpoint does not support"):
        await client.preflight()

    endpoint_route.mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "endpoints": [
                        {
                            "provider_slug": "test-provider/exact",
                            "supported_parameters": ["max_tokens", "temperature", "top_p", "reasoning"],
                            "quantization": "fp16",
                        }
                    ]
                }
            },
        )
    )
    with pytest.raises(ValueError, match="quantization"):
        await client.preflight()
    await client.close()


def test_build_payload_pins_routing_and_drops_nulls(app_config: AppConfig) -> None:
    client = make_client(app_config)
    payload = client.build_payload([{"role": "user", "content": "hello"}])
    assert payload["provider"] == {
        "order": ["test-provider/exact"],
        "only": ["test-provider/exact"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert payload["temperature"] == 0
    assert "thinking" not in payload
    assert payload["reasoning"]["effort"] == "low"


@respx.mock
async def test_successful_completion_records_usage_and_audit(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=success_response(), headers={"X-OpenRouter-Cache-Status": "MISS"})
    )
    telemetry = RecordingTelemetry()
    client = make_client(app_config, telemetry, debug=True)
    reply = await client.complete([{"role": "user", "content": "prompt"}], trace.INVALID_SPAN, {"game.id": 1})
    assert route.call_count == 1
    assert reply.content == "SUBMIT ABCD"
    assert reply.usage.reasoning_tokens == 1
    assert reply.usage.cached_tokens == 2
    assert reply.observed_endpoint == "test-provider/exact"
    assert reply.attempts[0].raw_response is not None
    assert telemetry.audits[0]["request"]["model"] == "test/model"
    await client.close()


@respx.mock
async def test_retries_429_then_succeeds(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=success_response()),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    client = make_client(app_config, sleep=record_sleep)
    reply = await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 2
    assert len(reply.attempts) == 2
    assert reply.attempts[0].error_category == "rate_limit"
    assert delays == [0.001]
    await client.close()


@respx.mock
async def test_model_rpm_paces_request_attempts(app_config: AppConfig) -> None:
    app_config.models["test-model"].rpm = 2
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=success_response()),
            httpx.Response(200, json=success_response()),
            httpx.Response(200, json=success_response()),
        ]
    )
    current = 0.0
    delays: list[float] = []

    def clock() -> float:
        return current

    async def advance(delay: float) -> None:
        nonlocal current
        delays.append(delay)
        current += delay

    client = make_client(app_config, sleep=advance, clock=clock)
    await client.complete([], trace.INVALID_SPAN, {})
    await client.complete([], trace.INVALID_SPAN, {})
    await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 3
    assert delays == [30.0, 30.0]
    await client.close()


@respx.mock
async def test_retry_exhaustion_has_four_attempts(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": "down"})
    )
    client = make_client(app_config)
    with pytest.raises(RetryExhausted) as captured:
        await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 4
    assert len(captured.value.attempts) == 4
    assert all(item.error_category == "server_error" for item in captured.value.attempts)
    await client.close()


@respx.mock
async def test_timeout_is_retried(app_config: AppConfig) -> None:
    request = httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        side_effect=[httpx.ReadTimeout("late", request=request), httpx.Response(200, json=success_response())]
    )
    client = make_client(app_config)
    reply = await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 2
    assert reply.attempts[0].error_category == "timeout"
    await client.close()


@respx.mock
async def test_nonretryable_error_fails_immediately(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad payload"}})
    )
    telemetry = RecordingTelemetry()
    client = make_client(app_config, telemetry=telemetry)
    with pytest.raises(ApiError, match="bad payload") as captured:
        await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 1
    assert captured.value.attempts[0].error_category == "client_error"
    assert telemetry.audits[0]["error"]["category"] == "client_error"
    await client.close()


@respx.mock
async def test_http_408_is_retried_as_timeout(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        side_effect=[httpx.Response(408), httpx.Response(200, json=success_response())]
    )
    client = make_client(app_config)
    reply = await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 2
    assert reply.attempts[0].error_category == "timeout"
    await client.close()


@respx.mock
async def test_nonretryable_error_retains_prior_retry_attempts(app_config: AppConfig) -> None:
    route = respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(400, json={"error": {"message": "bad payload"}}),
        ]
    )
    client = make_client(app_config)
    with pytest.raises(ApiError) as captured:
        await client.complete([], trace.INVALID_SPAN, {})
    assert route.call_count == 2
    assert [item.attempt_number for item in captured.value.attempts] == [1, 2]
    await client.close()


@respx.mock
async def test_malformed_success_is_api_error(app_config: AppConfig) -> None:
    respx.post("https://openrouter.test/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"bad": 1})
    )
    client = make_client(app_config)
    with pytest.raises(ApiError, match="malformed"):
        await client.complete([], trace.INVALID_SPAN, {})
    await client.close()


@respx.mock
async def test_generation_metadata_fills_missing_cost(app_config: AppConfig) -> None:
    body = success_response()
    body.pop("provider")
    body.pop("provider_endpoint")
    body["usage"].pop("cost")
    respx.post("https://openrouter.test/api/v1/chat/completions").mock(return_value=httpx.Response(200, json=body))
    metadata = respx.get("https://openrouter.test/api/v1/generation").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"provider_name": "test-provider", "endpoint": "test-provider/exact", "cost": 0.02}},
        )
    )
    client = make_client(app_config)
    reply = await client.complete([], trace.INVALID_SPAN, {})
    assert metadata.called
    assert reply.usage.cost_usd == 0.02
    assert reply.observed_provider == "test-provider"
    await client.close()


def test_usage_normalization_and_merge() -> None:
    usage = normalize_usage(
        {
            "input_tokens": 4,
            "output_tokens": 2,
            "reasoning_tokens": 1,
            "cached_input_tokens": 3,
            "cache_write_tokens": 2,
            "total_tokens": 6,
        }
    )
    assert usage == Usage(4, 2, 1, 3, 2, 6, None)
    assert merge_usage(Usage(input_tokens=4), Usage(input_tokens=9, cost_usd=0.3)) == Usage(
        input_tokens=4,
        cost_usd=0.3,
    )
    assert normalize_usage(None) == Usage()


def test_usage_adds_known_values() -> None:
    total = Usage(input_tokens=1, cost_usd=0.1)
    total.add(Usage(input_tokens=2, output_tokens=3, cost_usd=0.2))
    assert total.input_tokens == 3
    assert total.output_tokens == 3
    assert total.cost_usd == pytest.approx(0.3)


def test_sanitize_removes_secrets_and_plaintext_reasoning() -> None:
    value = {
        "Authorization": "secret",
        "messages": [
            {
                "content": "answer",
                "reasoning": "private",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "private", "signature": "sig"},
                    {"type": "reasoning.encrypted", "data": "opaque"},
                ],
            }
        ],
    }
    sanitized = sanitize_for_audit(value)
    encoded = json.dumps(sanitized)
    assert "secret" not in encoded
    assert "private" not in encoded
    assert "sig" in encoded
    assert "opaque" in encoded


async def test_null_telemetry_is_noop() -> None:
    telemetry = NullTelemetry()
    await telemetry.preflight()
    span = telemetry.start_span("test")
    telemetry.record_audit(span, {}, {})
    telemetry.end_span(span)
    assert telemetry.span_ids(span) == (None, None)
    assert await telemetry.force_flush()
    await telemetry.shutdown()
