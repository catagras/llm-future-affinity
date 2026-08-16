"""Minimal async OpenRouter client with routing, retries, usage, and auditing."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx
from opentelemetry.trace import Span

from llm_future_affinity.config import ExecutionConfig, ModelConfig, without_none
from llm_future_affinity.domain import HttpAttempt, ModelReply, Usage
from llm_future_affinity.telemetry import Telemetry, sanitize_for_audit

BASE_URL = "https://openrouter.ai/api/v1"
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, attempts: list[HttpAttempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


class RetryExhausted(OpenRouterError):
    pass


class ApiError(OpenRouterError):
    pass


class _RequestRateLimiter:
    """Pace request starts for one model client when an RPM cap is configured."""

    def __init__(self, rpm: int | None, sleep: Sleep, clock: Clock = time.monotonic) -> None:
        self._interval = 60.0 / rpm if rpm is not None else None
        self._sleep = sleep
        self._clock = clock
        self._next_request_at: float | None = None
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._interval is None:
            return
        async with self._lock:
            now = self._clock()
            if self._next_request_at is not None:
                delay = self._next_request_at - now
                if delay > 0:
                    await self._sleep(delay)
            now = self._clock()
            self._next_request_at = max(self._next_request_at or now, now) + self._interval


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: ModelConfig,
        execution: ExecutionConfig,
        telemetry: Telemetry,
        *,
        http_client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        random_source: random.Random | None = None,
        base_url: str = BASE_URL,
        debug: bool = False,
        clock: Clock = time.monotonic,
    ) -> None:
        self.model = model
        self.execution = execution
        self.telemetry = telemetry
        self.sleep = sleep
        self.random: random.Random = random_source or random.Random()
        self.base_url = base_url.rstrip("/")
        self.debug = debug
        self._rate_limiter = _RequestRateLimiter(model.rpm, sleep, clock)
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient(timeout=execution.request_timeout_seconds)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Cache": "false",
        }

    async def close(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def preflight(self) -> None:
        author, separator, slug = self.model.model_id.partition("/")
        if not separator or not author or not slug:
            raise ValueError(f"invalid OpenRouter model ID: {self.model.model_id!r}")
        model_response = await self.http.get(
            f"{self.base_url}/model/{author}/{slug}",
            headers={"Authorization": self.headers["Authorization"]},
        )
        model_response.raise_for_status()
        model_data = _data_object(model_response.json())
        supported = set(model_data.get("supported_parameters") or [])
        required = self._configured_parameters()
        unsupported = required - supported
        if unsupported:
            raise ValueError(f"model does not support configured parameters: {', '.join(sorted(unsupported))}")

        endpoints_response = await self.http.get(
            f"{self.base_url}/models/{author}/{slug}/endpoints",
            headers={"Authorization": self.headers["Authorization"]},
        )
        endpoints_response.raise_for_status()
        endpoints_data = _data_object(endpoints_response.json())
        endpoints = endpoints_data.get("endpoints") or []
        configured = self.model.routing.endpoint_slug
        matching = [
            endpoint for endpoint in endpoints if isinstance(endpoint, dict) and configured in _endpoint_slugs(endpoint)
        ]
        if not matching:
            raise ValueError(f"configured endpoint {configured!r} is not available for {self.model.model_id}")
        endpoint_supported = set(matching[0].get("supported_parameters") or [])
        if endpoint_supported and (endpoint_unsupported := required - endpoint_supported):
            raise ValueError(
                f"endpoint does not support configured parameters: {', '.join(sorted(endpoint_unsupported))}"
            )
        if self.model.routing.quantizations:
            advertised = _endpoint_quantizations(matching[0])
            if advertised and not advertised.intersection(self.model.routing.quantizations):
                raise ValueError(
                    f"endpoint quantization {sorted(advertised)!r} does not match configured filter "
                    f"{self.model.routing.quantizations!r}"
                )

    def build_payload(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        inference = without_none(self.model.inference.model_dump(mode="json"))
        custom_options = without_none(self.model.custom_options)
        routing: dict[str, Any] = {
            "order": [self.model.routing.endpoint_slug],
            "only": [self.model.routing.endpoint_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if self.model.routing.quantizations:
            routing["quantizations"] = self.model.routing.quantizations
        return {
            "model": self.model.model_id,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "provider": routing,
            **inference,
            **custom_options,
        }

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        parent_span: Span,
        audit_attributes: Mapping[str, Any],
    ) -> ModelReply:
        payload = self.build_payload(messages)
        call_span = self.telemetry.start_span("openrouter.logical_call", parent_span, audit_attributes)
        attempts: list[HttpAttempt] = []
        try:
            for attempt_number in range(1, self.execution.retry.max_attempts + 1):
                try:
                    result = await self._attempt(payload, attempt_number, call_span, audit_attributes)
                except ApiError as error:
                    error.attempts[:0] = attempts
                    raise
                attempts.append(result[0])
                response = result[1]
                if response is not None:
                    return await self._model_reply(response, attempts)
                if attempt_number < self.execution.retry.max_attempts:
                    await self.sleep(self._retry_delay(attempt_number))
            raise RetryExhausted(
                f"OpenRouter request failed after {self.execution.retry.max_attempts} attempts",
                attempts,
            )
        except BaseException as error:
            self.telemetry.end_span(call_span, error)
            raise
        finally:
            if call_span.is_recording():
                self.telemetry.end_span(call_span)

    async def _attempt(
        self,
        payload: dict[str, Any],
        attempt_number: int,
        parent_span: Span,
        audit_attributes: Mapping[str, Any],
    ) -> tuple[HttpAttempt, httpx.Response | None]:
        started_clock = time.perf_counter()
        started_at = _utc_now()
        span = self.telemetry.start_span(
            "openrouter.http_attempt",
            parent_span,
            {**audit_attributes, "http.attempt": attempt_number},
        )
        response: httpx.Response | None = None
        error_category: str | None = None
        error_message: str | None = None
        try:
            await self._rate_limiter.wait()
            response = await self.http.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.execution.request_timeout_seconds,
            )
        except httpx.TimeoutException as error:
            error_category, error_message = "timeout", str(error)
        except httpx.TransportError as error:
            error_category, error_message = "transport_error", str(error)

        raw_response = _response_body(response) if response is not None else None
        status_code = response.status_code if response is not None else None
        retryable_http_error = False
        if response is not None and response.status_code >= 400:
            error_category = _http_error_category(response.status_code)
            error_message = _error_message(raw_response, response.status_code)
            retryable_http_error = response.status_code in {408, 429, *range(500, 600)}

        attempt = HttpAttempt(
            attempt_number=attempt_number,
            started_at=started_at,
            finished_at=_utc_now(),
            duration_ms=round((time.perf_counter() - started_clock) * 1000),
            status_code=status_code,
            error_category=error_category,
            error_message=error_message,
            raw_request=sanitize_for_audit(payload) if self.debug else None,
            raw_response=sanitize_for_audit(raw_response) if self.debug else None,
        )
        self.telemetry.record_audit(
            span,
            {
                "request": sanitize_for_audit(payload),
                "response": sanitize_for_audit(raw_response),
                "response_headers": _audit_headers(response),
                "error": {"category": error_category, "message": error_message} if error_category else None,
            },
            {**audit_attributes, "http.attempt": attempt_number, "http.status_code": status_code or 0},
        )

        if error_category is not None:
            self.telemetry.end_span(span, RuntimeError(error_message or error_category))
            if retryable_http_error or response is None:
                return attempt, None
            raise ApiError(error_message or error_category, [attempt])
        assert response is not None
        self.telemetry.end_span(span)
        return attempt, response

    async def _model_reply(self, response: httpx.Response, attempts: list[HttpAttempt]) -> ModelReply:
        try:
            body = response.json()
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ApiError(f"malformed OpenRouter response: {error}", attempts) from error

        if not isinstance(body, dict):
            raise ApiError("malformed OpenRouter response types", attempts)

        usage = normalize_usage(body.get("usage"))
        generation_id = _string(body.get("id")) or response.headers.get("X-Generation-Id")
        metadata = await self._generation_metadata(generation_id, usage, body)
        usage = merge_usage(usage, normalize_usage(metadata.get("usage")))
        if usage.cost_usd is None:
            usage.cost_usd = _float(metadata.get("total_cost") or metadata.get("cost"))

        observed_provider = _first_string(body, metadata, keys=("provider", "provider_name"))
        observed_endpoint = _first_string(body, metadata, keys=("provider_endpoint", "endpoint", "endpoint_name"))
        observed_quantization = _first_string(body, metadata, keys=("quantization",))
        cache_status = response.headers.get("X-OpenRouter-Cache-Status", "UNKNOWN").upper()
        final_attempt = attempts[-1]
        final_attempt.usage = usage
        final_attempt.generation_id = generation_id
        final_attempt.cache_status = cache_status
        final_attempt.observed_provider = observed_provider
        final_attempt.observed_endpoint = observed_endpoint
        final_attempt.observed_quantization = observed_quantization

        try:
            message = body["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ApiError(f"malformed OpenRouter response: {error}", attempts) from error
        if not isinstance(message, dict) or not isinstance(content, str):
            if _finish_reason(body) == "length":
                error_message = "model output truncated at the configured token limit"
                final_attempt.error_category = "output_truncated"
                final_attempt.error_message = error_message
                raise ApiError(error_message, attempts)
            raise ApiError("malformed OpenRouter response types", attempts)

        return ModelReply(
            content=content,
            assistant_message=dict(message),
            attempts=attempts,
            usage=usage,
            generation_id=generation_id,
            cache_status=cache_status,
            observed_provider=observed_provider,
            observed_endpoint=observed_endpoint,
            observed_quantization=observed_quantization,
        )

    async def _generation_metadata(
        self, generation_id: str | None, usage: Usage, body: dict[str, Any]
    ) -> dict[str, Any]:
        if generation_id is None or (usage.cost_usd is not None and body.get("provider") is not None):
            return {}
        deadline = time.monotonic() + self.execution.metadata_timeout_seconds
        delay = 0.1
        while time.monotonic() < deadline:
            try:
                response = await self.http.get(
                    f"{self.base_url}/generation",
                    headers={"Authorization": self.headers["Authorization"]},
                    params={"id": generation_id},
                    timeout=self.execution.metadata_timeout_seconds,
                )
                if response.is_success:
                    data = _data_object(response.json())
                    if data:
                        return data
            except httpx.HTTPError, TypeError, ValueError:
                pass
            await self.sleep(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 2.0)
        return {}

    def _retry_delay(self, attempt_number: int) -> float:
        retry = self.execution.retry
        base = min(retry.initial_delay_seconds * (2 ** (attempt_number - 1)), retry.max_delay_seconds)
        jitter = base * retry.jitter_ratio
        return float(max(0.0, base + self.random.uniform(-jitter, jitter)))

    def _configured_parameters(self) -> set[str]:
        raw = self.model.inference.model_dump(mode="json")
        return {key for key, value in raw.items() if value is not None}


def normalize_usage(value: Any) -> Usage:
    if not isinstance(value, dict):
        return Usage()
    prompt_details = value.get("prompt_tokens_details") or {}
    completion_details = value.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=_int(value.get("prompt_tokens") or value.get("input_tokens")),
        output_tokens=_int(value.get("completion_tokens") or value.get("output_tokens")),
        reasoning_tokens=_int(completion_details.get("reasoning_tokens") or value.get("reasoning_tokens")),
        cached_tokens=_int(prompt_details.get("cached_tokens") or value.get("cached_input_tokens")),
        cache_write_tokens=_int(prompt_details.get("cache_write_tokens") or value.get("cache_write_tokens")),
        total_tokens=_int(value.get("total_tokens")),
        cost_usd=_float(value.get("cost")),
    )


def merge_usage(primary: Usage, fallback: Usage) -> Usage:
    return Usage(
        input_tokens=primary.input_tokens if primary.input_tokens is not None else fallback.input_tokens,
        output_tokens=primary.output_tokens if primary.output_tokens is not None else fallback.output_tokens,
        reasoning_tokens=primary.reasoning_tokens
        if primary.reasoning_tokens is not None
        else fallback.reasoning_tokens,
        cached_tokens=primary.cached_tokens if primary.cached_tokens is not None else fallback.cached_tokens,
        cache_write_tokens=(
            primary.cache_write_tokens if primary.cache_write_tokens is not None else fallback.cache_write_tokens
        ),
        total_tokens=primary.total_tokens if primary.total_tokens is not None else fallback.total_tokens,
        cost_usd=primary.cost_usd if primary.cost_usd is not None else fallback.cost_usd,
    )


def _data_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("OpenRouter response must be a JSON object")
    data = value.get("data", value)
    if not isinstance(data, dict):
        raise ValueError("OpenRouter response data must be a JSON object")
    return data


def _endpoint_slugs(endpoint: dict[str, Any]) -> set[str]:
    return {
        value
        for key in ("name", "provider_name", "provider_slug", "slug", "tag")
        if isinstance((value := endpoint.get(key)), str)
    }


def _endpoint_quantizations(endpoint: dict[str, Any]) -> set[str]:
    value = endpoint.get("quantizations", endpoint.get("quantization"))
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _response_body(response: httpx.Response | None) -> dict[str, Any] | None:
    if response is None:
        return None
    try:
        value = response.json()
    except ValueError:
        return {"text": response.text}
    return value if isinstance(value, dict) else {"value": value}


def _audit_headers(response: httpx.Response | None) -> dict[str, str] | None:
    if response is None:
        return None
    allowed = {"x-openrouter-cache-status", "x-openrouter-cache-age", "x-generation-id", "content-type"}
    return {key: value for key, value in response.headers.items() if key.lower() in allowed}


def _error_message(body: dict[str, Any] | None, status_code: int) -> str:
    if body:
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return f"HTTP {status_code}: {error['message']}"
        if isinstance(error, str):
            return f"HTTP {status_code}: {error}"
    return f"HTTP {status_code}"


def _http_error_category(status_code: int) -> str:
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if status_code >= 500:
        return "server_error"
    if status_code in {401, 403}:
        return "authentication"
    if status_code == 402:
        return "insufficient_credit"
    return "client_error"


def _first_string(*values: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for value in values:
        for key in keys:
            if isinstance((candidate := value.get(key)), str):
                return candidate
    return None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _finish_reason(body: dict[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return _string(choices[0].get("finish_reason"))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
