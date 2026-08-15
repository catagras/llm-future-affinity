from __future__ import annotations

import httpx
import pytest
import respx
from opentelemetry.sdk.trace.export import SpanExportResult

from llm_future_affinity.config import ObservabilityConfig
from llm_future_affinity.telemetry import OtelTelemetry


def otel_config() -> ObservabilityConfig:
    return ObservabilityConfig(
        otlp_endpoint="http://otel.test:4318",
        health_endpoint="http://otel.test:13133",
        service_name="test",
        flush_timeout_seconds=0.1,
    )


@respx.mock
async def test_otel_preflight_success_and_span_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get("http://otel.test:13133").mock(return_value=httpx.Response(200))
    telemetry = OtelTelemetry(otel_config())
    await telemetry.preflight()
    span = telemetry.start_span("test", attributes={"value": [1, 2], "none": None})
    telemetry.record_audit(span, {"prompt": "hello"}, {"game": 1})
    trace_id, span_id = telemetry.span_ids(span)
    assert trace_id is not None and len(trace_id) == 32
    assert span_id is not None and len(span_id) == 16
    telemetry.end_span(span)
    monkeypatch.setattr(telemetry._tracer_provider, "force_flush", lambda timeout: True)
    monkeypatch.setattr(telemetry._logger_provider, "force_flush", lambda timeout: True)
    monkeypatch.setattr(telemetry._tracer_provider, "shutdown", lambda: None)
    monkeypatch.setattr(telemetry._logger_provider, "shutdown", lambda: None)
    assert await telemetry.force_flush()
    await telemetry.shutdown()


@respx.mock
async def test_otel_preflight_failure() -> None:
    respx.get("http://otel.test:13133").mock(return_value=httpx.Response(503))
    telemetry = OtelTelemetry(otel_config())
    with pytest.raises(RuntimeError, match="unavailable"):
        await telemetry.preflight()
    assert telemetry.failed
    await telemetry.shutdown()


async def test_otel_flush_failure_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = OtelTelemetry(otel_config())
    monkeypatch.setattr(telemetry._tracer_provider, "force_flush", lambda timeout: False)
    monkeypatch.setattr(telemetry._logger_provider, "force_flush", lambda timeout: True)
    monkeypatch.setattr(telemetry._tracer_provider, "shutdown", lambda: None)
    monkeypatch.setattr(telemetry._logger_provider, "shutdown", lambda: None)
    assert not await telemetry.force_flush()
    assert telemetry.failed
    await telemetry.shutdown()


async def test_otel_exporter_failure_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = OtelTelemetry(otel_config())
    monkeypatch.setattr(telemetry._span_exporter._delegate, "export", lambda spans: SpanExportResult.FAILURE)
    assert telemetry._span_exporter.export([]) is SpanExportResult.FAILURE
    assert telemetry.failed
    monkeypatch.setattr(telemetry._tracer_provider, "force_flush", lambda timeout: True)
    monkeypatch.setattr(telemetry._logger_provider, "force_flush", lambda timeout: True)
    monkeypatch.setattr(telemetry._tracer_provider, "shutdown", lambda: None)
    monkeypatch.setattr(telemetry._logger_provider, "shutdown", lambda: None)
    await telemetry.shutdown()
