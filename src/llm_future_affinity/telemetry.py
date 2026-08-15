"""OpenTelemetry traces and correlated audit logs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import httpx
from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode

from llm_future_affinity.config import ObservabilityConfig

type AttributeValue = str | bool | int | float


class Telemetry(Protocol):
    failed: bool

    async def preflight(self) -> None: ...

    def start_span(
        self, name: str, parent: Span | None = None, attributes: Mapping[str, Any] | None = None
    ) -> Span: ...

    def end_span(self, span: Span, error: BaseException | None = None) -> None: ...

    def span_ids(self, span: Span) -> tuple[str | None, str | None]: ...

    def record_audit(self, span: Span, body: dict[str, Any], attributes: Mapping[str, Any]) -> None: ...

    async def force_flush(self) -> bool: ...

    async def shutdown(self) -> None: ...


class NullTelemetry:
    failed = False

    async def preflight(self) -> None:
        return None

    def start_span(self, name: str, parent: Span | None = None, attributes: Mapping[str, Any] | None = None) -> Span:
        del name, parent, attributes
        return trace.INVALID_SPAN

    def end_span(self, span: Span, error: BaseException | None = None) -> None:
        del span, error

    def span_ids(self, span: Span) -> tuple[str | None, str | None]:
        del span
        return None, None

    def record_audit(self, span: Span, body: dict[str, Any], attributes: Mapping[str, Any]) -> None:
        del span, body, attributes

    async def force_flush(self) -> bool:
        return True

    async def shutdown(self) -> None:
        return None


class OtelTelemetry:
    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self.failed = False
        resource = Resource.create({"service.name": config.service_name})
        timeout_ms = int(config.flush_timeout_seconds * 1000)

        self._tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        self._span_exporter = _TrackingSpanExporter(
            OTLPSpanExporter(
                endpoint=f"{config.otlp_endpoint.rstrip('/')}/v1/traces",
                timeout=config.flush_timeout_seconds,
            ),
            self._mark_failed,
        )
        self._tracer_provider.add_span_processor(
            BatchSpanProcessor(self._span_exporter, export_timeout_millis=timeout_ms),
        )
        self._tracer = self._tracer_provider.get_tracer(config.service_name)

        self._logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        self._log_exporter = _TrackingLogExporter(
            cast(
                _LogExporterDelegate,
                OTLPLogExporter(
                    endpoint=f"{config.otlp_endpoint.rstrip('/')}/v1/logs",
                    timeout=config.flush_timeout_seconds,
                ),
            ),
            self._mark_failed,
        )
        self._logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(self._log_exporter, export_timeout_millis=timeout_ms),
        )
        self._logger = self._logger_provider.get_logger(f"{config.service_name}.audit")

    async def preflight(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.config.flush_timeout_seconds) as client:
                response = await client.get(self.config.health_endpoint)
                response.raise_for_status()
        except httpx.HTTPError as error:
            self.failed = True
            raise RuntimeError(f"required OTEL backend is unavailable: {error}") from error

    def start_span(self, name: str, parent: Span | None = None, attributes: Mapping[str, Any] | None = None) -> Span:
        context = trace.set_span_in_context(parent) if parent is not None else None
        return self._tracer.start_span(name, context=context, attributes=_clean_attributes(attributes or {}))

    def end_span(self, span: Span, error: BaseException | None = None) -> None:
        if error is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    def span_ids(self, span: Span) -> tuple[str | None, str | None]:
        context = span.get_span_context()
        if not context.is_valid:
            return None, None
        return f"{context.trace_id:032x}", f"{context.span_id:016x}"

    def record_audit(self, span: Span, body: dict[str, Any], attributes: Mapping[str, Any]) -> None:
        self._logger.emit(
            context=trace.set_span_in_context(span),
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            attributes=_clean_attributes(attributes),
            event_name="openrouter.audit",
        )

    async def force_flush(self) -> bool:
        timeout_ms = int(self.config.flush_timeout_seconds * 1000)
        spans_ok, logs_ok = await asyncio.gather(
            asyncio.to_thread(self._tracer_provider.force_flush, timeout_ms),
            asyncio.to_thread(self._logger_provider.force_flush, timeout_ms),
        )
        self.failed = self.failed or not (spans_ok and logs_ok)
        return not self.failed

    async def shutdown(self) -> None:
        await self.force_flush()
        await asyncio.gather(
            asyncio.to_thread(self._tracer_provider.shutdown),
            asyncio.to_thread(self._logger_provider.shutdown),
        )

    def _mark_failed(self) -> None:
        self.failed = True


class _TrackingSpanExporter(SpanExporter):
    def __init__(self, delegate: SpanExporter, mark_failed: Callable[[], None]) -> None:
        self._delegate = delegate
        self._mark_failed = mark_failed

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            result = self._delegate.export(spans)
        except BaseException:
            self._mark_failed()
            raise
        if result is not SpanExportResult.SUCCESS:
            self._mark_failed()
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class _LogExporterDelegate(Protocol):
    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = 10_000) -> bool: ...


class _TrackingLogExporter(LogRecordExporter):
    def __init__(self, delegate: _LogExporterDelegate, mark_failed: Callable[[], None]) -> None:
        self._delegate = delegate
        self._mark_failed = mark_failed

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        try:
            result = self._delegate.export(batch)
        except BaseException:
            self._mark_failed()
            raise
        if result is not LogRecordExportResult.SUCCESS:
            self._mark_failed()
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 10_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def sanitize_for_audit(value: Any) -> Any:
    """Remove secrets and plaintext hidden reasoning before data leaves the process."""
    if isinstance(value, list):
        return [sanitize_for_audit(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        lowered = key.lower()
        if lowered in {"authorization", "api_key", "x-api-key"}:
            continue
        if lowered in {"reasoning", "reasoning_content"}:
            continue
        if lowered == "reasoning_details" and isinstance(item, list):
            sanitized[key] = [_sanitize_reasoning_detail(detail) for detail in item]
            continue
        sanitized[key] = sanitize_for_audit(item)
    return sanitized


def _sanitize_reasoning_detail(value: Any) -> Any:
    if not isinstance(value, dict):
        return sanitize_for_audit(value)
    sanitized = dict(value)
    sanitized.pop("text", None)
    sanitized.pop("summary", None)
    return sanitize_for_audit(sanitized)


def _clean_attributes(attributes: Mapping[str, Any]) -> dict[str, AttributeValue]:
    cleaned: dict[str, AttributeValue] = {}
    for key, value in attributes.items():
        if isinstance(value, str | bool | int | float):
            cleaned[key] = value
        elif value is not None:
            cleaned[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return cleaned
