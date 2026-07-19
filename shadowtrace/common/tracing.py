"""Local-first OpenTelemetry tracing (plan.md §4.5).

Default exporter writes spans as JSON lines to a local file — nothing
leaves the machine. OTLP export to the user's own collector is opt-in
only (a real collector endpoint must be passed explicitly) and needs the
`otel` extra (`pip install '.[otel]'`) for the OTLP exporter package;
its absence degrades gracefully (local export still works, a warning is
logged) rather than failing.

Deliberately does *not* use the global `opentelemetry.trace` provider
registration — `TracerProvider.get_tracer()` is called directly on an
instance instead. `set_tracer_provider()` is a process-wide singleton
that can only be set once, which makes it awkward to test (each test
would fight over the same global state); building an explicit provider
per caller (one in the CLI process, one per test) sidesteps that
entirely while still producing real, spec-compliant spans.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from shadowtrace.common.logging import get_logger

logger = get_logger("common.tracing")


class LocalFileSpanExporter(SpanExporter):
    """Appends each finished span as a JSON line. No network involved."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("could not create traces directory for %s", self.path)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            lines = [json.dumps(_span_to_dict(span)) for span in spans]
            with self._lock, self.path.open("a") as f:
                for line in lines:
                    f.write(line + "\n")
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("failed to export spans to %s", self.path)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else None,
        "span_id": format(ctx.span_id, "016x") if ctx else None,
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "attributes": dict(span.attributes) if span.attributes else {},
        "status": span.status.status_code.name if span.status else None,
    }


def build_tracer_provider(traces_path: Path, otlp_endpoint: str | None = None) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "shadowtrace"}))
    provider.add_span_processor(SimpleSpanProcessor(LocalFileSpanExporter(traces_path)))

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        except ImportError:
            logger.warning(
                "SHADOWTRACE_OTLP_ENDPOINT is set but the OTLP exporter isn't installed "
                "(pip install '.[otel]'); local file export only"
            )

    return provider


def get_default_tracer() -> trace.Tracer:
    """A provider-less tracer: spans it creates are real objects with the
    OTel API shape but are never exported anywhere. Safe fallback for any
    component that isn't handed an explicit tracer (e.g. in tests that
    don't care about tracing output)."""
    return trace.get_tracer("shadowtrace")
