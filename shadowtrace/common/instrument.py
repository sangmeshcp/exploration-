"""Observability-by-default: one wrapper that gives any operation a span, a
duration histogram, an outcome counter and a structured error log, all with
the same names and labels everywhere.

    @instrument("etl.run", component="etl")
    def run_etl(...): ...

    with instrumented("replay.call", component="replay", attributes=...) as span:
        ...

Every instrumented operation emits:

  - span `<operation>` (attributes: `component`, plus whatever is passed)
  - histogram `operation_duration_ms{component, operation}`
  - counter   `operations_total{component, operation, status=ok|error}`
  - on error: one ERROR log line carrying the span's trace id, then re-raise

Because names and labels are fixed, one dashboard panel or alert rule
(`monitor/slo.py`) covers every pipeline stage — a new stage gets latency,
throughput and error-rate tracking by adding a decorator, not by inventing
metric names.

`genai_attributes()` builds the OpenTelemetry GenAI semantic-convention
attributes (`gen_ai.*`) for spans that wrap a model call, so the same spans
render as LLM calls in any backend that understands the convention (Langfuse,
Arize Phoenix, Datadog LLM Observability, Honeycomb, Grafana) once
`SHADOWTRACE_OTLP_ENDPOINT` points at it.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.tracing import get_default_tracer

logger = get_logger("common.instrument")

P = ParamSpec("P")
R = TypeVar("R")

AttributeValue = str | bool | int | float


@contextmanager
def instrumented(
    operation: str,
    *,
    component: str,
    attributes: dict[str, AttributeValue] | None = None,
    tracer: trace.Tracer | None = None,
) -> Iterator[trace.Span]:
    tracer = tracer or get_default_tracer()
    labels = {"component": component, "operation": operation}
    span_attributes: dict[str, AttributeValue] = {"component": component, **(attributes or {})}
    start = time.perf_counter()
    with tracer.start_as_current_span(
        operation, attributes=span_attributes, record_exception=False
    ) as span:
        try:
            yield span
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            REGISTRY.observe("operation_duration_ms", duration_ms, labels=labels)
            REGISTRY.inc("operations_total", labels={**labels, "status": "error"})
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            ctx = span.get_span_context()
            logger.error(
                "%s failed: %s: %s (trace_id=%s)",
                operation,
                type(exc).__name__,
                exc,
                format(ctx.trace_id, "032x") if ctx and ctx.is_valid else "-",
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        REGISTRY.observe("operation_duration_ms", duration_ms, labels=labels)
        REGISTRY.inc("operations_total", labels={**labels, "status": "ok"})
        span.set_attribute("duration_ms", duration_ms)


def instrument(operation: str, *, component: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of `instrumented()`; works on sync and async functions."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(fn):
            async_fn = cast(Callable[P, Awaitable[Any]], fn)

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with instrumented(operation, component=component):
                    return await async_fn(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with instrumented(operation, component=component):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


_PROVIDER_TO_GENAI_SYSTEM = {
    "anthropic": "anthropic",
    "openai": "openai",
    "openrouter": "openrouter",
    "ollama": "ollama",
    "vllm": "vllm",
}


def genai_attributes(
    *,
    provider: str,
    request_model: str,
    operation: str = "chat",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    response_model: str | None = None,
) -> dict[str, AttributeValue]:
    """OpenTelemetry GenAI semantic-convention attributes for a model call."""
    attrs: dict[str, AttributeValue] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": _PROVIDER_TO_GENAI_SYSTEM.get(provider, provider),
        "gen_ai.request.model": request_model,
    }
    if response_model is not None:
        attrs["gen_ai.response.model"] = response_model
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    return attrs
