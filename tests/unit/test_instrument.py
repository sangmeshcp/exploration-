import json
from pathlib import Path

import pytest

from shadowtrace.common.instrument import genai_attributes, instrument, instrumented
from shadowtrace.common.metrics import REGISTRY, MetricsRegistry
from shadowtrace.common.tracing import build_tracer_provider, register_process_tracer_provider


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    REGISTRY.reset()


def _spans(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_instrumented_success_records_duration_and_ok_count(tmp_path: Path) -> None:
    provider = build_tracer_provider(tmp_path / "traces.jsonl")
    with instrumented(
        "etl.run", component="etl", attributes={"rows": 3}, tracer=provider.get_tracer("t")
    ):
        pass

    labels = {"component": "etl", "operation": "etl.run"}
    assert REGISTRY.get_counter("operations_total", {**labels, "status": "ok"}) == 1
    assert REGISTRY.find_percentile("operation_duration_ms", 0.5) is not None
    (span,) = _spans(tmp_path / "traces.jsonl")
    assert span["name"] == "etl.run"
    assert span["attributes"]["component"] == "etl"  # type: ignore[index]
    assert span["attributes"]["rows"] == 3  # type: ignore[index]
    assert span["status"] == "UNSET"


def test_instrumented_error_counts_marks_span_and_reraises(tmp_path: Path) -> None:
    provider = build_tracer_provider(tmp_path / "traces.jsonl")
    with (
        pytest.raises(ValueError),
        instrumented("etl.run", component="etl", tracer=provider.get_tracer("t")),
    ):
        raise ValueError("boom")

    labels = {"component": "etl", "operation": "etl.run", "status": "error"}
    assert REGISTRY.get_counter("operations_total", labels) == 1
    (span,) = _spans(tmp_path / "traces.jsonl")
    assert span["status"] == "ERROR"


def test_decorator_uses_registered_process_provider_for_sync_and_async(tmp_path: Path) -> None:
    register_process_tracer_provider(build_tracer_provider(tmp_path / "traces.jsonl"))

    @instrument("sync.op", component="test")
    def add(a: int, b: int) -> int:
        return a + b

    @instrument("async.op", component="test")
    async def double(x: int) -> int:
        return x * 2

    import asyncio

    assert add(1, 2) == 3
    assert asyncio.run(double(4)) == 8
    names = [s["name"] for s in _spans(tmp_path / "traces.jsonl")]
    assert names == ["sync.op", "async.op"]
    assert double.__name__ == "double"


def test_genai_attributes_follow_semantic_conventions() -> None:
    attrs = genai_attributes(
        provider="anthropic", request_model="claude-haiku-5-5", input_tokens=10, output_tokens=2
    )
    assert attrs == {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-haiku-5-5",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 2,
    }
    assert "gen_ai.usage.input_tokens" not in genai_attributes(provider="x", request_model="m")


def test_prometheus_exposition() -> None:
    reg = MetricsRegistry()
    reg.inc("requests_total", labels={"status": "200"})
    reg.inc("requests_total", labels={"status": "200"})
    reg.set_gauge("capture_lag_seconds", 4.5)
    reg.inc("weird.name", labels={"path": 'a"b\\c'})
    for v in (1.0, 2.0, 3.0):
        reg.observe("latency_ms", v)

    text = reg.to_prometheus()
    assert "# TYPE shadowtrace_requests_total counter" in text
    assert 'shadowtrace_requests_total{status="200"} 2.0' in text
    assert "shadowtrace_capture_lag_seconds 4.5" in text
    assert 'shadowtrace_weird_name{path="a\\"b\\\\c"} 1.0' in text
    assert "# TYPE shadowtrace_latency_ms summary" in text
    assert 'shadowtrace_latency_ms{quantile="0.5"} 2.0' in text
    assert "shadowtrace_latency_ms_sum 6.0" in text
    assert "shadowtrace_latency_ms_count 3" in text
    assert text.endswith("\n")


def test_find_distinguishes_missing_from_zero() -> None:
    reg = MetricsRegistry()
    assert reg.find("nope") is None
    reg.inc("errors_total", value=0)
    assert reg.find("errors_total") == 0
    reg.inc("errors_total", labels={"k": "a"})
    reg.inc("errors_total", labels={"k": "b"}, value=2)
    assert reg.find("errors_total") == 3
    assert reg.find("errors_total", labels={"k": "b"}) == 2
    assert reg.find_percentile("nope", 0.99) is None
