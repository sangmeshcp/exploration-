import json
from pathlib import Path

from shadowtrace.common.tracing import build_tracer_provider, get_default_tracer


def test_default_tracer_creates_noop_spans_without_crashing() -> None:
    tracer = get_default_tracer()
    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("x", 1)
    # no exporter registered — nothing to assert beyond "didn't raise"


def test_local_file_exporter_writes_span_as_json_line(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("my_span", attributes={"foo": "bar"}) as span:
        span.set_attribute("n", 42)

    lines = traces_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "my_span"
    assert record["attributes"]["foo"] == "bar"
    assert record["attributes"]["n"] == 42
    assert record["trace_id"] is not None
    assert record["span_id"] is not None


def test_local_file_exporter_captures_parent_child_relationship(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("root") as root:
        root_span_id = format(root.get_span_context().span_id, "016x")
        with tracer.start_as_current_span("child"):
            pass

    records = [json.loads(line) for line in traces_path.read_text().strip().split("\n")]
    assert len(records) == 2
    child = next(r for r in records if r["name"] == "child")
    root_record = next(r for r in records if r["name"] == "root")
    assert child["parent_span_id"] == root_span_id
    assert child["trace_id"] == root_record["trace_id"]  # same trace


def test_local_file_exporter_survives_unwritable_path(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_dir" / "traces.jsonl"
    (tmp_path / "not_a_dir").write_text("i am a file")
    provider = build_tracer_provider(bad_path)
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("span"):
        pass  # must not raise even though the exporter can't write anywhere


def test_missing_otlp_exporter_degrades_to_local_only(tmp_path: Path) -> None:
    # no `.[otel]` extra installed in this test env -> falls back with a
    # warning instead of raising
    provider = build_tracer_provider(
        tmp_path / "traces.jsonl", otlp_endpoint="http://localhost:4318"
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("span"):
        pass
    assert (tmp_path / "traces.jsonl").exists()
