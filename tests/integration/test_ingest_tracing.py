"""Transcript ingest is traced per file (plan.md §4.5)."""

import json
from pathlib import Path

from shadowtrace.common.tracing import build_tracer_provider
from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.proxy.capture import CaptureWriter


def _read_spans(traces_path: Path) -> list[dict[str, object]]:
    if not traces_path.exists():
        return []
    return [json.loads(line) for line in traces_path.read_text().strip().split("\n") if line]


def test_scan_once_emits_one_span_per_transcript_file(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "session1.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "a1",
                "timestamp": "2026-07-18T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "model": "x",
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
        + "\n"
    )
    (projects_dir / "session2.jsonl").write_text("")

    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill")
    watcher = TranscriptWatcher(projects_dir, capture, tracer=tracer)
    watcher.scan_once()

    spans = _read_spans(traces_path)
    ingest_spans = [s for s in spans if s["name"] == "ingest_transcript_file"]
    assert len(ingest_spans) == 2
    paths = {s["attributes"]["path"] for s in ingest_spans}
    assert str(projects_dir / "session1.jsonl") in paths
    records_by_path = {
        s["attributes"]["path"]: s["attributes"]["records_ingested"] for s in ingest_spans
    }
    assert records_by_path[str(projects_dir / "session1.jsonl")] == 1
    assert records_by_path[str(projects_dir / "session2.jsonl")] == 0
