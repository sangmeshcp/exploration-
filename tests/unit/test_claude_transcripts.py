import json
from pathlib import Path

import pytest

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.ingest.claude_transcripts import TranscriptFile, TranscriptWatcher
from shadowtrace.proxy.capture import CaptureWriter


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    REGISTRY.reset()


def _user_line(uuid: str, text: str, ts: str = "2026-07-18T00:00:00Z") -> str:
    return json.dumps(
        {
            "type": "user",
            "uuid": uuid,
            "timestamp": ts,
            "message": {"role": "user", "content": text},
        }
    )


def _assistant_line(
    uuid: str, text: str, model: str = "claude-sonnet-5", ts: str = "2026-07-18T00:00:01Z"
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "uuid": uuid,
            "timestamp": ts,
            "message": {
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        }
    )


def test_parses_user_assistant_pair(tmp_path: Path) -> None:
    path = tmp_path / "session1.jsonl"
    path.write_text(_user_line("u1", "hello") + "\n" + _assistant_line("a1", "hi there") + "\n")

    tf = TranscriptFile(path)
    records = tf.poll()

    assert len(records) == 1
    r = records[0]
    assert r.source == "transcript"
    assert r.tool == "claude_code"
    assert r.model == "claude-sonnet-5"
    assert r.request_json["content"] == "hello"
    assert r.response_json["content"][0]["text"] == "hi there"
    assert r.tokens_in == 10
    assert r.tokens_out == 20
    assert r.cost_usd is None
    assert r.ingest_key == "a1"
    assert r.session_id == "session1"


def test_incremental_tail_parse_only_reads_new_lines(tmp_path: Path) -> None:
    path = tmp_path / "session2.jsonl"
    path.write_text(_user_line("u1", "hello") + "\n" + _assistant_line("a1", "hi") + "\n")
    tf = TranscriptFile(path)
    first = tf.poll()
    assert len(first) == 1

    with path.open("a") as f:
        f.write(_user_line("u2", "again") + "\n" + _assistant_line("a2", "sure") + "\n")
    second = tf.poll()
    assert len(second) == 1
    assert second[0].ingest_key == "a2"
    assert second[0].parent_id == first[0].id  # chained via parent_id


def test_live_appending_partial_line_not_parsed_until_complete(tmp_path: Path) -> None:
    path = tmp_path / "session3.jsonl"
    path.write_text(_user_line("u1", "hello") + "\n")
    tf = TranscriptFile(path)
    assert tf.poll() == []

    # simulate a partial write of the assistant line (no trailing newline yet)
    partial = _assistant_line("a1", "hi")[:20]
    with path.open("a") as f:
        f.write(partial)
    assert tf.poll() == []  # incomplete line must not be parsed

    with path.open("a") as f:
        f.write(_assistant_line("a1", "hi")[20:] + "\n")
    records = tf.poll()
    assert len(records) == 1
    assert records[0].response_json["content"][0]["text"] == "hi"


def test_malformed_line_skipped_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "session4.jsonl"
    path.write_text(
        _user_line("u1", "hello")
        + "\n"
        + "{not valid json truncated"
        + "\n"
        + _assistant_line("a1", "hi")
        + "\n"
    )
    tf = TranscriptFile(path)
    records = tf.poll()
    assert len(records) == 1  # malformed line skipped, good line still processed
    assert REGISTRY.get_counter("parse_errors_total") == 1


def test_unknown_schema_line_flagged_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "session5.jsonl"
    path.write_text(json.dumps({"type": "summary", "text": "session recap"}) + "\n")
    tf = TranscriptFile(path)
    records = tf.poll()
    assert records == []
    assert REGISTRY.get_counter("unknown_schema_lines_total") == 1


@pytest.mark.asyncio
async def test_watcher_scan_once_ingests_into_capture(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "proj1").mkdir()
    session_path = projects_dir / "proj1" / "session1.jsonl"
    session_path.write_text(_user_line("u1", "hi") + "\n" + _assistant_line("a1", "hello") + "\n")

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    watcher = TranscriptWatcher(projects_dir, capture)
    count = watcher.scan_once()
    assert count == 1
    await capture.flush()

    import sqlite3

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    rows = conn.execute("SELECT source, ingest_key FROM traces").fetchall()
    assert rows == [("transcript", "a1")]


def test_watcher_idempotent_rescan_is_deduped_at_db_layer(tmp_path: Path) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    session_path = projects_dir / "session1.jsonl"
    session_path.write_text(_user_line("u1", "hi") + "\n" + _assistant_line("a1", "hello") + "\n")

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    watcher1 = TranscriptWatcher(projects_dir, capture)
    watcher1.scan_once()

    # A brand-new watcher (simulating a restart) re-reads the whole file from
    # offset 0 — the DB-level unique (source, ingest_key) index must dedup it.
    watcher2 = TranscriptWatcher(projects_dir, capture)
    watcher2.scan_once()

    import asyncio
    import sqlite3

    asyncio.run(capture.flush())
    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 1


def test_watcher_missing_dir_returns_zero(tmp_path: Path) -> None:
    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill")
    watcher = TranscriptWatcher(tmp_path / "does-not-exist", capture)
    assert watcher.scan_once() == 0
