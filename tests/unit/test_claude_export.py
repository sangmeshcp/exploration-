import asyncio
import json
import sqlite3
from pathlib import Path

from shadowtrace.ingest.claude_export import import_export_file
from shadowtrace.proxy.capture import CaptureWriter


def _write_export(path: Path) -> None:
    data = {
        "conversations": [
            {
                "uuid": "convo-1",
                "chat_messages": [
                    {"uuid": "m1", "sender": "human", "text": "hi there"},
                    {
                        "uuid": "m2",
                        "sender": "assistant",
                        "text": "hello!",
                        "model": "claude-sonnet-5",
                        "created_at": "2026-07-18T00:00:00Z",
                    },
                ],
            }
        ]
    }
    path.write_text(json.dumps(data))


def test_import_export_file_enqueues_assistant_turns(tmp_path: Path) -> None:
    export_path = tmp_path / "conversations.json"
    _write_export(export_path)

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    count = import_export_file(export_path, capture)
    assert count == 1
    asyncio.run(capture.flush())

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    row = conn.execute("SELECT source, session_id, ingest_key, model FROM traces").fetchone()
    assert row == ("claude_export", "convo-1", "m2", "claude-sonnet-5")


def test_reimporting_same_file_is_idempotent(tmp_path: Path) -> None:
    export_path = tmp_path / "conversations.json"
    _write_export(export_path)

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    import_export_file(export_path, capture)
    import_export_file(export_path, capture)  # re-import, same file
    asyncio.run(capture.flush())

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 1


def test_extended_export_only_ingests_new_rows(tmp_path: Path) -> None:
    export_path = tmp_path / "conversations.json"
    _write_export(export_path)
    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    import_export_file(export_path, capture)
    asyncio.run(capture.flush())

    data = json.loads(export_path.read_text())
    data["conversations"][0]["chat_messages"].extend(
        [
            {"uuid": "m3", "sender": "human", "text": "another question"},
            {"uuid": "m4", "sender": "assistant", "text": "another answer", "model": "x"},
        ]
    )
    export_path.write_text(json.dumps(data))
    import_export_file(export_path, capture)
    asyncio.run(capture.flush())

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 2


def test_malformed_export_file_returns_zero_not_crash(tmp_path: Path) -> None:
    export_path = tmp_path / "bad.json"
    export_path.write_text("not json")
    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill")
    assert import_export_file(export_path, capture) == 0
