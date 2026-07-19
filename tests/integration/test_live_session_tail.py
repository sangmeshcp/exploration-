"""E2: live-session tail, at the TranscriptWatcher pipeline level (not
just a single TranscriptFile in isolation). Multiple sessions grow
incrementally and concurrently; rows must appear as soon as a line is
complete, never waiting for the file to close, and never as partial or
duplicated rows (plan.md §3.3).
"""

import json
import sqlite3
from pathlib import Path

import pytest

from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.proxy.capture import CaptureWriter


def _user(uuid: str, text: str, ts: str) -> str:
    return json.dumps(
        {
            "type": "user",
            "uuid": uuid,
            "timestamp": ts,
            "message": {"role": "user", "content": text},
        }
    )


def _assistant(uuid: str, text: str, ts: str) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "uuid": uuid,
            "timestamp": ts,
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            },
        }
    )


@pytest.mark.asyncio
async def test_watcher_ingests_incrementally_across_two_concurrent_growing_sessions(
    tmp_path: Path,
) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    session_a = projects_dir / "session_a.jsonl"
    session_b = projects_dir / "session_b.jsonl"
    session_a.touch()
    session_b.touch()

    capture = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    watcher = TranscriptWatcher(projects_dir, capture)

    # "tick" 1: only session_a has produced a full turn so far
    with session_a.open("a") as f:
        f.write(_user("a-u1", "hi from a", "2026-07-18T00:00:00Z") + "\n")
        f.write(_assistant("a-a1", "hello a", "2026-07-18T00:00:01Z") + "\n")
    assert watcher.scan_once() == 1
    await capture.flush()

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    assert conn.execute("SELECT count(*) FROM traces").fetchone()[0] == 1

    # "tick" 2: session_b starts, session_a appends a second turn but its
    # assistant line isn't complete yet (still being written, no trailing \n)
    with session_b.open("a") as f:
        f.write(_user("b-u1", "hi from b", "2026-07-18T00:00:02Z") + "\n")
        f.write(_assistant("b-a1", "hello b", "2026-07-18T00:00:03Z") + "\n")
    with session_a.open("a") as f:
        f.write(_user("a-u2", "second turn", "2026-07-18T00:00:04Z") + "\n")
        f.write(_assistant("a-a2", "second reply", "2026-07-18T00:00:05Z")[:15])  # partial, no \n
    assert watcher.scan_once() == 1  # only session_b's complete turn counted
    await capture.flush()
    assert conn.execute("SELECT count(*) FROM traces").fetchone()[0] == 2

    rows = conn.execute("SELECT response_json FROM traces").fetchall()
    for (response_json,) in rows:
        json.loads(response_json)  # every persisted row must be complete, parseable JSON

    # "tick" 3: session_a's partial line finally completes
    with session_a.open("a") as f:
        f.write(_assistant("a-a2", "second reply", "2026-07-18T00:00:05Z")[15:] + "\n")
    assert watcher.scan_once() == 1
    await capture.flush()
    assert conn.execute("SELECT count(*) FROM traces").fetchone()[0] == 3

    ingest_keys = {r[0] for r in conn.execute("SELECT ingest_key FROM traces").fetchall()}
    assert ingest_keys == {"a-a1", "b-a1", "a-a2"}  # no duplicates, no partial-row artifacts

    # a fourth scan with no new data must be a true no-op
    assert watcher.scan_once() == 0
