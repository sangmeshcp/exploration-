"""E6: redaction end-to-end. Seeded secrets in prompts must never appear
anywhere in SQLite, DuckDB, or logs after a full ingest -> capture -> ETL
run; quarantined (high-entropy, non-pattern) rows are preserved but
excluded from replay sampling (plan.md §3.3, §3.5).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
from pathlib import Path

from shadowtrace.common.config import Settings
from shadowtrace.common.logging import get_logger
from shadowtrace.db import duckdb_store
from shadowtrace.ingest.claude_transcripts import TranscriptWatcher
from shadowtrace.ingest.etl import run_etl
from shadowtrace.proxy.capture import CaptureWriter
from shadowtrace.replay.sampler import sample_traces

SECRET = "AKIAABCDEFGHIJKLMNOP"


def _transcript_line(entry_type: str, uuid: str, ts: str, **kwargs: object) -> str:
    return json.dumps({"type": entry_type, "uuid": uuid, "timestamp": ts, **kwargs})


def test_no_known_secret_leaks_into_log_output() -> None:
    logger = get_logger("test.redaction_e2e_logging")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)

    logger.warning("captured payload containing secret %s here", SECRET)

    output = stream.getvalue()
    assert SECRET not in output
    assert "REDACTED" in output


async def _run_pipeline(tmp_path: Path, code_hash: str) -> Settings:
    projects_dir = tmp_path / "claude_projects"
    projects_dir.mkdir()
    session = projects_dir / "session1.jsonl"
    lines = [
        _transcript_line(
            "user",
            "u1",
            "2026-07-18T00:00:00Z",
            message={"role": "user", "content": f"my key is {SECRET} please help"},
        ),
        _transcript_line(
            "assistant",
            "a1",
            "2026-07-18T00:00:01Z",
            message={
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "ok, noted"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        ),
        _transcript_line(
            "user",
            "u2",
            "2026-07-18T00:00:02Z",
            message={"role": "user", "content": f"look at this hash {code_hash}"},
        ),
        _transcript_line(
            "assistant",
            "a2",
            "2026-07-18T00:00:03Z",
            message={
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "looks fine"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        ),
    ]
    session.write_text("\n".join(lines) + "\n")

    settings = Settings(home=tmp_path / "shadowtrace_home")
    settings.ensure_dirs()

    capture = CaptureWriter(settings.sqlite_path, settings.spill_dir, flush_interval=100.0)
    watcher = TranscriptWatcher(projects_dir, capture)
    watcher.scan_once()
    await capture.flush()
    run_etl(settings.sqlite_path, settings.duckdb_path)
    return settings


async def test_redaction_e2e_no_secret_leak_and_quarantine_excluded_from_sampler(
    tmp_path: Path,
) -> None:
    code_hash = hashlib.sha256(b"some proprietary source code blob").hexdigest()
    settings = await _run_pipeline(tmp_path, code_hash)

    # 1. semantic check: query every stored text column in both stores
    sconn = sqlite3.connect(settings.sqlite_path)
    for request_json, response_json in sconn.execute(
        "SELECT request_json, response_json FROM traces"
    ).fetchall():
        assert SECRET not in request_json
        assert SECRET not in response_json

    dconn = duckdb_store.connect(settings.duckdb_path)
    for request_json, response_json in dconn.execute(
        "SELECT request_json, response_json FROM traces"
    ).fetchall():
        assert SECRET not in request_json
        assert SECRET not in response_json

    # 2. defense-in-depth: raw byte scan of both database files on disk
    assert SECRET.encode() not in settings.sqlite_path.read_bytes()
    assert SECRET.encode() not in settings.duckdb_path.read_bytes()

    # 3. the high-entropy (but non-pattern) hash is quarantined, not deleted
    quarantined = dconn.execute(
        "SELECT request_json FROM traces WHERE quarantined = TRUE"
    ).fetchall()
    assert len(quarantined) == 1
    assert code_hash in quarantined[0][0]  # preserved, just flagged

    # 4. quarantined rows never reach the replay sampler
    dconn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) "
        "VALUES ('a1', 'X', False, 0, 0)"
    )
    for (trace_id,) in dconn.execute("SELECT id FROM traces").fetchall():
        dconn.execute(
            "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
            "VALUES (?, 'a1', 0.1, 0)",
            [trace_id],
        )
    samples = sample_traces(dconn, n_per_archetype=10, seed=1)
    sampled_ids = {s.trace_id for s in samples.get("a1", [])}
    quarantined_ids = {
        r[0] for r in dconn.execute("SELECT id FROM traces WHERE quarantined = TRUE").fetchall()
    }
    assert sampled_ids.isdisjoint(quarantined_ids)
    assert len(sampled_ids) == 1  # only the non-quarantined trace was sampled
