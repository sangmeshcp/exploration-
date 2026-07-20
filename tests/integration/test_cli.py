import contextlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from shadowtrace.cli import main
from shadowtrace.db import duckdb_store
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "shadowtrace_home"
    monkeypatch.setenv("SHADOWTRACE_HOME", str(home))
    from shadowtrace.common.config import reset_settings_cache

    reset_settings_cache()
    return home


def test_pause_and_resume_toggle_bypass_flag(_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["pause"])
    assert result.exit_code == 0
    assert (_env / "PAUSED").exists()

    result = runner.invoke(main, ["resume"])
    assert result.exit_code == 0
    assert not (_env / "PAUSED").exists()


def test_sync_runs_etl_and_reports_counts(_env: Path) -> None:
    import asyncio

    writer = CaptureWriter(_env / "capture.sqlite3", _env / "spill", flush_interval=100.0)
    writer.enqueue(
        TraceRecord(
            id="01AAA0",
            ts=1000,
            source="proxy",
            model="x",
            request_json={"role": "user", "content": "hi"},
            response_json={"content": []},
        )
    )
    asyncio.run(writer.flush())

    runner = CliRunner()
    result = runner.invoke(main, ["sync"])
    assert result.exit_code == 0
    assert "inserted=1" in result.output


def test_report_shows_trace_count(_env: Path) -> None:
    import asyncio

    writer = CaptureWriter(_env / "capture.sqlite3", _env / "spill", flush_interval=100.0)
    writer.enqueue(
        TraceRecord(
            id="01AAA1",
            ts=1000,
            source="transcript",
            model="x",
            request_json={"role": "user", "content": "hi"},
            response_json={"content": []},
        )
    )
    asyncio.run(writer.flush())

    runner = CliRunner()
    result = runner.invoke(main, ["report"])
    assert result.exit_code == 0
    assert "traces: 1" in result.output


def test_report_raw_reads_sqlite_directly(_env: Path) -> None:
    import asyncio

    writer = CaptureWriter(_env / "capture.sqlite3", _env / "spill", flush_interval=100.0)
    writer.enqueue(
        TraceRecord(
            id="01AAA2",
            ts=1000,
            source="transcript",
            model="x",
            request_json={"role": "user", "content": "hi"},
            response_json={"content": []},
        )
    )
    asyncio.run(writer.flush())

    runner = CliRunner()
    result = runner.invoke(main, ["report", "--raw"])
    assert result.exit_code == 0
    assert "transcript: 1" in result.output


def test_apply_dry_run_then_confirmed(_env: Path) -> None:
    conn = duckdb_store.connect(_env / "analytics.duckdb")
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES ('a1', 'Code', False, ?, ?)",
        [now, now],
    )
    conn.execute(
        "INSERT INTO recommendations (id, archetype_id, candidate, currency, value, pass_rate, "
        "pass_rate_ci_low, pass_rate_ci_high, latency_delta_ms, created_ts, expires_ts) "
        "VALUES ('rec1', 'a1', 'claude-haiku-4-5', 'usd_per_month', 1.0, 0.95, 0.9, 0.98, NULL, ?, ?)",
        [now, now + 1_000_000],
    )
    conn.close()

    output_path = _env / "claude_settings.json"
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["apply", "rec1", "--writer", "claude_code", "--output-path", str(output_path)],
        input="n\n",
    )
    assert result.exit_code == 0
    assert not output_path.exists()

    result2 = runner.invoke(
        main,
        ["apply", "rec1", "--writer", "claude_code", "--output-path", str(output_path), "--yes"],
    )
    assert result2.exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["model_routing"]["Code"] == "claude-haiku-4-5"


def test_apply_unknown_recommendation_errors(_env: Path) -> None:
    duckdb_store.connect(_env / "analytics.duckdb").close()
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "apply",
            "does-not-exist",
            "--writer",
            "claude_code",
            "--output-path",
            str(_env / "x.json"),
        ],
    )
    assert result.exit_code != 0


def test_metrics_command_outputs_json(_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["metrics"])
    assert result.exit_code == 0
    json.loads(result.output)  # must be valid JSON


def test_cluster_with_no_data_does_not_crash(_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["cluster"])
    assert result.exit_code == 0
    assert "no prompt text" in result.output


def test_run_with_no_data_does_not_crash(_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--budget", "1.0"])
    assert result.exit_code == 0
    assert "no eligible traces" in result.output


async def test_up_wires_real_tracing_into_proxy_and_watcher(_env: Path) -> None:
    """Regression test: `shadow up` must actually wire a real tracer into
    both the proxy and the transcript watcher (not silently fall back to
    the no-op default), since the README documents proxy/ingest tracing
    as on-by-default. Exercises build_up_components() directly rather
    than booting the real uvicorn server `up()` runs."""
    import httpx

    from shadowtrace.cli import build_up_components
    from shadowtrace.common.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()
    app, watcher = build_up_components(settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        # any request is fine here — we only care that a proxy_request span
        # gets emitted, not that it succeeds against a real upstream
        with contextlib.suppress(httpx.HTTPError):
            await client.post("/v1/messages", json={"model": "x", "messages": []}, timeout=2.0)

    watcher.scan_once()

    assert settings.traces_path.exists()
    lines = settings.traces_path.read_text().strip().split("\n")
    span_names = {json.loads(line)["name"] for line in lines if line}
    assert "proxy_request" in span_names


def test_ingest_backfills_existing_transcript_history(
    _env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """shadow ingest must pick up transcripts that already existed before
    it ran (TranscriptFile always starts at byte 0 on first sight), not
    just messages appended afterward — this is the whole point of the
    command."""
    projects_dir = _env.parent / "claude_projects"
    projects_dir.mkdir()
    (projects_dir / "old_session.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "u1",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "message": {"role": "user", "content": "an old pre-existing question"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "a1",
                        "timestamp": "2026-01-01T00:00:01Z",
                        "message": {
                            "role": "assistant",
                            "model": "claude-sonnet-5",
                            "content": [{"type": "text", "text": "an old pre-existing answer"}],
                            "usage": {"input_tokens": 5, "output_tokens": 5},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("SHADOWTRACE_CLAUDE_PROJECTS_DIR", str(projects_dir))
    from shadowtrace.common.config import reset_settings_cache

    reset_settings_cache()

    runner = CliRunner()
    result = runner.invoke(main, ["ingest"])
    assert result.exit_code == 0
    assert "ingested 1 messages" in result.output
    assert "inserted=1" in result.output

    conn = duckdb_store.connect(_env / "analytics.duckdb")
    row = conn.execute("SELECT source, model FROM traces").fetchone()
    assert row == ("transcript", "claude-sonnet-5")

    # a second run re-reads the file from byte 0 again (each invocation is a
    # fresh process with no persisted per-file offset — see
    # ingest/claude_transcripts.py's module docstring), so it re-*parses*
    # the same message, but the DB-level unique index on (source,
    # ingest_key) makes the write itself idempotent: no duplicate row.
    result2 = runner.invoke(main, ["ingest"])
    assert result2.exit_code == 0
    assert "ingested 1 messages" in result2.output
    assert "inserted=0" in result2.output
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 1


def test_ingest_reset_wipes_stale_data_before_reingesting(
    _env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--reset exists specifically for recovering from a fixed transcript
    parser: normal `ingest` dedups by message id and never touches rows
    it already wrote, so a row captured with a since-fixed bug in
    request_json construction is stuck that way forever unless the store
    is wiped and rebuilt from the original (untouched) transcript files."""
    projects_dir = _env.parent / "claude_projects_reset"
    projects_dir.mkdir()
    (projects_dir / "session.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "u1",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "message": {"role": "user", "content": "a fresh real question"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "a1",
                        "timestamp": "2026-01-01T00:00:01Z",
                        "message": {
                            "role": "assistant",
                            "model": "claude-sonnet-5",
                            "content": [{"type": "text", "text": "a fresh real answer"}],
                            "usage": {"input_tokens": 5, "output_tokens": 5},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    monkeypatch.setenv("SHADOWTRACE_CLAUDE_PROJECTS_DIR", str(projects_dir))
    from shadowtrace.common.config import reset_settings_cache

    reset_settings_cache()

    # simulate stale, buggily-captured data already sitting in the store
    writer = CaptureWriter(_env / "capture.sqlite3", _env / "spill", flush_interval=100.0)
    writer.enqueue(
        TraceRecord(
            id="01STALE0",
            ts=500,
            source="transcript",
            model="claude-sonnet-5",
            request_json={"role": "user", "content": None},  # the bug: no prompt captured
            response_json={"content": []},
            ingest_key="stale-uuid",
        )
    )
    import asyncio

    asyncio.run(writer.flush())

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--reset"])
    assert result.exit_code == 0
    assert "reset: capture store wiped" in result.output
    assert "ingested 1 messages" in result.output

    conn = duckdb_store.connect(_env / "analytics.duckdb")
    rows = conn.execute("SELECT id, model FROM traces").fetchall()
    assert len(rows) == 1  # the stale row is gone, only the fresh re-ingest remains
    assert rows[0][0] != "01STALE0"
