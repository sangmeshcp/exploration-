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
