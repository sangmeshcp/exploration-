"""The evaluation + monitoring loop driven through the real CLI:
run -> recommend -> evals -> review -> monitor -> slo, plus the dashboard
endpoints that expose the same data."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from shadowtrace import cli
from shadowtrace.cli import main
from shadowtrace.common.config import get_settings, reset_settings_cache
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.dashboard.api import create_app
from shadowtrace.db import duckdb_store
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import CallResult
from shadowtrace.replay.sampler import SampleRow
from tests.fakes.analytics import add_archetype, add_trace


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("SHADOWTRACE_HOME", str(home))
    for var in (
        "ANTHROPIC_API_KEY",
        "SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY",
        "SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    reset_settings_cache()
    REGISTRY.reset()
    return home


def _seed(home: Path, n: int = 30) -> None:
    import time

    conn = duckdb_store.connect(home / "analytics.duckdb")
    add_archetype(conn, "code", "Code questions")
    now = int(time.time() * 1000)
    for i in range(n):
        add_trace(
            conn,
            f"t{i}",
            now - i * 1000,
            "code",
            request=f"how do I reverse list number {i}?",
            response=f"use reversed() on list number {i}",
        )
    conn.close()


def _fake_live_call(fail_candidate: str) -> Any:
    async def call(candidate: Candidate, sample: SampleRow) -> CallResult:
        from shadowtrace.replay.messages import parse_response

        reference = parse_response(sample.response_json).text
        text = "I can't help with that." if candidate.name == fail_candidate else reference
        return CallResult(text, tokens_in=1000, tokens_out=100, latency_ms=0)

    return call


def test_run_persists_scores_then_recommend_evals_review(
    _env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(_env)
    monkeypatch.setattr(cli, "_live_call", _fake_live_call(fail_candidate="claude-haiku-4-5"))
    runner = CliRunner()

    result = runner.invoke(
        main, ["run", "--budget", "5.0", "--n-per-archetype", "30", "--no-chains"]
    )
    assert result.exit_code == 0, result.output
    assert "judge: local heuristic" in result.output

    conn = duckdb_store.connect(_env / "analytics.duckdb")
    stored = conn.execute(
        "SELECT count(*), count(response_text) FROM replay_results WHERE grader = 'suite'"
    ).fetchone()
    assert stored[0] > 0 and stored[0] == stored[1]
    scorers = {r[0] for r in conn.execute("SELECT DISTINCT scorer FROM eval_scores").fetchall()}
    assert {"completion", "judge"} <= scorers
    refused = conn.execute(
        "SELECT count(*) FROM eval_scores WHERE candidate = 'claude-haiku-4-5' "
        "AND scorer = 'completion' AND NOT passed"
    ).fetchone()[0]
    assert refused > 0
    conn.close()

    rec = runner.invoke(main, ["recommend", "--access-mode", "api_key"])
    assert rec.exit_code == 0, rec.output
    assert "code -> ollama-llama3.1-8b" in rec.output
    assert "claude-haiku-4-5" not in rec.output
    again = runner.invoke(main, ["recommend"])
    assert "no new recommendations" in again.output  # idempotent

    ev = runner.invoke(main, ["evals"])
    assert ev.exit_code == 0
    assert "Code questions" in ev.output and "completion" in ev.output

    # label two judge-decided results: one agree, one disagree, then quit
    rv = runner.invoke(main, ["review", "--enqueue", "2", "--limit", "5"], input="p\nf\nq\n")
    assert rv.exit_code == 0, rv.output
    assert "queued 2 new" in rv.output
    assert "judge vs human (n=2): agreement=50%" in rv.output
    stats = runner.invoke(main, ["review", "--stats"])
    assert "n=2" in stats.output


def test_monitor_and_slo_commands(_env: Path) -> None:
    _seed(_env, n=5)
    runner = CliRunner()

    mon = runner.invoke(main, ["monitor"])
    assert mon.exit_code == 0, mon.output
    assert "archetype_mix" in mon.output and "insufficient_data" in mon.output
    assert "missing_token_counts_ratio" in mon.output
    data = json.loads(runner.invoke(main, ["monitor", "--json"]).output)
    assert {"drift", "quality", "performance"} == set(data)

    slo_out = runner.invoke(main, ["slo"])
    assert slo_out.exit_code == 0, slo_out.output  # nothing breached
    assert "NO_DATA" in slo_out.output and "capture_lag" in slo_out.output
    evaluations = json.loads(runner.invoke(main, ["slo", "--json"]).output)
    assert any(e["name"] == "low_disk" and e["status"] == "ok" for e in evaluations)

    rules = get_settings().slo_rules_path
    rules.write_text(
        "objectives:\n"
        "  - {name: always_breached, sli: disk_free_gb, op: '<', threshold: 0, kind: slo,"
        " target: 0.5}\n"
    )
    breached = runner.invoke(main, ["slo"])
    assert breached.exit_code == 1
    assert "BREACH" in breached.output and "budget left" in breached.output


@pytest.mark.asyncio
async def test_dashboard_observability_endpoints(_env: Path) -> None:
    _seed(_env, n=3)
    REGISTRY.inc("requests_total", labels={"status": "200"})
    app = create_app(get_settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dash"
    ) as client:
        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert 'shadowtrace_requests_total{status="200"} 1.0' in metrics.text

        ready = (await client.get("/readyz")).json()
        assert ready["checks"]["analytics_store"] is True
        assert set(ready["checks"]) == {"analytics_store", "state_dir_writable", "disk_headroom"}

        slo_body = (await client.get("/api/slo")).json()
        assert {e["name"] for e in slo_body["evaluations"]} >= {"capture_lag", "low_disk"}
        assert "capture_lag" in slo_body["compliance"]

        monitor = (await client.get("/api/monitor")).json()
        assert monitor["quality"][0]["name"] == "missing_token_counts_ratio"

        evals = (await client.get("/api/evals")).json()
        assert evals == {
            "scorers": [],
            "judge_agreement": {
                "n": 0,
                "agreement": None,
                "cohen_kappa": None,
                "false_pass_rate": None,
                "false_fail_rate": None,
            },
            "pending_reviews": 0,
        }


def test_evaluate_objectives_records_and_notifies(_env: Path) -> None:
    from shadowtrace.monitor import slo

    settings = get_settings()
    settings.ensure_dirs()
    REGISTRY.inc("fail_open_events_total")
    notifier = slo.AlertNotifier()
    evaluations = cli.evaluate_objectives(settings, notifier)
    assert any(e.name == "fail_open_triggered" and e.status == "breach" for e in evaluations)
    assert REGISTRY.find("alerts_fired_total", labels={"objective": "fail_open_triggered"}) == 1
    conn = duckdb_store.connect(settings.duckdb_path)
    assert conn.execute("SELECT count(*) FROM slo_evaluations").fetchone()[0] > 0
