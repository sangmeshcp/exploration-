from pathlib import Path

from shadowtrace.db import duckdb_store
from shadowtrace.recommend import engine


def _seed_verdict(conn, archetype_id: str, candidate: str, n_samples: int, n_pass: int) -> None:  # type: ignore[no-untyped-def]
    conn.execute(
        "INSERT INTO sprt_verdicts (archetype_id, candidate, run_id, state, n_samples, n_pass, "
        "log_likelihood_ratio, ts) VALUES (?, ?, 'run-1', 'pass', ?, ?, 0.0, 0)",
        [archetype_id, candidate, n_samples, n_pass],
    )


def test_select_currency_by_access_mode() -> None:
    assert engine.select_currency("subscription") == "quota_headroom_pct"
    assert engine.select_currency("api_key") == "usd_per_month"


def test_wilson_ci_bounds_are_sane() -> None:
    low, high = engine.wilson_ci(19, 20)
    assert 0.0 <= low < 0.95 < high <= 1.0

    low0, high0 = engine.wilson_ci(0, 0)
    assert (low0, high0) == (0.0, 1.0)


def test_is_expired() -> None:
    rec = engine.Recommendation(
        id="r1",
        archetype_id="a",
        candidate="c",
        currency="usd_per_month",
        value=1.0,
        pass_rate=0.9,
        pass_rate_ci_low=0.8,
        pass_rate_ci_high=0.95,
        latency_delta_ms=None,
        created_ts=1000,
        expires_ts=2000,
    )
    assert engine.is_expired(rec, now_ms=2500) is True
    assert engine.is_expired(rec, now_ms=1500) is False


def test_build_recommendations_subscription_currency_is_quota_headroom(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES ('arch-1', 'Code', False, ?, ?)",
        [now, now],
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO traces (id, ts, source, model, request_json, response_json, tokens_in, tokens_out) "
            "VALUES (?, ?, 'transcript', 'x', '{}', '{}', 100000, 50000)",
            [f"t{i}", now + i],
        )
        conn.execute(
            "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) VALUES (?, 'arch-1', 0.1, ?)",
            [f"t{i}", now],
        )
    _seed_verdict(conn, "arch-1", "claude-haiku-4-5", n_samples=20, n_pass=19)

    recs = engine.build_recommendations(
        conn, access_mode="subscription", quota_period_tokens=1_000_000, now_ms=now
    )
    assert len(recs) == 1
    rec = recs[0]
    assert rec.currency == "quota_headroom_pct"
    # 3 traces * 150,000 tokens = 450,000 / 1,000,000 quota => 45%
    assert abs(rec.value - 45.0) < 1e-6
    assert rec.pass_rate == 0.95

    stored = conn.execute("SELECT count(*) FROM recommendations").fetchone()[0]
    assert stored == 1


def test_build_recommendations_api_key_currency_is_usd(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES ('arch-1', 'Code', False, ?, ?)",
        [now, now],
    )
    for i in range(2):
        conn.execute(
            "INSERT INTO traces (id, ts, source, model, request_json, response_json, cost_usd) "
            "VALUES (?, ?, 'proxy', 'x', '{}', '{}', 0.10)",
            [f"t{i}", now + i],
        )
        conn.execute(
            "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) VALUES (?, 'arch-1', 0.1, ?)",
            [f"t{i}", now],
        )
    conn.execute(
        "INSERT INTO replay_results (id, archetype_id, candidate, source_trace_id, verdict, cost_usd, "
        "latency_ms, grader, run_id, ts) VALUES ('r1', 'arch-1', 'claude-haiku-4-5', 't0', 'pass', 0.02, 10, 'judge', 'run-1', ?)",
        [now],
    )
    _seed_verdict(conn, "arch-1", "claude-haiku-4-5", n_samples=10, n_pass=9)

    recs = engine.build_recommendations(conn, access_mode="api_key", now_ms=now)
    assert recs[0].currency == "usd_per_month"
    # avg frontier cost 0.10, avg candidate cost 0.02, 2 calls -> (0.10-0.02)*2 = 0.16
    assert abs(recs[0].value - 0.16) < 1e-9


def test_regression_watch_requires_full_window(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    for _ in range(5):
        engine.record_spot_check(conn, "arch-1", "claude-haiku-4-5", "t0", passed=False)
    assert (
        engine.check_regression(conn, "arch-1", "claude-haiku-4-5", floor=0.9, window=20) is False
    )


def test_regression_watch_detects_breach(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    for i in range(20):
        engine.record_spot_check(conn, "arch-1", "claude-haiku-4-5", f"t{i}", passed=(i % 2 == 0))
    # 10/20 = 50% pass rate, well below a 90% floor
    assert engine.check_regression(conn, "arch-1", "claude-haiku-4-5", floor=0.9, window=20) is True


def test_regression_watch_no_breach_when_healthy(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    for i in range(20):
        engine.record_spot_check(conn, "arch-1", "claude-haiku-4-5", f"t{i}", passed=True)
    assert (
        engine.check_regression(conn, "arch-1", "claude-haiku-4-5", floor=0.9, window=20) is False
    )
