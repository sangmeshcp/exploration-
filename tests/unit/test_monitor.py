from pathlib import Path
from typing import Any

import pytest

from shadowtrace.db import duckdb_store
from shadowtrace.monitor import drift, performance, quality
from tests.fakes.analytics import DAY_MS, NOW, add_archetype, add_replay_result, add_trace


@pytest.fixture
def conn(tmp_path: Path) -> Any:
    return duckdb_store.connect(tmp_path / "a.duckdb")


def test_psi_and_js_reference_behaviour() -> None:
    same = {"a": 50.0, "b": 50.0}
    assert drift.psi(same, same) == pytest.approx(0.0)
    assert drift.js_divergence(same, same) == pytest.approx(0.0)

    shifted = drift.psi({"a": 90.0, "b": 10.0}, {"a": 10.0, "b": 90.0})
    assert shifted > drift.PSI_ALERT
    # disjoint supports: PSI is large but finite, JS hits its upper bound
    assert drift.js_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert drift.psi({}, {}) == 0.0
    assert drift.js_divergence({}, {"a": 1.0}) == 0.0


def test_drift_report_detects_new_task_mix_and_reports_insufficient_data(conn: Any) -> None:
    add_archetype(conn, "code")
    add_archetype(conn, "writing")
    for i in range(60):  # baseline: mostly code
        add_trace(conn, f"b{i}", NOW - 10 * DAY_MS, "code" if i < 54 else "writing", tokens_in=100)
    for i in range(40):  # recent: mostly writing + unassigned, longer prompts
        arch = "writing" if i < 20 else None if i < 35 else "code"
        add_trace(conn, f"r{i}", NOW - DAY_MS, arch, tokens_in=5000)

    report = {r.feature: r for r in drift.drift_report(conn, now_ms=NOW)}
    mix = report["archetype_mix"]
    assert mix.status == "significant"
    assert mix.psi is not None and mix.psi > drift.PSI_ALERT
    # code 90% -> 12.5%, writing 10% -> 50%, unassigned 0% -> 37.5%
    assert [s["key"] for s in mix.top_shifts] == ["code", "writing", "(unassigned)"]
    assert report["prompt_length"].status == "significant"
    assert report["model_mix"].status == "stable"

    sparse = drift.drift_report(conn, now_ms=NOW, min_rows=1000)
    assert all(r.status == "insufficient_data" and r.psi is None for r in sparse)


def test_data_quality_checks(conn: Any) -> None:
    add_archetype(conn, "a1")
    add_trace(conn, "ok", NOW - DAY_MS, "a1")
    add_trace(conn, "no_tokens", NOW - DAY_MS, "a1", tokens_in=None)
    add_trace(conn, "empty", NOW - DAY_MS, None, response="")
    add_trace(conn, "unknown", NOW - DAY_MS, "a1", model="unknown")
    add_trace(conn, "quarantined", NOW - DAY_MS, None, quarantined=True)
    add_trace(conn, "future", NOW + DAY_MS, "a1")
    add_trace(conn, "old", NOW - 30 * DAY_MS, None, tokens_in=None)  # outside the window

    checks = {c.name: c for c in quality.data_quality(conn, now_ms=NOW)}
    assert checks["missing_token_counts_ratio"].value == pytest.approx(1 / 6)
    assert checks["missing_token_counts_ratio"].ok is False
    assert checks["empty_response_ratio"].value == pytest.approx(1 / 6)
    assert checks["unknown_model_ratio"].value == pytest.approx(1 / 6)
    assert checks["quarantined_ratio"].value == pytest.approx(1 / 6)
    assert checks["quarantined_ratio"].ok is True
    # "empty" is the one non-quarantined unassigned trace among 5 replayable ones
    assert checks["unassigned_traces_ratio"].value == pytest.approx(1 / 5)
    assert checks["future_timestamp_ratio"].ok is False


def test_data_quality_without_data_or_archetypes(conn: Any) -> None:
    checks = quality.data_quality(conn, now_ms=NOW)
    assert all(c.value is None and c.ok is None for c in checks)
    add_trace(conn, "t", NOW - DAY_MS)
    unassigned = {c.name: c for c in quality.data_quality(conn, now_ms=NOW)}
    assert unassigned["unassigned_traces_ratio"].value is None  # nothing clustered yet


def test_performance_report_flags_significant_week_over_week_drop(conn: Any) -> None:
    add_archetype(conn, "a1", "Code")
    for i in range(30):  # previous week: 29/30 pass
        add_replay_result(conn, f"p{i}", "a1", "haiku", i != 0, NOW - 10 * DAY_MS)
        add_replay_result(conn, f"s{i}", "a1", "sonnet", True, NOW - 10 * DAY_MS)
    for i in range(30):  # this week: haiku 18/30, sonnet unchanged
        add_replay_result(conn, f"r{i}", "a1", "haiku", i < 18, NOW - DAY_MS)
        add_replay_result(conn, f"q{i}", "a1", "sonnet", True, NOW - DAY_MS)
    conn.execute(
        "INSERT INTO spot_checks (id, archetype_id, candidate, trace_id, result, ts) "
        "VALUES ('sc', 'a1', 'haiku', 't', 'fail', ?)",
        [NOW - DAY_MS],
    )

    report = {p.candidate: p for p in performance.performance_report(conn, now_ms=NOW)}
    haiku = report["haiku"]
    assert haiku.recent_n == 31  # replay + spot-check outcomes combined
    assert haiku.recent_pass_rate == pytest.approx(18 / 31)
    assert haiku.previous_pass_rate == pytest.approx(29 / 30)
    assert haiku.degraded and haiku.z is not None and haiku.z < -1.645
    assert haiku.recent_ci is not None and haiku.recent_ci[0] < haiku.recent_pass_rate
    assert haiku.mean_latency_ms == pytest.approx(100.0)
    assert not report["sonnet"].degraded
    assert report["sonnet"].archetype_label == "Code"


def test_performance_needs_minimum_samples(conn: Any) -> None:
    add_replay_result(conn, "x", "a1", "c", False, NOW - DAY_MS)
    (pair,) = performance.performance_report(conn, now_ms=NOW)
    assert pair.z is None and not pair.degraded
    assert performance.two_proportion_z(0, 0, 1, 1) is None
    assert performance.two_proportion_z(5, 5, 5, 5) == 0.0
