import sqlite3
from pathlib import Path
from typing import Any

import pytest

from shadowtrace.common.metrics import REGISTRY, MetricsRegistry
from shadowtrace.db import duckdb_store
from shadowtrace.monitor import slo
from tests.fakes.analytics import DAY_MS, NOW, add_trace


@pytest.fixture
def conn(tmp_path: Path) -> Any:
    return duckdb_store.connect(tmp_path / "a.duckdb")


def _objective(**kwargs: Any) -> slo.Objective:
    base: dict[str, Any] = {
        "name": "o",
        "sli": "fail_open_events_total",
        "op": "==",
        "threshold": 0,
    }
    base.update(kwargs)
    return slo.Objective(**base)


def test_default_rules_load_and_every_sli_exists() -> None:
    objectives = slo.load_objectives()
    names = {o.name for o in objectives}
    assert {"capture_lag", "proxy_overhead", "fail_open_triggered", "judge_unreliable"} <= names
    assert all(o.sli in slo.SLI_REGISTRY for o in objectives)
    assert all(o.target is not None for o in objectives if o.kind == "slo")


def test_user_rules_file_replaces_defaults_and_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        "objectives:\n  - {name: x, sli: disk_free_gb, op: '>=', threshold: 1, kind: alert}\n"
    )
    (only,) = slo.load_objectives(path)
    assert only.name == "x" and only.severity == "warning"
    assert slo.load_objectives(tmp_path / "missing.yaml")  # falls back to defaults

    for bad, message in [
        ("{name: x, sli: disk_free_gb, op: '~', threshold: 1}", "unknown op"),
        ("{name: x, sli: nope, op: '<', threshold: 1}", "unknown SLI"),
        ("{name: x, sli: disk_free_gb, op: '<', threshold: 1, kind: slo}", "needs a target"),
    ]:
        with pytest.raises(ValueError, match=message):
            slo.parse_objectives(f"objectives:\n  - {bad}\n")


def test_evaluate_statuses(conn: Any, tmp_path: Path) -> None:
    registry = MetricsRegistry()
    registry.inc("fail_open_events_total")
    registry.inc("operations_total", labels={"status": "ok", "operation": "a"}, value=9)
    registry.inc("operations_total", labels={"status": "error", "operation": "a"})
    for v in range(1, 101):
        registry.observe("proxy_added_latency_ms", float(v))

    def boom(ctx: slo.SLIContext) -> float:
        raise RuntimeError("broken SLI")

    slo.SLI_REGISTRY["test_boom"] = boom
    try:
        ctx = slo.SLIContext(conn=conn, registry=registry, home=tmp_path, now_ms=NOW)
        results = {
            e.name: e
            for e in slo.evaluate(
                [
                    _objective(name="fail_open"),
                    _objective(name="spill", sli="spill_rows_total"),
                    _objective(name="errors", sli="operation_error_ratio", op="<=", threshold=0.05),
                    _objective(
                        name="p99", sli="proxy_added_latency_p99_ms", op="<=", threshold=150
                    ),
                    _objective(name="disk", sli="disk_free_gb", op=">=", threshold=0),
                    _objective(name="boom", sli="test_boom"),
                ],
                ctx,
            )
        }
    finally:
        del slo.SLI_REGISTRY["test_boom"]

    assert results["fail_open"].status == "breach" and results["fail_open"].value == 1
    assert results["spill"].status == "no_data"
    assert results["errors"].status == "breach"
    assert results["errors"].value == pytest.approx(0.1)
    assert results["p99"].status == "ok" and results["p99"].value == 99
    assert results["disk"].status == "ok"
    assert results["boom"].status == "error"
    assert REGISTRY.find("slo_breached", labels={"objective": "fail_open"}) == 1.0


def test_store_backed_slis(conn: Any, tmp_path: Path) -> None:
    sqlite_path = tmp_path / "capture.sqlite3"
    src = sqlite3.connect(sqlite_path)
    src.execute("CREATE TABLE traces (ts INTEGER)")
    src.execute("INSERT INTO traces VALUES (?)", [NOW - 1000])
    src.commit()
    src.close()

    ctx = slo.SLIContext(conn=conn, sqlite_path=sqlite_path, now_ms=NOW)
    assert slo.SLI_REGISTRY["etl_lag_seconds"](ctx) == pytest.approx(1.0)  # nothing ETL'd yet
    add_trace(conn, "t", NOW - 61_000)
    assert slo.SLI_REGISTRY["etl_lag_seconds"](ctx) == pytest.approx(60.0)

    conn.execute(
        "INSERT INTO applied_policies (id, archetype_id, candidate, writer, diff, "
        "previous_content, applied_ts, reverted_ts) VALUES "
        "('p1', 'a1', 'c', 'w', '', '', 0, ?), ('p2', 'a1', 'c', 'w', '', '', 0, NULL)",
        [NOW - DAY_MS],
    )
    assert slo.SLI_REGISTRY["policies_reverted_7d"](ctx) == 1
    assert slo.SLI_REGISTRY["expired_applied_policies"](ctx) == 1  # p2 has no live rec
    conn.execute(
        "INSERT INTO recommendations (id, archetype_id, candidate, currency, value, pass_rate, "
        "pass_rate_ci_low, pass_rate_ci_high, created_ts, expires_ts) "
        "VALUES ('r', 'a1', 'c', 'usd_per_month', 0, 1, 1, 1, 0, ?)",
        [NOW + DAY_MS],
    )
    assert slo.SLI_REGISTRY["expired_applied_policies"](ctx) == 0
    assert slo.SLI_REGISTRY["degraded_candidate_pairs"](ctx) is None  # no evidence yet
    assert slo.SLI_REGISTRY["judge_human_agreement"](ctx) is None

    no_store = slo.SLIContext()
    for name in ("etl_lag_seconds", "archetype_mix_psi", "judge_false_pass_rate", "disk_free_gb"):
        assert slo.SLI_REGISTRY[name](no_store) is None


def test_record_and_error_budget(conn: Any) -> None:
    objective = _objective(name="lag", kind="slo", target=0.9, window_days=7)
    for i in range(20):
        status = "breach" if i < 1 else "ok"
        ev = slo.Evaluation("lag", "slo", "warning", "s", 1.0, "<=", 1, status, "", NOW - i * 1000)
        slo.record(conn, [ev])
    slo.record(conn, [slo.Evaluation("lag", "slo", "w", "s", None, "<=", 1, "no_data", "", NOW)])
    old = slo.Evaluation("lag", "slo", "w", "s", 9.0, "<=", 1, "breach", "", NOW - 30 * DAY_MS)
    slo.record(conn, [old])  # outside the 7-day window

    c = slo.compliance(conn, objective, now_ms=NOW)
    assert c.n == 20
    assert c.attainment == pytest.approx(0.95)
    # allowed failure rate 10%, used 5% -> half the budget left
    assert c.error_budget_remaining == pytest.approx(0.5)

    empty = slo.compliance(conn, _objective(name="other", kind="slo", target=0.9), now_ms=NOW)
    assert empty.attainment is None and empty.error_budget_remaining is None

    strict = slo.compliance(conn, _objective(name="lag", kind="slo", target=1.0), now_ms=NOW)
    assert strict.error_budget_remaining == 0.0


def test_notifier_fires_on_transitions_only() -> None:
    sent: list[tuple[str, str]] = []

    def fake_notify(title: str, body: str) -> bool:
        sent.append((title, body))
        return True

    notifier = slo.AlertNotifier(desktop=True, notify_fn=fake_notify)

    def ev(status: str, severity: str = "critical") -> list[slo.Evaluation]:
        return [slo.Evaluation("disk", "alert", severity, "s", 1.0, ">=", 2, status, "low", NOW)]

    assert len(notifier.process(ev("breach"))) == 1
    assert notifier.process(ev("breach")) == []  # still firing: no repeat
    assert notifier.process(ev("ok")) == []
    assert len(notifier.process(ev("breach"))) == 1
    assert sent == [("shadowtrace: disk", "low"), ("shadowtrace: disk", "low")]

    quiet = slo.AlertNotifier(desktop=True, notify_fn=fake_notify)
    quiet.process(ev("breach", severity="warning"))
    assert len(sent) == 2  # only critical alerts reach the desktop


def test_desktop_notify_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(slo.shutil, "which", lambda _: None)
    assert slo.desktop_notify("t", "b") is False
