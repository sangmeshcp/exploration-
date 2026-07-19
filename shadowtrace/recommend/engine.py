"""Objective-aware recommendation engine (plan.md M4.1/M4.2).

Currency selection follows D1: subscription access is flat-rate, so the
primary currency is *quota headroom* ("route archetypes X, Y to
Haiku/local -> reclaim ~N% of your 5-hour/weekly limits"); dollar savings
are only meaningful — and only shown — for the API-key lane, where traces
carry a real `cost_usd`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid
from shadowtrace.recommend import apply

EXPIRY_DAYS = 45
REGRESSION_SPOT_CHECK_WINDOW = 20


def select_currency(access_mode: str) -> str:
    return "quota_headroom_pct" if access_mode == "subscription" else "usd_per_month"


def wilson_ci(n_pass: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion — the pass-rate CI
    shown on each recommendation card."""
    if n == 0:
        return (0.0, 1.0)
    phat = n_pass / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class Recommendation:
    id: str
    archetype_id: str
    candidate: str
    currency: str
    value: float
    pass_rate: float
    pass_rate_ci_low: float
    pass_rate_ci_high: float
    latency_delta_ms: int | None
    created_ts: int
    expires_ts: int


def is_expired(rec: Recommendation, now_ms: int | None = None) -> bool:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return now >= rec.expires_ts


def _quota_headroom_value(conn: Any, archetype_id: str, quota_period_tokens: int) -> float:
    tokens_in, tokens_out = conn.execute(
        "SELECT coalesce(sum(t.tokens_in), 0), coalesce(sum(t.tokens_out), 0) "
        "FROM traces t JOIN archetype_assignments aa ON aa.trace_id = t.id "
        "WHERE aa.archetype_id = ?",
        [archetype_id],
    ).fetchone()
    total_tokens = (tokens_in or 0) + (tokens_out or 0)
    if quota_period_tokens <= 0:
        return 0.0
    return min(100.0, 100.0 * total_tokens / quota_period_tokens)


def _usd_per_month_value(conn: Any, archetype_id: str, candidate: str) -> float:
    avg_candidate_cost = (
        conn.execute(
            "SELECT avg(cost_usd) FROM replay_results WHERE archetype_id = ? AND candidate = ?",
            [archetype_id, candidate],
        ).fetchone()[0]
        or 0.0
    )
    avg_frontier_cost, call_count = conn.execute(
        "SELECT avg(t.cost_usd), count(*) FROM traces t "
        "JOIN archetype_assignments aa ON aa.trace_id = t.id "
        "WHERE aa.archetype_id = ? AND t.cost_usd IS NOT NULL",
        [archetype_id],
    ).fetchone()
    avg_frontier_cost = avg_frontier_cost or 0.0
    call_count = call_count or 0
    return max(0.0, (avg_frontier_cost - avg_candidate_cost) * call_count)


def build_recommendations(
    conn: Any,
    access_mode: str = "subscription",
    quota_period_tokens: int = 5_000_000,
    now_ms: int | None = None,
) -> list[Recommendation]:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    expires = now + EXPIRY_DAYS * 86_400_000
    currency = select_currency(access_mode)

    verdicts = conn.execute(
        "SELECT archetype_id, candidate, n_samples, n_pass FROM sprt_verdicts WHERE state = 'pass'"
    ).fetchall()

    recommendations: list[Recommendation] = []
    for archetype_id, candidate, n_samples, n_pass in verdicts:
        pass_rate = n_pass / n_samples if n_samples else 0.0
        ci_low, ci_high = wilson_ci(n_pass, n_samples)

        if currency == "quota_headroom_pct":
            value = _quota_headroom_value(conn, archetype_id, quota_period_tokens)
        else:
            value = _usd_per_month_value(conn, archetype_id, candidate)

        rec = Recommendation(
            id=new_ulid(),
            archetype_id=archetype_id,
            candidate=candidate,
            currency=currency,
            value=value,
            pass_rate=pass_rate,
            pass_rate_ci_low=ci_low,
            pass_rate_ci_high=ci_high,
            latency_delta_ms=None,
            created_ts=now,
            expires_ts=expires,
        )
        conn.execute(
            "INSERT INTO recommendations "
            "(id, archetype_id, candidate, currency, value, pass_rate, pass_rate_ci_low, "
            "pass_rate_ci_high, latency_delta_ms, created_ts, expires_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rec.id,
                rec.archetype_id,
                rec.candidate,
                rec.currency,
                rec.value,
                rec.pass_rate,
                rec.pass_rate_ci_low,
                rec.pass_rate_ci_high,
                rec.latency_delta_ms,
                rec.created_ts,
                rec.expires_ts,
            ],
        )
        recommendations.append(rec)

    REGISTRY.set_gauge("recommendations_pending", len(recommendations))
    return recommendations


def record_spot_check(
    conn: Any, archetype_id: str, candidate: str, trace_id: str, passed: bool
) -> None:
    conn.execute(
        "INSERT INTO spot_checks (id, archetype_id, candidate, trace_id, result, ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            new_ulid(),
            archetype_id,
            candidate,
            trace_id,
            "pass" if passed else "fail",
            int(time.time() * 1000),
        ],
    )
    REGISTRY.inc("spot_checks_total", labels={"result": "pass" if passed else "fail"})


def check_regression(
    conn: Any,
    archetype_id: str,
    candidate: str,
    floor: float,
    window: int = REGRESSION_SPOT_CHECK_WINDOW,
) -> bool:
    """Returns True if the most recent spot-check window's pass rate has
    dropped below `floor` — the applied-policy floor this candidate was
    approved against. The dashboard's regression-watch alert + one-click
    revert (M4.4) is driven by this signal."""
    rows = conn.execute(
        "SELECT result FROM spot_checks WHERE archetype_id = ? AND candidate = ? "
        "ORDER BY ts DESC LIMIT ?",
        [archetype_id, candidate, window],
    ).fetchall()
    if len(rows) < window:
        return False  # not enough data yet to judge a regression
    pass_count = sum(1 for (result,) in rows if result == "pass")
    breached = (pass_count / len(rows)) < floor
    if breached:
        REGISTRY.inc("regression_alerts_total")
    return breached


def watch_and_revert(
    conn: Any,
    writers: dict[str, apply.Writer],
    applied_policy_id: str,
    archetype_id: str,
    candidate: str,
    floor: float,
    window: int = REGRESSION_SPOT_CHECK_WINDOW,
) -> bool:
    """The M4.4 regression-watch loop, wired end to end: if the spot-check
    window has fallen below the floor this candidate was approved against,
    one-click revert the applied policy and return True. A no-op (returns
    False, nothing touched) if there isn't a full window of spot-check
    data yet or the pass rate is still healthy — `applied_policies` and
    the live router config are left exactly as they were.
    """
    if not check_regression(conn, archetype_id, candidate, floor, window):
        return False
    apply.revert_policy(
        conn, writers, applied_policy_id, reason=f"regression: pass rate below floor {floor:.2f}"
    )
    return True
