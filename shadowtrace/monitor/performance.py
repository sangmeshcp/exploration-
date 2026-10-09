"""Model performance over time, per (archetype, candidate).

Combines offline replay verdicts with online spot-checks of applied policies
and flags pairs whose pass rate in the most recent window dropped
significantly against the window before it (one-sided two-proportion
z-test). This catches a provider silently changing a model, or a candidate
that was fine on old traffic failing on new traffic, before the
spot-check regression watch's fixed floor is crossed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.recommend.engine import wilson_ci

DAY_MS = 86_400_000


@dataclass
class PairPerformance:
    archetype_id: str
    archetype_label: str
    candidate: str
    recent_n: int
    recent_pass_rate: float | None
    recent_ci: tuple[float, float] | None
    previous_n: int
    previous_pass_rate: float | None
    mean_latency_ms: float | None
    z: float | None
    degraded: bool


def two_proportion_z(pass_a: int, n_a: int, pass_b: int, n_b: int) -> float | None:
    """z for (rate_b - rate_a); negative when b is worse than a."""
    if not n_a or not n_b:
        return None
    pooled = (pass_a + pass_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0
    return (pass_b / n_b - pass_a / n_a) / se


def performance_report(
    conn: Any,
    now_ms: int | None = None,
    window_days: int = 7,
    z_threshold: float = 1.645,
    min_n: int = 10,
) -> list[PairPerformance]:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    recent_start = now - window_days * DAY_MS
    previous_start = recent_start - window_days * DAY_MS
    rows = conn.execute(
        """
        WITH outcomes AS (
            SELECT archetype_id, candidate, ts, verdict = 'pass' AS passed, latency_ms
            FROM replay_results WHERE grader != 'chain'
            UNION ALL
            SELECT archetype_id, candidate, ts, result = 'pass' AS passed, NULL AS latency_ms
            FROM spot_checks
        )
        SELECT o.archetype_id, coalesce(a.label, o.archetype_id), o.candidate,
               count(*) FILTER (WHERE o.ts >= ?),
               count(*) FILTER (WHERE o.ts >= ? AND o.passed),
               count(*) FILTER (WHERE o.ts < ?),
               count(*) FILTER (WHERE o.ts < ? AND o.passed),
               avg(o.latency_ms) FILTER (WHERE o.ts >= ?)
        FROM outcomes o LEFT JOIN archetypes a ON a.id = o.archetype_id
        WHERE o.ts >= ? AND o.ts <= ?
        GROUP BY 1, 2, 3
        ORDER BY 2, 3
        """,
        [recent_start] * 5 + [previous_start, now],
    ).fetchall()

    report = []
    degraded_count = 0
    for arch_id, label, candidate, rn, rp, pn, pp, latency in rows:
        z = two_proportion_z(pp, pn, rp, rn) if rn >= min_n and pn >= min_n else None
        degraded = z is not None and z <= -z_threshold
        degraded_count += int(degraded)
        report.append(
            PairPerformance(
                archetype_id=arch_id,
                archetype_label=label,
                candidate=candidate,
                recent_n=rn,
                recent_pass_rate=rp / rn if rn else None,
                recent_ci=wilson_ci(rp, rn) if rn else None,
                previous_n=pn,
                previous_pass_rate=pp / pn if pn else None,
                mean_latency_ms=float(latency) if latency is not None else None,
                z=z,
                degraded=degraded,
            )
        )
    REGISTRY.set_gauge("degraded_candidate_pairs", float(degraded_count))
    return report
