"""Distribution drift between a baseline window and a recent window.

Recommendations are only as good as the traffic they were evaluated on: if
what you ask models to do shifts, a candidate approved for last month's mix
may be wrong for this month's. Three features are tracked:

  archetype_mix  which task archetypes the traffic falls into ("(unassigned)"
                 included: a growing unassigned share means new kinds of
                 work the archetypes, and so the replay evidence, don't cover)
  model_mix      which models are serving the traffic
  prompt_length  input-token counts, in power-of-two buckets

Each is scored with the Population Stability Index, using the standard
reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
Jensen-Shannon divergence (base 2, bounded in [0, 1]) is reported alongside
because it stays finite and symmetric when a category appears or vanishes.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from shadowtrace.common.metrics import REGISTRY

PSI_WARN = 0.1
PSI_ALERT = 0.25
MIN_WINDOW_ROWS = 30
DAY_MS = 86_400_000


def _shares(counts: dict[str, float], keys: list[str], eps: float) -> list[float]:
    total = sum(counts.values())
    return [max(counts.get(k, 0.0) / total, eps) if total else eps for k in keys]


def psi(baseline: dict[str, float], recent: dict[str, float], eps: float = 1e-4) -> float:
    keys = sorted(set(baseline) | set(recent))
    if not keys:
        return 0.0
    b = _shares(baseline, keys, eps)
    r = _shares(recent, keys, eps)
    return sum((ri - bi) * math.log(ri / bi) for bi, ri in zip(b, r, strict=True))


def js_divergence(baseline: dict[str, float], recent: dict[str, float]) -> float:
    keys = sorted(set(baseline) | set(recent))
    tb, tr = sum(baseline.values()), sum(recent.values())
    if not keys or not tb or not tr:
        return 0.0
    p = [baseline.get(k, 0.0) / tb for k in keys]
    q = [recent.get(k, 0.0) / tr for k in keys]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q, strict=True)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b, strict=True) if ai > 0)

    return max(0.0, min(1.0, (kl(p, m) + kl(q, m)) / 2))


@dataclass
class DriftResult:
    feature: str
    psi: float | None
    js_divergence: float | None
    status: str  # stable | moderate | significant | insufficient_data
    baseline_n: int
    recent_n: int
    top_shifts: list[dict[str, Any]] = field(default_factory=list)


def compare(
    feature: str, baseline: dict[str, float], recent: dict[str, float], min_rows: int
) -> DriftResult:
    baseline_n, recent_n = int(sum(baseline.values())), int(sum(recent.values()))
    if baseline_n < min_rows or recent_n < min_rows:
        return DriftResult(feature, None, None, "insufficient_data", baseline_n, recent_n)
    score = psi(baseline, recent)
    status = "significant" if score > PSI_ALERT else "moderate" if score > PSI_WARN else "stable"
    deltas = []
    for key in set(baseline) | set(recent):
        b = baseline.get(key, 0.0) / baseline_n
        r = recent.get(key, 0.0) / recent_n
        deltas.append((abs(r - b), key, b, r))
    deltas.sort(reverse=True)
    shifts: list[dict[str, Any]] = [
        {"key": key, "baseline_share": b, "recent_share": r, "delta": r - b}
        for _, key, b, r in deltas
    ]
    REGISTRY.set_gauge("drift_psi", score, labels={"feature": feature})
    return DriftResult(
        feature, score, js_divergence(baseline, recent), status, baseline_n, recent_n, shifts[:5]
    )


def _length_bucket(tokens: int | None) -> str:
    if tokens is None or tokens <= 0:
        return "unknown"
    upper = 1 << max(0, math.ceil(math.log2(tokens)))
    return f"<={upper}"


def _windowed_counts(
    conn: Any, sql_expr: str, now_ms: int, recent_days: int, baseline_days: int
) -> tuple[dict[str, float], dict[str, float]]:
    recent_start = now_ms - recent_days * DAY_MS
    baseline_start = recent_start - baseline_days * DAY_MS
    rows = conn.execute(
        f"""
        SELECT {sql_expr} AS k, t.ts >= ? AS is_recent
        FROM traces t
        LEFT JOIN archetype_assignments aa ON aa.trace_id = t.id
        LEFT JOIN archetypes a ON a.id = aa.archetype_id
        WHERE t.ts >= ? AND t.ts <= ?
        """,
        [recent_start, baseline_start, now_ms],
    ).fetchall()
    baseline: Counter[str] = Counter()
    recent: Counter[str] = Counter()
    for key, is_recent in rows:
        (recent if is_recent else baseline)[str(key)] += 1
    return dict(baseline), dict(recent)


def drift_report(
    conn: Any,
    now_ms: int | None = None,
    recent_days: int = 7,
    baseline_days: int = 28,
    min_rows: int = MIN_WINDOW_ROWS,
) -> list[DriftResult]:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    results = []
    archetype_expr = "coalesce(coalesce(a.merged_into, a.id), '(unassigned)')"
    for feature, expr in (("archetype_mix", archetype_expr), ("model_mix", "t.model")):
        baseline, recent = _windowed_counts(conn, expr, now, recent_days, baseline_days)
        results.append(compare(feature, baseline, recent, min_rows))

    recent_start = now - recent_days * DAY_MS
    rows = conn.execute(
        "SELECT tokens_in, ts >= ? FROM traces WHERE ts >= ? AND ts <= ?",
        [recent_start, recent_start - baseline_days * DAY_MS, now],
    ).fetchall()
    base_len: Counter[str] = Counter()
    recent_len: Counter[str] = Counter()
    for tokens_in, is_recent in rows:
        (recent_len if is_recent else base_len)[_length_bucket(tokens_in)] += 1
    results.append(compare("prompt_length", dict(base_len), dict(recent_len), min_rows))
    return results
