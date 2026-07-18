"""Stratified sampling per archetype for shadow replay (plan.md M3.1)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class SampleRow:
    trace_id: str
    archetype_id: str
    ts: int
    request_json: str
    response_json: str  # the frontier response — free baseline for the judge


def sample_traces(
    conn: Any,
    n_per_archetype: int = 10,
    seed: int = 0,
    archetype_id: str | None = None,
) -> dict[str, list[SampleRow]]:
    """Stratified by archetype: up to `n_per_archetype` traces per
    archetype, quarantined traces excluded entirely. Recency is
    prioritized directly (the most recent half of the quota is always
    included); the rest is a seeded random sample of the remainder so
    higher-volume archetypes get proportionally more diversity without
    needing an explicit weighting step. Deterministic for a given
    (seed, data) pair.
    """
    where_archetype = "AND aa.archetype_id = ?" if archetype_id else ""
    params: list[Any] = [archetype_id] if archetype_id else []
    rows = conn.execute(
        f"""
        SELECT t.id, aa.archetype_id, t.ts, t.request_json, t.response_json
        FROM traces t
        JOIN archetype_assignments aa ON aa.trace_id = t.id
        JOIN archetypes a ON a.id = aa.archetype_id AND a.merged_into IS NULL
        WHERE t.quarantined = FALSE
        {where_archetype}
        ORDER BY t.ts DESC
        """,
        params,
    ).fetchall()

    by_archetype: dict[str, list[SampleRow]] = {}
    for trace_id, arch_id, ts, request_json, response_json in rows:
        by_archetype.setdefault(arch_id, []).append(
            SampleRow(trace_id, arch_id, ts, request_json, response_json)
        )

    result: dict[str, list[SampleRow]] = {}
    for arch_id, pool in by_archetype.items():
        rng = random.Random(f"{seed}:{arch_id}")
        recent_n = min(len(pool), max(1, n_per_archetype // 2))
        recent = pool[:recent_n]
        remaining_pool = pool[recent_n:]
        extra_n = max(0, n_per_archetype - len(recent))
        extra = rng.sample(remaining_pool, min(extra_n, len(remaining_pool)))
        result[arch_id] = recent + extra

    return result
