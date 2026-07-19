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
    session_id: str | None = None


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
        SELECT t.id, aa.archetype_id, t.ts, t.request_json, t.response_json, t.session_id
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
    for trace_id, arch_id, ts, request_json, response_json, session_id in rows:
        by_archetype.setdefault(arch_id, []).append(
            SampleRow(trace_id, arch_id, ts, request_json, response_json, session_id)
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


def top_archetypes_by_volume(conn: Any, n: int = 2) -> list[str]:
    """The archetypes eligible for chain replay (plan.md M3.6, review R6:
    restricted to the top-N by volume so chain-replay cost can't explode
    across the whole archetype set)."""
    rows = conn.execute(
        "SELECT aa.archetype_id, count(*) AS n FROM archetype_assignments aa "
        "JOIN archetypes a ON a.id = aa.archetype_id AND a.merged_into IS NULL "
        "GROUP BY aa.archetype_id ORDER BY n DESC LIMIT ?",
        [n],
    ).fetchall()
    return [r[0] for r in rows]


@dataclass
class ReplayChain:
    archetype_id: str
    session_id: str
    steps: list[SampleRow]  # ordered oldest-first


def sample_chains(
    conn: Any,
    archetype_ids: list[str],
    max_chains_per_archetype: int = 3,
    max_steps_per_chain: int = 5,
    seed: int = 0,
) -> list[ReplayChain]:
    """Group traces into ordered multi-step sessions for chain-replay mode.
    Only sessions with >= 2 steps qualify as a chain; single-step sessions
    are left for the ordinary single-call replay path in `sample_traces`.
    """
    chains: list[ReplayChain] = []
    for archetype_id in archetype_ids:
        rows = conn.execute(
            """
            SELECT t.id, aa.archetype_id, t.ts, t.request_json, t.response_json, t.session_id
            FROM traces t
            JOIN archetype_assignments aa ON aa.trace_id = t.id
            JOIN archetypes a ON a.id = aa.archetype_id AND a.merged_into IS NULL
            WHERE t.quarantined = FALSE AND aa.archetype_id = ? AND t.session_id IS NOT NULL
            ORDER BY t.session_id, t.ts ASC
            """,
            [archetype_id],
        ).fetchall()

        by_session: dict[str, list[SampleRow]] = {}
        for trace_id, arch_id, ts, request_json, response_json, session_id in rows:
            by_session.setdefault(session_id, []).append(
                SampleRow(trace_id, arch_id, ts, request_json, response_json, session_id)
            )

        eligible = sorted(sid for sid, steps in by_session.items() if len(steps) >= 2)
        rng = random.Random(f"{seed}:chains:{archetype_id}")
        rng.shuffle(eligible)
        for session_id in eligible[:max_chains_per_archetype]:
            chains.append(
                ReplayChain(
                    archetype_id=archetype_id,
                    session_id=session_id,
                    steps=by_session[session_id][:max_steps_per_chain],
                )
            )
    return chains
