"""Persists replay results and their per-scorer evaluation scores to DuckDB."""

from __future__ import annotations

from typing import Any

from shadowtrace.replay.runner import ChainReplayResult, ReplayResult


def save_replay_results(conn: Any, results: list[ReplayResult]) -> None:
    for r in results:
        conn.execute(
            "INSERT INTO replay_results (id, archetype_id, candidate, source_trace_id, verdict, "
            "cost_usd, latency_ms, grader, run_id, ts, response_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r.id,
                r.archetype_id,
                r.candidate,
                r.source_trace_id,
                r.verdict,
                r.cost_usd,
                r.latency_ms,
                r.grader,
                r.run_id,
                r.ts,
                r.response_text,
            ],
        )
        _save_scores(conn, r.id, r.run_id, r.archetype_id, r.candidate, r.ts, r.scores)


def save_chain_results(conn: Any, results: list[ChainReplayResult]) -> None:
    for cr in results:
        conn.execute(
            "INSERT INTO replay_results (id, archetype_id, candidate, source_trace_id, verdict, "
            "cost_usd, latency_ms, grader, run_id, ts, response_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                cr.id,
                cr.archetype_id,
                cr.candidate,
                cr.session_id,
                cr.verdict,
                cr.cost_usd,
                None,
                "chain",
                cr.run_id,
                cr.ts,
                cr.response_text,
            ],
        )
        _save_scores(conn, cr.id, cr.run_id, cr.archetype_id, cr.candidate, cr.ts, cr.scores)


def _save_scores(
    conn: Any,
    result_id: str,
    run_id: str,
    archetype_id: str,
    candidate: str,
    ts: int,
    scores: list[Any],
) -> None:
    for score in scores:
        conn.execute(
            "INSERT INTO eval_scores (replay_result_id, run_id, archetype_id, candidate, scorer, "
            "value, passed, detail, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [
                result_id,
                run_id,
                archetype_id,
                candidate,
                score.name,
                score.value,
                score.passed,
                score.detail,
                ts,
            ],
        )


def scorer_summary(conn: Any) -> list[dict[str, Any]]:
    """Mean score and pass rate per (archetype, candidate, scorer)."""
    rows = conn.execute(
        "SELECT e.archetype_id, coalesce(a.label, e.archetype_id), e.candidate, e.scorer, "
        "count(*), avg(e.value), avg(CASE WHEN e.passed THEN 1.0 ELSE 0.0 END) "
        "FROM eval_scores e LEFT JOIN archetypes a ON a.id = e.archetype_id "
        "GROUP BY 1, 2, 3, 4 ORDER BY 2, 3, 4"
    ).fetchall()
    return [
        {
            "archetype_id": r[0],
            "archetype_label": r[1],
            "candidate": r[2],
            "scorer": r[3],
            "n": int(r[4]),
            "mean_value": float(r[5]),
            "pass_rate": float(r[6]),
        }
        for r in rows
    ]
