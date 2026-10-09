"""Human-in-the-loop review of judge verdicts (plan.md M4.5, review R4).

The judge is the weakest link in replay grading: it is itself a model, often
from the same family as the frontier baseline. This module samples results
where the judge's verdict was decisive (every deterministic scorer passed,
so pass/fail came down to the judge), queues them for a human label, and
measures how well the judge agrees with those labels:

  agreement        share of reviewed results where judge == human
  cohen_kappa      agreement corrected for chance (1 = perfect, 0 = chance)
  false_pass_rate  share of judge PASSes a human marked FAIL — the costly
                   error, since it routes traffic to a model that can't do
                   the job
  false_fail_rate  share of judge FAILs a human marked PASS — wasted savings

A low kappa or a high false-pass rate means SPRT verdicts built on that
judge shouldn't be trusted yet; `monitor/slo.py` alerts on both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.messages import parse_response, request_text

_DETERMINISTIC_SCORERS = ("format", "completion", "trajectory")


def _now_ms() -> int:
    return int(time.time() * 1000)


def enqueue(conn: Any, per_archetype: int = 10) -> int:
    """Queue up to `per_archetype` judge-decided results per archetype that
    aren't queued or reviewed yet, most recent first. Returns how many
    were added."""
    placeholders = ", ".join("?" for _ in _DETERMINISTIC_SCORERS)
    rows = conn.execute(
        f"""
        SELECT r.id, r.archetype_id
        FROM replay_results r
        JOIN eval_scores j ON j.replay_result_id = r.id AND j.scorer = 'judge'
        WHERE r.grader != 'chain'
          AND r.id NOT IN (SELECT replay_result_id FROM human_reviews)
          AND NOT EXISTS (
              SELECT 1 FROM eval_scores d
              WHERE d.replay_result_id = r.id AND d.scorer IN ({placeholders}) AND NOT d.passed
          )
        ORDER BY r.ts DESC
        """,
        list(_DETERMINISTIC_SCORERS),
    ).fetchall()
    existing = dict(
        conn.execute(
            "SELECT r.archetype_id, count(*) FROM human_reviews h "
            "JOIN replay_results r ON r.id = h.replay_result_id GROUP BY 1"
        ).fetchall()
    )
    added = 0
    now = _now_ms()
    for result_id, archetype_id in rows:
        if existing.get(archetype_id, 0) >= per_archetype:
            continue
        conn.execute(
            "INSERT INTO human_reviews (replay_result_id, queued_ts) VALUES (?, ?)",
            [result_id, now],
        )
        existing[archetype_id] = existing.get(archetype_id, 0) + 1
        added += 1
    REGISTRY.set_gauge("human_reviews_pending", float(len(pending(conn, limit=10_000))))
    return added


@dataclass
class ReviewItem:
    replay_result_id: str
    archetype_label: str
    candidate: str
    request: str
    frontier_answer: str
    candidate_answer: str
    judge_passed: bool


def pending(conn: Any, limit: int = 20) -> list[ReviewItem]:
    rows = conn.execute(
        """
        SELECT r.id, coalesce(a.label, r.archetype_id), r.candidate, t.request_json,
               t.response_json, coalesce(r.response_text, ''), j.passed
        FROM human_reviews h
        JOIN replay_results r ON r.id = h.replay_result_id
        JOIN eval_scores j ON j.replay_result_id = r.id AND j.scorer = 'judge'
        LEFT JOIN traces t ON t.id = r.source_trace_id
        LEFT JOIN archetypes a ON a.id = r.archetype_id
        WHERE h.human_verdict IS NULL
        ORDER BY h.queued_ts, r.id
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    return [
        ReviewItem(
            replay_result_id=r[0],
            archetype_label=r[1],
            candidate=r[2],
            request=request_text(r[3]) if r[3] is not None else "",
            frontier_answer=parse_response(r[4]).text if r[4] is not None else "",
            candidate_answer=r[5],
            judge_passed=bool(r[6]),
        )
        for r in rows
    ]


def record(conn: Any, replay_result_id: str, passed: bool, note: str | None = None) -> None:
    updated = conn.execute(
        "UPDATE human_reviews SET human_verdict = ?, note = ?, reviewed_ts = ? "
        "WHERE replay_result_id = ? RETURNING replay_result_id",
        ["pass" if passed else "fail", note, _now_ms(), replay_result_id],
    ).fetchall()
    if not updated:
        raise KeyError(f"{replay_result_id!r} is not in the review queue")
    REGISTRY.inc("human_reviews_total", labels={"verdict": "pass" if passed else "fail"})


@dataclass
class AgreementStats:
    n: int
    agreement: float | None
    cohen_kappa: float | None
    false_pass_rate: float | None
    false_fail_rate: float | None


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    p_a = sum(1 for a, _ in pairs if a) / n
    p_b = sum(1 for _, b in pairs if b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(conn: Any, archetype_id: str | None = None) -> AgreementStats:
    where = "AND r.archetype_id = ?" if archetype_id else ""
    params = [archetype_id] if archetype_id else []
    rows = conn.execute(
        f"""
        SELECT j.passed, h.human_verdict = 'pass'
        FROM human_reviews h
        JOIN replay_results r ON r.id = h.replay_result_id
        JOIN eval_scores j ON j.replay_result_id = r.id AND j.scorer = 'judge'
        WHERE h.human_verdict IS NOT NULL {where}
        """,
        params,
    ).fetchall()
    pairs = [(bool(judge_pass), bool(human_pass)) for judge_pass, human_pass in rows]
    if not pairs:
        return AgreementStats(0, None, None, None, None)
    n = len(pairs)
    judge_passes = [h for j, h in pairs if j]
    judge_fails = [h for j, h in pairs if not j]
    stats = AgreementStats(
        n=n,
        agreement=sum(1 for j, h in pairs if j == h) / n,
        cohen_kappa=cohen_kappa(pairs),
        false_pass_rate=(
            sum(1 for h in judge_passes if not h) / len(judge_passes) if judge_passes else None
        ),
        false_fail_rate=(
            sum(1 for h in judge_fails if h) / len(judge_fails) if judge_fails else None
        ),
    )
    if archetype_id is None:
        if stats.agreement is not None:
            REGISTRY.set_gauge("judge_human_agreement", stats.agreement)
        if stats.cohen_kappa is not None:
            REGISTRY.set_gauge("judge_cohen_kappa", stats.cohen_kappa)
        if stats.false_pass_rate is not None:
            REGISTRY.set_gauge("judge_false_pass_rate", stats.false_pass_rate)
    return stats
