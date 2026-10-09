"""Helpers that seed a DuckDB analytics store for monitoring/eval tests."""

from __future__ import annotations

import json
from typing import Any

DAY_MS = 86_400_000
NOW = 1_800_000_000_000


def add_archetype(conn: Any, archetype_id: str, label: str | None = None) -> None:
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) "
        "VALUES (?, ?, FALSE, 0, 0)",
        [archetype_id, label or archetype_id],
    )


def add_trace(
    conn: Any,
    trace_id: str,
    ts: int,
    archetype_id: str | None = None,
    model: str = "claude-sonnet-5",
    tokens_in: int | None = 100,
    tokens_out: int | None = 50,
    response: str = "an answer",
    quarantined: bool = False,
    request: str = "a question",
) -> None:
    conn.execute(
        "INSERT INTO traces (id, ts, source, model, request_json, response_json, tokens_in, "
        "tokens_out, quarantined) VALUES (?, ?, 'transcript', ?, ?, ?, ?, ?, ?)",
        [
            trace_id,
            ts,
            model,
            json.dumps({"role": "user", "content": request}),
            json.dumps({"role": "assistant", "content": [{"type": "text", "text": response}]}),
            tokens_in,
            tokens_out,
            quarantined,
        ],
    )
    if archetype_id is not None:
        conn.execute(
            "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
            "VALUES (?, ?, 0.1, 0)",
            [trace_id, archetype_id],
        )


def add_replay_result(
    conn: Any,
    result_id: str,
    archetype_id: str,
    candidate: str,
    passed: bool,
    ts: int,
    source_trace_id: str = "t0",
    scores: dict[str, bool] | None = None,
    grader: str = "suite",
    response_text: str = "candidate answer",
) -> None:
    conn.execute(
        "INSERT INTO replay_results (id, archetype_id, candidate, source_trace_id, verdict, "
        "cost_usd, latency_ms, grader, run_id, ts, response_text) "
        "VALUES (?, ?, ?, ?, ?, 0.0, 100, ?, 'run', ?, ?)",
        [
            result_id,
            archetype_id,
            candidate,
            source_trace_id,
            "pass" if passed else "fail",
            grader,
            ts,
            response_text,
        ],
    )
    for scorer, scorer_passed in (scores or {}).items():
        conn.execute(
            "INSERT INTO eval_scores (replay_result_id, run_id, archetype_id, candidate, scorer, "
            "value, passed, detail, ts) VALUES (?, 'run', ?, ?, ?, ?, ?, '', ?)",
            [
                result_id,
                archetype_id,
                candidate,
                scorer,
                1.0 if scorer_passed else 0.0,
                scorer_passed,
                ts,
            ],
        )
