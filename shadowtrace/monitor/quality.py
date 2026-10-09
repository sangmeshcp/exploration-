"""Data-quality checks on captured traces.

Everything downstream (clustering, sampling, grading, quota math) silently
degrades on bad input: rows without token counts under-report quota
headroom, empty responses become free "passes" for any candidate, and a
transcript format change shows up as a wave of unknown models long before
anyone notices clusters going strange. These checks make that visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.messages import parse_response

DAY_MS = 86_400_000
FUTURE_SKEW_MS = 5 * 60_000


@dataclass
class QualityCheck:
    name: str
    value: float | None
    threshold: float
    ok: bool | None  # None = no data in the window
    description: str


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _check(name: str, value: float | None, threshold: float, description: str) -> QualityCheck:
    ok = None if value is None else value <= threshold
    if value is not None:
        REGISTRY.set_gauge("data_quality", value, labels={"check": name})
    return QualityCheck(name, value, threshold, ok, description)


def data_quality(conn: Any, now_ms: int | None = None, window_days: int = 7) -> list[QualityCheck]:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    since = now - window_days * DAY_MS
    rows = conn.execute(
        """
        SELECT t.tokens_in, t.tokens_out, t.model, t.quarantined, t.response_json, t.ts,
               aa.trace_id IS NOT NULL AS assigned
        FROM traces t LEFT JOIN archetype_assignments aa ON aa.trace_id = t.id
        WHERE t.ts >= ?
        """,
        [since],
    ).fetchall()
    n = len(rows)
    any_assigned = conn.execute("SELECT count(*) FROM archetype_assignments").fetchone()[0] > 0

    missing_tokens = sum(1 for r in rows if r[0] is None or r[1] is None)
    unknown_model = sum(1 for r in rows if not r[2] or r[2] == "unknown")
    quarantined = sum(1 for r in rows if r[3])
    future = sum(1 for r in rows if r[5] > now + FUTURE_SKEW_MS)
    empty = 0
    for r in rows:
        parsed = parse_response(r[4])
        if not parsed.text.strip() and not parsed.tool_calls:
            empty += 1
    unassigned = sum(1 for r in rows if not r[6] and not r[3])
    eligible = n - quarantined

    return [
        _check(
            "missing_token_counts_ratio",
            _ratio(missing_tokens, n),
            0.05,
            "traces without token counts (quota headroom is under-reported)",
        ),
        _check(
            "empty_response_ratio",
            _ratio(empty, n),
            0.02,
            "traces whose response has no text and no tool calls",
        ),
        _check(
            "unknown_model_ratio",
            _ratio(unknown_model, n),
            0.01,
            "traces with no model id (often a transcript format change)",
        ),
        _check(
            "quarantined_ratio",
            _ratio(quarantined, n),
            0.20,
            "traces quarantined by entropy redaction (excluded from replay)",
        ),
        _check(
            "unassigned_traces_ratio",
            _ratio(unassigned, eligible) if any_assigned else None,
            0.20,
            "replayable traces outside every archetype (run `shadow cluster`)",
        ),
        _check(
            "future_timestamp_ratio",
            _ratio(future, n),
            0.0,
            "traces timestamped in the future (clock skew or parse error)",
        ),
    ]
