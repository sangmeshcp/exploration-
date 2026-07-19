"""Idempotent SQLite (hot path) → DuckDB (analytics) batch ETL (plan.md M1.8).

Idempotency is achieved structurally rather than via a watermark: every run
reads the full `traces` table and inserts with `ON CONFLICT (id) DO
NOTHING`, so running the job twice on unchanged data always yields
identical DuckDB state (§3.1 requirement). This trades a bit of redundant
scanning for a job that's trivially safe to re-run after a crash or a
manual `shadow report --resync`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.db import duckdb_store
from shadowtrace.db import sqlite as sqlite_db

logger = get_logger("ingest.etl")

_COLUMNS = [
    "id",
    "ts",
    "source",
    "tool",
    "model",
    "request_json",
    "response_json",
    "tokens_in",
    "tokens_out",
    "latency_ms",
    "cost_usd",
    "session_id",
    "parent_id",
    "redaction_flags",
    "quarantined",
]


@dataclass
class ETLStats:
    rows_scanned: int = 0
    rows_inserted: int = 0
    rows_deduped: int = 0
    lag_seconds: float = 0.0


def _fetch_sqlite_rows(sqlite_path: Path) -> list[sqlite3.Row]:
    if not sqlite_path.exists():
        return []
    conn = sqlite_db.connect(sqlite_path)
    try:
        return conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM traces ORDER BY ts").fetchall()
    finally:
        conn.close()


def run_etl(sqlite_path: Path, duckdb_path: Path) -> ETLStats:
    stats = ETLStats()
    try:
        rows = _fetch_sqlite_rows(sqlite_path)
    except Exception:
        logger.exception("failed to read sqlite capture store")
        REGISTRY.inc("etl_failures_total")
        return stats

    stats.rows_scanned = len(rows)
    if not rows:
        return stats

    try:
        conn = duckdb_store.connect(duckdb_path)
    except Exception:
        logger.exception("failed to open duckdb analytics store")
        REGISTRY.inc("etl_failures_total")
        return stats

    try:
        before = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        insert_sql = (
            f"INSERT INTO traces ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            "ON CONFLICT (id) DO NOTHING"
        )
        for row in rows:
            values: list[Any] = [row[c] for c in _COLUMNS]
            values[_COLUMNS.index("quarantined")] = bool(values[_COLUMNS.index("quarantined")])
            conn.execute(insert_sql, values)
        after = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
        stats.rows_inserted = after - before
        stats.rows_deduped = stats.rows_scanned - stats.rows_inserted
        max_ts = max(row["ts"] for row in rows)
        stats.lag_seconds = max(0.0, time.time() - max_ts / 1000)
        REGISTRY.inc("etl_rows_total", value=stats.rows_inserted)
        REGISTRY.inc("dedup_dropped_total", value=stats.rows_deduped)
        REGISTRY.set_gauge("etl_lag_seconds", stats.lag_seconds)
    except Exception:
        logger.exception("etl write to duckdb failed")
        REGISTRY.inc("etl_failures_total")
    finally:
        conn.close()

    return stats


def extract_prompt_text(request_json_raw: str) -> str | None:
    """Best-effort extraction of user-visible text from a stored request_json
    blob, for mining.embed to consume. Ignores system boilerplate."""
    try:
        data = json.loads(request_json_raw)
    except json.JSONDecodeError:
        return None
    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        return text or None
    return None


def backfill_prompt_text(duckdb_path: Path) -> int:
    conn = duckdb_store.connect(duckdb_path)
    try:
        rows = conn.execute(
            "SELECT id, request_json FROM traces WHERE prompt_text IS NULL"
        ).fetchall()
        updated = 0
        for trace_id, request_json_raw in rows:
            text = extract_prompt_text(request_json_raw)
            if text:
                conn.execute("UPDATE traces SET prompt_text = ? WHERE id = ?", [text, trace_id])
                updated += 1
        return updated
    finally:
        conn.close()
