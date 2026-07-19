"""Hot-path SQLite (WAL) connection helpers.

Kept intentionally thin: capture.py owns write batching/spill logic, this
module only owns "how do we open the DB and get the schema in place."
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from shadowtrace.db.migrations import migrate_sqlite


def _schema_sql() -> str:
    return resources.files("shadowtrace.db").joinpath("sqlite_schema.sql").read_text()


def connect(db_path: Path, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=busy_timeout_ms / 1000, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.row_factory = sqlite3.Row
    conn.executescript(_schema_sql())
    migrate_sqlite(conn)
    return conn
