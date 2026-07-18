"""Hot-path SQLite (WAL) connection helpers.

Kept intentionally thin: capture.py owns write batching/spill logic, this
module only owns "how do we open the DB and get the schema in place."
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


def _schema_sql() -> str:
    return resources.files("shadowtrace.db").joinpath("sqlite_schema.sql").read_text()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    conn.executescript(_schema_sql())
    return conn
