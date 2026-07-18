"""Analytics store (DuckDB) connection helper."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any


def _schema_sql() -> str:
    return resources.files("shadowtrace.db").joinpath("duckdb_schema.sql").read_text()


def connect(db_path: Path) -> Any:
    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    for statement in _schema_sql().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    return conn
