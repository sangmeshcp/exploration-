from pathlib import Path

from shadowtrace.db import sqlite as sqlite_db


def test_sqlite_connect_creates_wal_db(tmp_path: Path) -> None:
    conn = sqlite_db.connect(tmp_path / "capture.sqlite3")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "traces" in tables
    conn.close()


def test_sqlite_schema_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "capture.sqlite3"
    sqlite_db.connect(path).close()
    conn = sqlite_db.connect(path)  # re-running CREATE IF NOT EXISTS must not error
    conn.execute(
        "INSERT INTO traces (id, ts, source, model, request_json, response_json) "
        "VALUES ('01', 0, 'proxy', 'x', '{}', '{}')"
    )
    row = conn.execute("SELECT id FROM traces").fetchone()
    assert row["id"] == "01"
    conn.close()
