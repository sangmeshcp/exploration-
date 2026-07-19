"""E14: run vN schema, apply vN+1 migration on a populated DB, data intact,
app boots (plan.md §3.3).
"""

import sqlite3
from importlib import resources
from pathlib import Path

from shadowtrace.db import duckdb_store
from shadowtrace.db import sqlite as sqlite_db
from shadowtrace.db.migrations import (
    SQLITE_CURRENT_VERSION,
    get_duckdb_version,
    get_sqlite_version,
    migrate_duckdb,
    migrate_sqlite,
)


def _raw_v1_sqlite(db_path: Path) -> sqlite3.Connection:
    """A hand-built v1 database — the base schema only, no migrations run —
    simulating a DB created by a prior version of this app."""
    schema = resources.files("shadowtrace.db").joinpath("sqlite_schema.sql").read_text()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema)
    conn.execute("PRAGMA user_version = 1")
    return conn


def test_fresh_sqlite_db_lands_on_current_version(tmp_path: Path) -> None:
    conn = sqlite_db.connect(tmp_path / "cap.sqlite3")
    assert get_sqlite_version(conn) == SQLITE_CURRENT_VERSION


def test_sqlite_migration_preserves_populated_data_and_adds_column(tmp_path: Path) -> None:
    db_path = tmp_path / "cap.sqlite3"
    conn = _raw_v1_sqlite(db_path)
    conn.execute(
        "INSERT INTO traces (id, ts, source, model, request_json, response_json) "
        "VALUES ('t1', 1000, 'proxy', 'x', '{\"a\": 1}', '{\"b\": 2}')"
    )
    conn.commit()
    conn.close()

    # reopen at v1, confirm starting state, then migrate forward
    conn2 = sqlite3.connect(str(db_path))
    assert conn2.execute("PRAGMA user_version").fetchone()[0] == 1
    cols_before = {row[1] for row in conn2.execute("PRAGMA table_info(traces)").fetchall()}
    assert "notes" not in cols_before

    new_version = migrate_sqlite(conn2)
    assert new_version == SQLITE_CURRENT_VERSION

    cols_after = {row[1] for row in conn2.execute("PRAGMA table_info(traces)").fetchall()}
    assert "notes" in cols_after

    row = conn2.execute(
        "SELECT id, request_json, response_json FROM traces WHERE id = 't1'"
    ).fetchone()
    assert row == ("t1", '{"a": 1}', '{"b": 2}')  # checksummable: data untouched by the migration
    conn2.close()

    # the app boots normally against the now-migrated file (db.sqlite.connect
    # re-runs the idempotent base schema + migrate_sqlite with no ill effect)
    conn3 = sqlite_db.connect(db_path)
    row3 = conn3.execute("SELECT id FROM traces WHERE id = 't1'").fetchone()
    assert row3["id"] == "t1"
    assert get_sqlite_version(conn3) == SQLITE_CURRENT_VERSION


def test_sqlite_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "cap.sqlite3"
    conn = _raw_v1_sqlite(db_path)
    conn.execute(
        "INSERT INTO traces (id, ts, source, model, request_json, response_json) "
        "VALUES ('t1', 1000, 'proxy', 'x', '{}', '{}')"
    )
    conn.commit()

    migrate_sqlite(conn)
    migrate_sqlite(conn)  # re-running must not error (e.g. duplicate ADD COLUMN)
    assert get_sqlite_version(conn) == SQLITE_CURRENT_VERSION


def test_duckdb_migration_preserves_data_and_adds_column(tmp_path: Path) -> None:
    db_path = tmp_path / "a.duckdb"
    conn = duckdb_store.connect(db_path)  # lands on current version already
    conn.execute(
        "INSERT INTO traces (id, ts, source, model, request_json, response_json) "
        "VALUES ('t1', 1000, 'proxy', 'x', '{\"a\": 1}', '{\"b\": 2}')"
    )

    # force it back down to v1 to simulate an old DB, then re-migrate
    conn.execute("UPDATE schema_meta SET version = 1")
    conn.execute("ALTER TABLE traces DROP COLUMN notes")

    new_version = migrate_duckdb(conn)
    assert new_version == 2
    assert get_duckdb_version(conn) == 2

    cols = {row[1] for row in conn.execute("PRAGMA table_info('traces')").fetchall()}
    assert "notes" in cols

    row = conn.execute(
        "SELECT id, request_json, response_json FROM traces WHERE id = 't1'"
    ).fetchone()
    assert row == ("t1", '{"a": 1}', '{"b": 2}')


def test_duckdb_migration_is_idempotent(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    conn.execute("UPDATE schema_meta SET version = 1")
    conn.execute("ALTER TABLE traces DROP COLUMN notes")

    migrate_duckdb(conn)
    migrate_duckdb(conn)
    assert get_duckdb_version(conn) == 2
