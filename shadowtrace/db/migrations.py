"""Versioned schema migrations for both stores (plan.md §3.3, E14).

Each store tracks its own integer schema version and migrates forward
in-place — a populated database is upgraded, not recreated, so real
captured history survives a version bump. `BASE_VERSION` is the schema
shipped in `sqlite_schema.sql` / `duckdb_schema.sql` (both files use
`CREATE TABLE IF NOT EXISTS`, so they're always safe to re-run
unconditionally on every connect before migrations are considered);
anything past that is a real structural change (e.g. `ALTER TABLE ...
ADD COLUMN`) registered here, guarded so it's also safe to re-run.

`_v2_add_notes_column` is a real (if minor) migration — a nullable
`notes` column for future manual trace annotation — included specifically
to exercise this framework against a populated table rather than leaving
it as untested scaffolding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Migration = Callable[[Any], None]

BASE_VERSION = 1

# --- SQLite ---

SQLITE_CURRENT_VERSION = 2


def _sqlite_v2_add_notes_column(conn: Any) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(traces)").fetchall()}
    if "notes" not in cols:
        conn.execute("ALTER TABLE traces ADD COLUMN notes TEXT")


SQLITE_MIGRATIONS: dict[int, Migration] = {
    2: _sqlite_v2_add_notes_column,
}


def get_sqlite_version(conn: Any) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def migrate_sqlite(conn: Any, target_version: int = SQLITE_CURRENT_VERSION) -> int:
    current = get_sqlite_version(conn)
    if current == 0:
        # Fresh DB: the base schema was just applied unconditionally by
        # connect() via CREATE TABLE IF NOT EXISTS — record that as BASE_VERSION
        # rather than replaying a "version 1 migration" that doesn't exist.
        current = BASE_VERSION
        conn.execute(f"PRAGMA user_version = {current}")
    for version in range(current + 1, target_version + 1):
        migration = SQLITE_MIGRATIONS.get(version)
        if migration is not None:
            migration(conn)
        conn.execute(f"PRAGMA user_version = {version}")
    return get_sqlite_version(conn)


# --- DuckDB ---

DUCKDB_CURRENT_VERSION = 2


def _duckdb_v2_add_notes_column(conn: Any) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info('traces')").fetchall()}
    if "notes" not in cols:
        conn.execute("ALTER TABLE traces ADD COLUMN notes VARCHAR")


DUCKDB_MIGRATIONS: dict[int, Migration] = {
    2: _duckdb_v2_add_notes_column,
}


def get_duckdb_version(conn: Any) -> int:
    row = conn.execute("SELECT version FROM schema_meta").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_meta (version) VALUES (?)", [BASE_VERSION])
        return BASE_VERSION
    return int(row[0])


def migrate_duckdb(conn: Any, target_version: int = DUCKDB_CURRENT_VERSION) -> int:
    current = get_duckdb_version(conn)
    for version in range(current + 1, target_version + 1):
        migration = DUCKDB_MIGRATIONS.get(version)
        if migration is not None:
            migration(conn)
    if target_version > current:
        conn.execute("UPDATE schema_meta SET version = ?", [target_version])
    return get_duckdb_version(conn)
