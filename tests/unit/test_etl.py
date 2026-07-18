import asyncio
from pathlib import Path

import pytest

from shadowtrace.db import duckdb_store
from shadowtrace.ingest import etl
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord


def _seed_sqlite(db_path: Path, spill_dir: Path, n: int) -> None:
    writer = CaptureWriter(db_path, spill_dir, flush_interval=100.0)
    for i in range(n):
        writer.enqueue(
            TraceRecord(
                id=f"01AAA{i:03d}",
                ts=1000 + i,
                source="proxy",
                model="claude-haiku-4-5",
                request_json={"role": "user", "content": f"question {i}"},
                response_json={"content": [{"type": "text", "text": "answer"}]},
            )
        )
    asyncio.run(writer.flush())


def test_etl_copies_rows_into_duckdb(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "cap.sqlite3"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_sqlite(sqlite_path, tmp_path / "spill", 5)

    stats = etl.run_etl(sqlite_path, duckdb_path)
    assert stats.rows_scanned == 5
    assert stats.rows_inserted == 5
    assert stats.rows_deduped == 0

    conn = duckdb_store.connect(duckdb_path)
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 5


def test_etl_is_idempotent_on_rerun(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "cap.sqlite3"
    duckdb_path = tmp_path / "analytics.duckdb"
    _seed_sqlite(sqlite_path, tmp_path / "spill", 5)

    etl.run_etl(sqlite_path, duckdb_path)
    conn = duckdb_store.connect(duckdb_path)
    first_state = conn.execute("SELECT id, ts, model FROM traces ORDER BY id").fetchall()
    conn.close()

    stats2 = etl.run_etl(sqlite_path, duckdb_path)
    assert stats2.rows_inserted == 0
    assert stats2.rows_deduped == 5

    conn = duckdb_store.connect(duckdb_path)
    second_state = conn.execute("SELECT id, ts, model FROM traces ORDER BY id").fetchall()
    assert first_state == second_state


def test_etl_handles_missing_sqlite_gracefully(tmp_path: Path) -> None:
    stats = etl.run_etl(tmp_path / "nope.sqlite3", tmp_path / "analytics.duckdb")
    assert stats.rows_scanned == 0
    assert stats.rows_inserted == 0


def test_extract_prompt_text_from_string_content() -> None:
    import json

    raw = json.dumps({"role": "user", "content": "hello world"})
    assert etl.extract_prompt_text(raw) == "hello world"


def test_extract_prompt_text_from_content_blocks() -> None:
    import json

    raw = json.dumps({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    assert etl.extract_prompt_text(raw) == "hi"


@pytest.mark.parametrize("raw", ["not json", "{}"])
def test_extract_prompt_text_returns_none_on_bad_input(raw: str) -> None:
    assert etl.extract_prompt_text(raw) is None
