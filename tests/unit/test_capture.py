import sqlite3
from pathlib import Path

import pytest

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.db import sqlite as sqlite_db
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    REGISTRY.reset()


def _record(id_: str = "01AAAA") -> TraceRecord:
    return TraceRecord(
        id=id_,
        ts=1000,
        source="proxy",
        model="claude-haiku-4-5",
        request_json={"messages": [{"role": "user", "content": "hi"}]},
        response_json={"content": [{"type": "text", "text": "hello"}]},
    )


@pytest.mark.asyncio
async def test_capture_writes_batch_to_sqlite(tmp_path: Path) -> None:
    writer = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    await writer.start()
    writer.enqueue(_record())
    await writer.flush()
    await writer.stop()

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    rows = conn.execute("SELECT id, model FROM traces").fetchall()
    assert rows == [("01AAAA", "claude-haiku-4-5")]


@pytest.mark.asyncio
async def test_capture_batches_multiple_records(tmp_path: Path) -> None:
    writer = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    await writer.start()
    for i in range(5):
        writer.enqueue(_record(id_=f"01AAA{i}"))
    await writer.flush()
    await writer.stop()

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
    assert count == 5


@pytest.mark.asyncio
async def test_capture_spills_to_jsonl_when_sqlite_unavailable(tmp_path: Path) -> None:
    bad_db_path = tmp_path / "not_a_dir" / "cap.sqlite3"
    tmp_path.joinpath("not_a_dir").write_text("i am a file, not a directory")  # sabotage mkdir

    writer = CaptureWriter(bad_db_path, tmp_path / "spill", flush_interval=100.0)
    writer.enqueue(_record())
    await writer.flush()

    spill_files = list((tmp_path / "spill").glob("spill-*.jsonl"))
    assert len(spill_files) == 1
    assert REGISTRY.get_counter("fail_open_events_total", {"component": "capture"}) == 1
    assert REGISTRY.get_counter("spill_rows_total") == 1


@pytest.mark.asyncio
async def test_capture_enqueue_never_raises_on_bad_input(tmp_path: Path) -> None:
    writer = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)

    class Unserializable:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    bad_record = _record()
    bad_record.request_json = {"x": Unserializable()}
    writer.enqueue(bad_record)  # must not raise


@pytest.mark.asyncio
async def test_capture_redacts_before_persisting(tmp_path: Path) -> None:
    writer = CaptureWriter(tmp_path / "cap.sqlite3", tmp_path / "spill", flush_interval=100.0)
    await writer.start()
    record = _record()
    record.request_json = {"messages": [{"role": "user", "content": "key AKIAABCDEFGHIJKLMNOP"}]}
    writer.enqueue(record)
    await writer.flush()
    await writer.stop()

    conn = sqlite3.connect(tmp_path / "cap.sqlite3")
    request_json = conn.execute("SELECT request_json FROM traces").fetchone()[0]
    assert "AKIAABCDEFGHIJKLMNOP" not in request_json


@pytest.mark.asyncio
async def test_capture_spills_when_sqlite_is_locked_by_another_writer(tmp_path: Path) -> None:
    """E5: SQLite locked by a concurrent connection. The live session must
    not hang indefinitely or crash — it spills within its (short, for this
    test) busy_timeout and the fail-open metric fires."""
    db_path = tmp_path / "cap.sqlite3"
    sqlite_db.connect(db_path).close()  # create schema up front

    blocker = sqlite3.connect(str(db_path), isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        writer = CaptureWriter(
            db_path, tmp_path / "spill", flush_interval=100.0, busy_timeout_ms=200
        )
        writer.enqueue(_record())
        await writer.flush()
    finally:
        blocker.execute("COMMIT")
        blocker.close()

    spill_files = list((tmp_path / "spill").glob("spill-*.jsonl"))
    assert len(spill_files) == 1
    assert REGISTRY.get_counter("fail_open_events_total", {"component": "capture"}) == 1

    # once the lock clears, the live session resumes normally
    writer2 = CaptureWriter(db_path, tmp_path / "spill", flush_interval=100.0)
    writer2.enqueue(_record(id_="01BBBB"))
    await writer2.flush()
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT count(*) FROM traces").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_capture_drops_batch_without_crashing_when_sqlite_and_spill_both_fail(
    tmp_path: Path,
) -> None:
    """E5: disk-full-on-spill-dir, compounded with SQLite also unavailable.
    Total loss of the batch is acceptable; a crash propagating out of
    flush() is not."""
    bad_db_dir = tmp_path / "not_a_dir"
    bad_db_dir.write_text("i am a file, not a directory")
    bad_spill_dir = tmp_path / "not_a_spill_dir_either"
    bad_spill_dir.write_text("also a file")

    writer = CaptureWriter(bad_db_dir / "cap.sqlite3", bad_spill_dir, flush_interval=100.0)
    writer.enqueue(_record())
    await writer.flush()  # must not raise

    assert REGISTRY.get_counter("spill_failed_total") == 1
