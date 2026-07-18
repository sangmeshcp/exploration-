import sqlite3
from pathlib import Path

import pytest

from shadowtrace.common.metrics import REGISTRY
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
