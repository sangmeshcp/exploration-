"""Async writer → SQLite WAL, with JSONL spill fallback (plan.md M1.3/M1.4).

Fail-open contract: `enqueue()` must never raise or block the caller's
request path. Disk I/O is deferred to a background flush loop; if SQLite
is unavailable the batch is spilled to a JSONL file instead of being lost
or blocking, and a `fail_open_events_total` metric fires so the dashboard
can alert on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid
from shadowtrace.db import sqlite as sqlite_db
from shadowtrace.proxy.redact import redact_trace

logger = get_logger("capture")


@dataclass
class TraceRecord:
    id: str
    ts: int
    source: str
    model: str
    request_json: Any
    response_json: Any
    tool: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    session_id: str | None = None
    parent_id: str | None = None
    ingest_key: str | None = None
    parser_version: str | None = None
    redaction_flags: list[str] = field(default_factory=list)
    quarantined: bool = False


class CaptureWriter:
    def __init__(
        self,
        db_path: Path,
        spill_dir: Path,
        batch_size: int = 20,
        flush_interval: float = 1.0,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = db_path
        self.spill_dir = spill_dir
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.busy_timeout_ms = busy_timeout_ms
        self._pending: list[TraceRecord] = []
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._closed = True
        await self.flush()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def enqueue(self, record: TraceRecord) -> None:
        try:
            redaction = redact_trace(record.request_json, record.response_json)
            record.request_json = redaction.request_json
            record.response_json = redaction.response_json
            record.redaction_flags = redaction.flags
            record.quarantined = redaction.quarantined
            self._pending.append(record)
            REGISTRY.set_gauge("capture_queue_depth", len(self._pending))
        except Exception:
            logger.exception("capture.enqueue failed; dropping trace to preserve fail-open")

    async def flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        REGISTRY.set_gauge("capture_queue_depth", len(self._pending))
        try:
            await asyncio.to_thread(self._write_batch, batch)
        except Exception:
            logger.exception("sqlite write failed; spilling batch to JSONL")
            REGISTRY.inc("fail_open_events_total", labels={"component": "capture"})
            self._spill(batch)

    def _write_batch(self, batch: list[TraceRecord]) -> None:
        conn = sqlite_db.connect(self.db_path, busy_timeout_ms=self.busy_timeout_ms)
        try:
            with conn:
                for r in batch:
                    conn.execute(
                        """INSERT OR IGNORE INTO traces
                        (id, ts, source, tool, model, request_json, response_json,
                         tokens_in, tokens_out, latency_ms, cost_usd, session_id,
                         parent_id, redaction_flags, quarantined, ingest_key, parser_version)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            r.id,
                            r.ts,
                            r.source,
                            r.tool,
                            r.model,
                            json.dumps(r.request_json),
                            json.dumps(r.response_json),
                            r.tokens_in,
                            r.tokens_out,
                            r.latency_ms,
                            r.cost_usd,
                            r.session_id,
                            r.parent_id,
                            json.dumps(r.redaction_flags),
                            int(r.quarantined),
                            r.ingest_key,
                            r.parser_version,
                        ),
                    )
        finally:
            conn.close()

    def _spill(self, batch: list[TraceRecord]) -> None:
        """Last-resort persistence. If even this fails (e.g. disk full),
        the batch is dropped with a loud log rather than propagating —
        losing a batch of traces is acceptable; crashing the process (and
        with it `stop()`'s shutdown path, or the flush loop) is not.
        """
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            path = self.spill_dir / f"spill-{new_ulid()}.jsonl"
            with path.open("w") as f:
                for r in batch:
                    f.write(json.dumps(asdict(r), default=str) + "\n")
            REGISTRY.inc("spill_rows_total", value=len(batch))
            logger.warning("spilled %d rows to %s", len(batch), path)
        except Exception:
            REGISTRY.inc("spill_failed_total", value=len(batch))
            logger.exception(
                "spill also failed; dropping %d rows to preserve fail-open", len(batch)
            )

    async def _flush_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("flush loop iteration failed")
