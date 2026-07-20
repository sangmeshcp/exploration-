"""PRIMARY capture lane: tail-parse Claude Code session transcripts under
`~/.claude/projects/**/*.jsonl` (plan.md M1.1, D1 consequence #1).

Read-only on Claude Code's own files — this module never locks or writes
into `claude_projects_dir`. Handles format drift defensively: a line that
fails to parse, or parses but doesn't look like a message we understand,
is skipped and counted, never raised. `PARSER_VERSION` is stamped onto
every row so a later transcript-schema change can be detected and
back-filled instead of silently misparsed forever.

Simplification vs. a literal reading of the schema note "full message list
incl. tool_use blocks": each row's `request_json` is the single preceding
user turn, not the full conversation history replayed at every row (that
would duplicate the whole growing transcript on every turn). Full
multi-turn context is reconstructed downstream by joining rows on
`session_id` ordered by `ts` / `parent_id` — that join is exactly what
chain-replay (M3.6) needs anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.timeparse import parse_ts_ms
from shadowtrace.common.tracing import get_default_tracer
from shadowtrace.common.ulid import new_ulid
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord

logger = get_logger("ingest.claude_transcripts")

PARSER_VERSION = "1"


def _extract_human_text(content: Any) -> str | None:
    """True human-authored text vs. a tool_result continuation. A plain
    string is always human text; a content-block list only counts if it
    has at least one `type: "text"` block — a list of purely `tool_result`
    blocks (the common case mid-agentic-chain) yields nothing here."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        return text or None
    return None


@dataclass
class TranscriptFile:
    """Incremental tail-parser for a single transcript file.

    Tracks a byte offset and a buffer for the last (possibly incomplete)
    line so a file that's actively being appended to by a live Claude Code
    session never has its in-progress final line parsed as garbage.
    """

    path: Path
    session_id: str | None = None
    _offset: int = 0
    _buffer: str = ""
    _last_user_content: Any = None
    _last_assistant_trace_id: str | None = None
    last_ts: int | None = None

    def __post_init__(self) -> None:
        if self.session_id is None:
            self.session_id = self.path.stem

    def poll(self) -> list[TraceRecord]:
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                new_bytes = f.read()
                self._offset = f.tell()
        except FileNotFoundError:
            return []

        text = self._buffer + new_bytes.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._buffer = lines[-1]  # last element has no trailing \n yet — may be incomplete
        complete_lines = lines[:-1]

        records: list[TraceRecord] = []
        for line in complete_lines:
            line = line.strip()
            if not line:
                continue
            record = self._parse_line(line)
            if record is not None:
                records.append(record)
        return records

    def _parse_line(self, line: str) -> TraceRecord | None:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            REGISTRY.inc("parse_errors_total")
            logger.warning("skipping malformed transcript line in %s", self.path)
            return None

        if not isinstance(entry, dict):
            REGISTRY.inc("unknown_schema_lines_total")
            return None

        entry_type = entry.get("type")
        message = entry.get("message")
        ts = parse_ts_ms(entry.get("timestamp"))
        self.last_ts = ts

        if entry_type == "user":
            if isinstance(message, dict):
                content = message.get("content")
                if _extract_human_text(content) is not None:
                    self._last_user_content = content
                # else: this "user" turn is a tool_result continuation, not
                # genuine human input (very common in agentic sessions —
                # most turns are tool calls, not free text). Deliberately
                # keep the previous _last_user_content in that case, so
                # every assistant turn in a tool-use chain still carries
                # the *original* human ask as its prompt, instead of going
                # prompt-less and getting silently dropped from mining
                # (mining/embed.py has nothing to embed without it).
            return None

        if entry_type != "assistant" or not isinstance(message, dict):
            REGISTRY.inc("unknown_schema_lines_total")
            return None

        uuid = entry.get("uuid")
        record = TraceRecord(
            id=new_ulid(ts_ms=ts),
            ts=ts,
            source="transcript",
            tool="claude_code",
            model=message.get("model", "unknown"),
            request_json={"role": "user", "content": self._last_user_content},
            response_json={"role": "assistant", "content": message.get("content")},
            tokens_in=(message.get("usage") or {}).get("input_tokens"),
            tokens_out=(message.get("usage") or {}).get("output_tokens"),
            cost_usd=None,  # subscription lane: flat-rate, no per-call dollar cost
            session_id=self.session_id,
            parent_id=self._last_assistant_trace_id,
            ingest_key=uuid,
            parser_version=PARSER_VERSION,
        )
        self._last_assistant_trace_id = record.id
        REGISTRY.inc("messages_ingested_total")
        return record


class TranscriptWatcher:
    """Discovers and incrementally polls every `*.jsonl` transcript under
    `projects_dir`. `scan_once()` is the deterministic, testable core;
    `run_forever()` is a thin polling loop wrapper for the `shadow up`
    daemon.
    """

    def __init__(
        self, projects_dir: Path, capture: CaptureWriter, tracer: trace.Tracer | None = None
    ) -> None:
        self.projects_dir = projects_dir
        self.capture = capture
        self.tracer = tracer or get_default_tracer()
        self._files: dict[Path, TranscriptFile] = {}

    def scan_once(self) -> int:
        """Poll all known + newly discovered transcript files once.

        Never raises: a single bad file is logged and skipped so the rest
        of the scan (and the caller's event loop) is unaffected.
        """
        if not self.projects_dir.exists():
            return 0

        ingested = 0
        for path in sorted(self.projects_dir.rglob("*.jsonl")):
            tf = self._files.setdefault(path, TranscriptFile(path))
            with self.tracer.start_as_current_span(
                "ingest_transcript_file", attributes={"path": str(path)}
            ) as span:
                try:
                    records = tf.poll()
                except Exception:
                    logger.exception("failed to poll transcript file %s", path)
                    span.set_status(Status(StatusCode.ERROR))
                    continue
                span.set_attribute("records_ingested", len(records))
                for record in records:
                    self.capture.enqueue(record)
                    ingested += 1

        REGISTRY.set_gauge("transcripts_watched", len(self._files))
        newest_ts = max(
            (tf.last_ts for tf in self._files.values() if tf.last_ts is not None), default=None
        )
        if newest_ts is not None:
            REGISTRY.set_gauge("capture_lag_seconds", max(0.0, time.time() - newest_ts / 1000))
        return ingested

    async def run_forever(self, poll_interval: float = 2.0) -> None:
        import asyncio

        while True:
            try:
                self.scan_once()
            except Exception:
                logger.exception("transcript watcher scan failed")
            await asyncio.sleep(poll_interval)
