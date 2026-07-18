"""claude.ai `conversations.json` export importer (plan.md M1.7, review R8).

Idempotent by construction: each imported message is enqueued with
`ingest_key=<message id>` and `source="claude_export"`, and the SQLite
schema's partial unique index on `(source, ingest_key)` makes re-importing
the same file (or an extended file containing already-imported messages)
a no-op for those rows — the capture writer's `INSERT OR IGNORE` just skips
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shadowtrace.common.logging import get_logger
from shadowtrace.common.timeparse import parse_ts_ms
from shadowtrace.common.ulid import new_ulid
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord

logger = get_logger("ingest.claude_export")


def _extract_text(message: dict[str, Any]) -> Any:
    if "text" in message:
        return message["text"]
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(p for p in parts if p)
    return content


def import_export_file(path: Path, capture: CaptureWriter) -> int:
    """Parse and enqueue one claude.ai `conversations.json` export. Returns
    the number of assistant turns enqueued (duplicates included — actual
    dedup happens at the capture/DB layer)."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("failed to read claude.ai export %s", path)
        return 0

    conversations = data if isinstance(data, list) else data.get("conversations", [])
    count = 0
    for convo in conversations:
        if not isinstance(convo, dict):
            continue
        convo_id = convo.get("uuid") or convo.get("id") or new_ulid()
        messages = convo.get("chat_messages") or convo.get("messages") or []

        prior_human_content: Any = None
        prior_trace_id: str | None = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            sender = msg.get("sender") or msg.get("role")
            text = _extract_text(msg)
            if sender in ("human", "user"):
                prior_human_content = text
                continue
            if sender not in ("assistant", "bot"):
                continue

            msg_id = msg.get("uuid") or msg.get("id")
            ts = parse_ts_ms(msg.get("created_at") or msg.get("timestamp"))
            record = TraceRecord(
                id=new_ulid(ts_ms=ts),
                ts=ts,
                source="claude_export",
                tool="chat",
                model=msg.get("model", "unknown"),
                request_json={"role": "user", "content": prior_human_content},
                response_json={"role": "assistant", "content": text},
                session_id=str(convo_id),
                parent_id=prior_trace_id,
                ingest_key=str(msg_id) if msg_id is not None else None,
            )
            capture.enqueue(record)
            prior_trace_id = record.id
            count += 1

    return count
