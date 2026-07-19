"""Actuation: dry-run diff -> confirm -> write -> record (plan.md M4.3).

Every writer (claude_code.py, litellm.py, nanoclaw.py) implements the same
three-method shape so this module can drive all of them identically:

    preview(archetype_label, candidate) -> unified diff string
    write(archetype_label, candidate)   -> previous full file content
    revert(previous_content)            -> restores that exact content

`apply_recommendation` is a dry-run by default; only `dry_run=False`
actually calls `write()` and records the change in `applied_policies`
(with the previous content, so `revert_policy` can restore it
byte-identically — the E10 requirement).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid

logger = get_logger("recommend.apply")


class Writer(Protocol):
    name: str

    def preview(self, archetype_label: str, candidate: str) -> str: ...
    def write(self, archetype_label: str, candidate: str) -> str: ...
    def revert(self, previous_content: str) -> None: ...


@dataclass
class ApplyResult:
    id: str
    archetype_id: str
    candidate: str
    writer: str
    diff: str
    applied_ts: int


def apply_recommendation(
    conn: Any,
    writer: Writer,
    archetype_id: str,
    archetype_label: str,
    candidate: str,
    dry_run: bool = True,
) -> str | ApplyResult:
    """Returns the preview diff when `dry_run` (the default); only writes
    and records the change when explicitly told not to dry-run."""
    diff = writer.preview(archetype_label, candidate)
    if dry_run:
        return diff

    previous_content = writer.write(archetype_label, candidate)
    now = int(time.time() * 1000)
    applied_id = new_ulid()
    conn.execute(
        "INSERT INTO applied_policies "
        "(id, archetype_id, candidate, writer, diff, previous_content, applied_ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [applied_id, archetype_id, candidate, writer.name, diff, previous_content, now],
    )
    REGISTRY.inc("active_policies")
    logger.info(
        "applied policy %s: %s -> %s via %s", applied_id, archetype_label, candidate, writer.name
    )
    return ApplyResult(
        id=applied_id,
        archetype_id=archetype_id,
        candidate=candidate,
        writer=writer.name,
        diff=diff,
        applied_ts=now,
    )


def revert_policy(
    conn: Any, writers: dict[str, Writer], applied_policy_id: str, reason: str
) -> None:
    row = conn.execute(
        "SELECT writer, previous_content FROM applied_policies WHERE id = ? AND reverted_ts IS NULL",
        [applied_policy_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"no active applied_policy with id={applied_policy_id!r}")
    writer_name, previous_content = row
    writer = writers.get(writer_name)
    if writer is None:
        raise ValueError(f"no writer registered for {writer_name!r}")

    writer.revert(previous_content)
    now = int(time.time() * 1000)
    conn.execute(
        "UPDATE applied_policies SET reverted_ts = ?, revert_reason = ? WHERE id = ?",
        [now, reason, applied_policy_id],
    )
    REGISTRY.inc("active_policies", value=-1)
    logger.warning("reverted policy %s: %s", applied_policy_id, reason)
