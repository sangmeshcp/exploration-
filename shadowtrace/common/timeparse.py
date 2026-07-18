"""Shared timestamp parsing for ingest sources with inconsistent formats."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any


def parse_ts_ms(value: Any) -> int:
    """Best-effort ISO-8601 -> epoch-ms. Falls back to "now" for anything
    unparseable so a malformed timestamp field never crashes an ingester."""
    if isinstance(value, str):
        try:
            text = value.replace("Z", "+00:00")
            return int(datetime.fromisoformat(text).timestamp() * 1000)
        except ValueError:
            pass
    return int(time.time() * 1000)
