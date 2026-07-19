"""Minimal dependency-free ULID generator.

26-char Crockford base32 encoding of a 48-bit millisecond timestamp
followed by 80 bits of randomness. Lexicographically sortable by creation
time, which the capture store relies on for ordering without a separate
autoincrement column.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    chars = ["0"] * length
    for i in range(length - 1, -1, -1):
        chars[i] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(chars)


def new_ulid(*, ts_ms: int | None = None) -> str:
    """Generate a new ULID string. `ts_ms` is injectable for deterministic tests."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    return _encode(ts_ms, 10) + _encode(randomness, 16)


def ulid_timestamp_ms(ulid: str) -> int:
    """Recover the millisecond timestamp encoded in a ULID's first 10 chars."""
    value = 0
    for ch in ulid[:10]:
        value = (value << 5) | _CROCKFORD.index(ch)
    return value
