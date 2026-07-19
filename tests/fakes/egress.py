"""Socket-blocking test fixture for egress containment (plan.md §3.3, E13).

Patches `socket.socket.connect` at the lowest level so it catches
`httpx`, `sqlite3`-over-network (n/a here, but any future network client)
and asyncio's default selector event loop alike, all of which eventually
call down to a real `socket.socket().connect(...)`.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator


class EgressBlocked(PermissionError):
    pass


@contextlib.contextmanager
def block_egress(allowed_hosts: set[str]) -> Iterator[None]:
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in allowed_hosts:
            raise EgressBlocked(f"egress blocked: attempted connection to {host!r}")
        original_connect(self, address)  # type: ignore[arg-type]

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
