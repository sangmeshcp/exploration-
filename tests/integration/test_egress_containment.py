"""E13: a socket-blocking fixture allows only the fake-upstream/candidate
hosts; any other outbound attempt fails (plan.md §3.3, §4 NFR).
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator

import httpx
import pytest

from tests.fakes.egress import EgressBlocked, block_egress

# RFC 5737 TEST-NET-3: reserved for documentation, guaranteed non-routable —
# safe to "attempt" a connection to without any risk of it ever succeeding
# or leaking real traffic, even if the blocking mechanism itself were broken.
DISALLOWED_HOST = "203.0.113.1"


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


@pytest.fixture
def local_http_server() -> Iterator[int]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_socket_connect_to_allowlisted_host_succeeds(local_http_server: int) -> None:
    with block_egress({"127.0.0.1"}):
        sock = socket.create_connection(("127.0.0.1", local_http_server), timeout=2.0)
        sock.close()


def test_socket_connect_to_disallowed_host_is_blocked() -> None:
    with block_egress({"127.0.0.1"}), pytest.raises(EgressBlocked, match="egress blocked"):
        socket.create_connection((DISALLOWED_HOST, 80), timeout=2.0)


def test_httpx_request_to_allowlisted_host_succeeds_under_egress_block(
    local_http_server: int,
) -> None:
    with block_egress({"127.0.0.1"}):
        resp = httpx.get(f"http://127.0.0.1:{local_http_server}/", timeout=5.0)
    assert resp.status_code == 200


def test_httpx_request_to_disallowed_host_fails_under_egress_block() -> None:
    with block_egress({"127.0.0.1"}), pytest.raises(httpx.ConnectError):
        httpx.get(f"http://{DISALLOWED_HOST}/", timeout=2.0)


def test_egress_block_restores_original_connect_after_context_exit(
    local_http_server: int,
) -> None:
    original = socket.socket.connect
    with block_egress({"127.0.0.1"}):
        assert socket.socket.connect is not original
    assert socket.socket.connect is original
