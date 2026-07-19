"""Streaming tee + fail-open forwarding (plan.md M1.2/M1.3).

The core invariant under test: the client must receive an upstream response
that is byte-identical to a direct call, regardless of what happens on the
capture side. Capture only ever *observes* the stream after it has already
been forwarded — it can never slow it down, alter it, or block on it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY

logger = get_logger("proxy.passthrough")

OnComplete = Callable[[bytes, int], None]


async def tee_stream(
    upstream_iter: AsyncIterator[bytes],
    status_code: int,
    on_complete: OnComplete,
) -> AsyncIterator[bytes]:
    """Forward chunks to the client immediately; assemble a copy for capture.

    `on_complete` runs only after the full stream has already been handed to
    the client, and any exception it raises is caught here — capture must
    never affect what the client already received.
    """
    buffer = bytearray()
    try:
        async for chunk in upstream_iter:
            buffer.extend(chunk)
            yield chunk
    finally:
        try:
            on_complete(bytes(buffer), status_code)
        except Exception:
            logger.exception("capture callback failed after stream completed")
            REGISTRY.inc("fail_open_events_total", labels={"component": "proxy"})


async def forward(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes,
) -> httpx.Response:
    """Issue the upstream request and return a streamed httpx.Response.

    Upstream errors (4xx/5xx) are returned as-is for the caller to relay
    verbatim; network-level failures (timeout, connection refused) raise
    httpx.HTTPError and must be relayed as a proxy error, never silently
    retried or swallowed — that would break byte-identical passthrough.
    """
    request = client.build_request(method, url, headers=headers, content=content)
    return await client.send(request, stream=True)
