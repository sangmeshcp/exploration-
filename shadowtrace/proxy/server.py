"""FastAPI passthrough gateway: /v1/messages (Anthropic) and
/v1/chat/completions (OpenAI-compat). Secondary capture lane for the
API-key path (plan.md M1.2) — the primary lane is transcript ingestion
(ingest/claude_transcripts.py).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from shadowtrace.common.config import Settings, get_settings
from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid
from shadowtrace.proxy.capture import CaptureWriter, TraceRecord
from shadowtrace.proxy.passthrough import forward, tee_stream

logger = get_logger("proxy.server")

# Best-effort estimate for the API-key lane only; subscription-lane traces
# always have cost_usd=NULL (quota is flat-rate, see plan.md D1.3). This is
# intentionally a small local table, separate from replay/ladder.py's more
# complete candidate price table — proxy capture and replay pricing are
# different concerns that happen to overlap on a few model names.
_PRICE_PER_MTOK_USD: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _filtered_headers(headers: httpx.Headers | Any) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


def estimate_cost_usd(model: str, tokens_in: int | None, tokens_out: int | None) -> float | None:
    prices = _PRICE_PER_MTOK_USD.get(model)
    if prices is None or tokens_in is None or tokens_out is None:
        return None
    in_price, out_price = prices
    return (tokens_in / 1_000_000) * in_price + (tokens_out / 1_000_000) * out_price


def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    text = raw.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            data_text = "\n".join(data_lines)
            try:
                data: Any = json.loads(data_text)
            except json.JSONDecodeError:
                data = data_text
            events.append({"event": event_name, "data": data})
        event_name = None
        data_lines = []

    for line in text.split("\n"):
        line = line.rstrip("\r")
        if line == "":
            _flush()
        elif line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    _flush()
    return events


def _find_usage(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "usage" in value and isinstance(value["usage"], dict):
            return value["usage"]
        for v in value.values():
            found = _find_usage(v)
            if found is not None:
                return found
    elif isinstance(value, list):
        for v in value:
            found = _find_usage(v)
            if found is not None:
                return found
    return None


def _extract_tokens(response_body: Any) -> tuple[int | None, int | None]:
    usage = _find_usage(response_body)
    if usage is None:
        return None, None
    tokens_in = usage.get("input_tokens", usage.get("prompt_tokens"))
    tokens_out = usage.get("output_tokens", usage.get("completion_tokens"))
    return tokens_in, tokens_out


def parse_response_body(raw: bytes, content_type: str) -> Any:
    if "text/event-stream" in content_type:
        return {"events": _parse_sse_events(raw)}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw.decode("utf-8", errors="replace")}


@dataclass
class ProxyConfig:
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com"
    settings: Settings | None = None
    request_timeout_s: float = 60.0
    transport: httpx.AsyncBaseTransport | None = None  # test hook: route via ASGITransport


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    config = config or ProxyConfig()
    settings = config.settings or get_settings()
    settings.ensure_dirs()

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.request_timeout_s), transport=config.transport
    )
    capture = CaptureWriter(settings.sqlite_path, settings.spill_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await capture.start()
        try:
            yield
        finally:
            await capture.stop()
            await client.aclose()

    app = FastAPI(title="shadowtrace proxy", lifespan=lifespan)
    app.state.client = client
    app.state.capture = capture

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict[str, bool]:
        ready = settings.sqlite_path.parent.exists()
        return {"ready": ready}

    async def _proxy(request: Request, base_url: str, tool_hint: str) -> Response:
        REGISTRY.inc("requests_total", labels={"status": "started"})
        body = await request.body()
        target_url = base_url.rstrip("/") + request.url.path
        if request.url.query:
            target_url += "?" + request.url.query
        headers = _filtered_headers(request.headers)

        start = time.perf_counter()
        try:
            upstream_resp = await forward(client, request.method, target_url, headers, body)
        except httpx.HTTPError as exc:
            logger.exception("upstream request failed")
            REGISTRY.inc("requests_total", labels={"status": "upstream_error"})
            return Response(content=str(exc), status_code=502)
        proxy_overhead_ms = (time.perf_counter() - start) * 1000
        REGISTRY.observe("proxy_added_latency_ms", proxy_overhead_ms)

        response_headers = _filtered_headers(upstream_resp.headers)
        content_type = upstream_resp.headers.get("content-type", "")
        trace_id = new_ulid()
        received_ts = int(time.time() * 1000)

        def on_complete(raw_body: bytes, status_code: int) -> None:
            REGISTRY.inc("requests_total", labels={"status": str(status_code)})
            if status_code >= 400:
                # Never capture an error as if it were a successful trace.
                return
            if settings.is_paused():
                # `shadow pause` bypass flag: forward normally, capture nothing.
                return
            try:
                request_body_json = json.loads(body) if body else {}
            except json.JSONDecodeError:
                request_body_json = {"raw_text": body.decode("utf-8", errors="replace")}
            response_body_json = parse_response_body(raw_body, content_type)
            tokens_in, tokens_out = _extract_tokens(response_body_json)
            model = (
                request_body_json.get("model", "unknown")
                if isinstance(request_body_json, dict)
                else "unknown"
            )
            record = TraceRecord(
                id=trace_id,
                ts=received_ts,
                source="proxy",
                tool=tool_hint,
                model=model,
                request_json=request_body_json,
                response_json=response_body_json,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=int((time.time() * 1000) - received_ts),
                cost_usd=estimate_cost_usd(model, tokens_in, tokens_out),
                session_id=request.headers.get("x-shadowtrace-session"),
            )
            capture.enqueue(record)

        return StreamingResponse(
            tee_stream(upstream_resp.aiter_bytes(), upstream_resp.status_code, on_complete),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            media_type=content_type or None,
        )

    @app.api_route("/v1/messages", methods=["POST"])
    async def anthropic_messages(request: Request) -> Response:
        return await _proxy(request, config.anthropic_base_url, "script")

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def openai_chat_completions(request: Request) -> Response:
        return await _proxy(request, config.openai_base_url, "script")

    return app
