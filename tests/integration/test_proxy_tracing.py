"""Proxy request tracing: correlation id stamped into both the span and
the captured row (plan.md §4.5 log/trace/data joinability)."""

import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from shadowtrace.common.config import Settings
from shadowtrace.common.tracing import build_tracer_provider
from shadowtrace.proxy.server import ProxyConfig, create_app
from tests.fakes.anthropic_server import app as fake_upstream_app
from tests.fakes.anthropic_server import reset_call_log


def _read_spans(traces_path: Path) -> list[dict[str, object]]:
    if not traces_path.exists():
        return []
    return [json.loads(line) for line in traces_path.read_text().strip().split("\n") if line]


@pytest.mark.asyncio
async def test_proxy_request_span_correlation_id_matches_captured_row(tmp_path: Path) -> None:
    reset_call_log()
    settings = Settings(home=tmp_path / "shadowtrace")
    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic",
        settings=settings,
        transport=httpx.ASGITransport(app=fake_upstream_app),
        tracer=tracer,
    )
    proxy_app = create_app(config)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        await proxy_app.state.capture.flush()

    spans = _read_spans(traces_path)
    proxy_spans = [s for s in spans if s["name"] == "proxy_request"]
    assert len(proxy_spans) == 1
    correlation_id = proxy_spans[0]["attributes"]["correlation_id"]
    assert proxy_spans[0]["attributes"]["upstream_status_code"] == 200

    conn = sqlite3.connect(settings.sqlite_path)
    row = conn.execute("SELECT id FROM traces").fetchone()
    assert row[0] == correlation_id
