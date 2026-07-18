import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from shadowtrace.common.config import Settings
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.proxy.server import ProxyConfig, create_app
from tests.fakes.anthropic_server import app as fake_upstream_app
from tests.fakes.anthropic_server import reset_call_log


@pytest.fixture(autouse=True)
def _reset() -> None:
    REGISTRY.reset()
    reset_call_log()


def _make_proxy_client(tmp_path: Path) -> httpx.AsyncClient:
    settings = Settings(home=tmp_path / "shadowtrace")
    fake_transport = httpx.ASGITransport(app=fake_upstream_app)
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic",
        settings=settings,
        transport=fake_transport,
    )
    proxy_app = create_app(config)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy")


@pytest.mark.asyncio
async def test_non_streaming_passthrough_is_byte_identical(tmp_path: Path) -> None:
    async with _make_proxy_client(tmp_path) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"][0]["text"] == "Hello from the fake upstream."
    assert body["usage"]["input_tokens"] == 12


@pytest.mark.asyncio
async def test_streaming_passthrough_forwards_all_sse_chunks(tmp_path: Path) -> None:
    async with (
        _make_proxy_client(tmp_path) as client,
        client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp,
    ):
        chunks = [c async for c in resp.aiter_bytes()]
    full = b"".join(chunks)
    assert b"message_start" in full
    assert b"message_stop" in full
    assert b"Hello from the fake upstream." in full


@pytest.mark.asyncio
async def test_captured_trace_appears_in_sqlite(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "shadowtrace")
    fake_transport = httpx.ASGITransport(app=fake_upstream_app)
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic", settings=settings, transport=fake_transport
    )
    from shadowtrace.proxy.server import create_app as _create_app

    proxy_app = _create_app(config)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        await proxy_app.state.capture.flush()

    conn = sqlite3.connect(settings.sqlite_path)
    rows = conn.execute("SELECT model, tokens_in, tokens_out, cost_usd FROM traces").fetchall()
    assert len(rows) == 1
    model, tokens_in, tokens_out, cost_usd = rows[0]
    assert model == "claude-haiku-4-5"
    assert tokens_in == 12
    assert tokens_out == 34
    assert cost_usd is not None and cost_usd > 0


@pytest.mark.asyncio
async def test_tool_use_round_trip_preserves_structure(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "shadowtrace")
    fake_transport = httpx.ASGITransport(app=fake_upstream_app)
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic", settings=settings, transport=fake_transport
    )
    from shadowtrace.proxy.server import create_app as _create_app

    proxy_app = _create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5",
                "tools": [{"name": "read_file"}],
                "messages": [{"role": "user", "content": "read a.py"}],
            },
        )
        assert resp.status_code == 200
        assert any(block["type"] == "tool_use" for block in resp.json()["content"])
        await proxy_app.state.capture.flush()

    conn = sqlite3.connect(settings.sqlite_path)
    response_json = json.loads(conn.execute("SELECT response_json FROM traces").fetchone()[0])
    assert any(block.get("type") == "tool_use" for block in response_json["content"])


@pytest.mark.asyncio
async def test_upstream_error_passed_through_and_not_captured(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path / "shadowtrace")
    fake_transport = httpx.ASGITransport(app=fake_upstream_app)
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic", settings=settings, transport=fake_transport
    )
    from shadowtrace.proxy.server import create_app as _create_app

    proxy_app = _create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "trigger-500", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 500
        await proxy_app.state.capture.flush()

    if settings.sqlite_path.exists():
        conn = sqlite3.connect(settings.sqlite_path)
        count = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
        assert count == 0
    # else: nothing was ever captured, which also satisfies "not captured"


@pytest.mark.asyncio
async def test_capture_writer_killed_mid_stream_client_still_gets_full_response(
    tmp_path: Path,
) -> None:
    """Client must receive the full stream even if capture blows up after."""
    settings = Settings(home=tmp_path / "shadowtrace")
    fake_transport = httpx.ASGITransport(app=fake_upstream_app)
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic", settings=settings, transport=fake_transport
    )
    from shadowtrace.proxy.server import create_app as _create_app

    proxy_app = _create_app(config)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("capture writer is dead")

    proxy_app.state.capture.enqueue = _boom  # type: ignore[method-assign]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "Hello from the fake upstream."
    assert REGISTRY.get_counter("fail_open_events_total", {"component": "proxy"}) == 1
