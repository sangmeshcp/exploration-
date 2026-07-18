from pathlib import Path

import httpx
import pytest

from shadowtrace.common.config import Settings
from shadowtrace.dashboard.api import create_app
from shadowtrace.db import duckdb_store


def _seeded_settings(tmp_path: Path) -> Settings:
    settings = Settings(home=tmp_path / "st")
    settings.ensure_dirs()
    conn = duckdb_store.connect(settings.duckdb_path)
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES ('a1', 'Code', True, ?, ?)",
        [now, now],
    )
    conn.execute(
        "INSERT INTO traces (id, ts, source, tool, model, request_json, response_json, tokens_in, "
        "tokens_out, cost_usd) VALUES ('t1', ?, 'transcript', 'claude_code', 'claude-sonnet-5', "
        '\'{"content": "hi"}\', \'{"content": "hello"}\', 10, 20, NULL)',
        [now],
    )
    conn.execute(
        "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
        "VALUES ('t1', 'a1', 0.1, ?)",
        [now],
    )
    conn.execute(
        "INSERT INTO recommendations (id, archetype_id, candidate, currency, value, pass_rate, "
        "pass_rate_ci_low, pass_rate_ci_high, latency_delta_ms, created_ts, expires_ts) "
        "VALUES ('r1', 'a1', 'claude-haiku-4-5', 'quota_headroom_pct', 12.5, 0.95, 0.9, 0.98, NULL, ?, ?)",
        [now, now + 1_000_000],
    )
    conn.close()
    return settings


@pytest.fixture
def client(tmp_path: Path) -> httpx.AsyncClient:
    settings = _seeded_settings(tmp_path)
    app = create_app(settings)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://dashboard")


@pytest.mark.asyncio
async def test_healthz(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


@pytest.mark.asyncio
async def test_readyz_reflects_duckdb_existence(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/readyz")
    assert resp.json()["ready"] is True


@pytest.mark.asyncio
async def test_summary(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/summary")
    body = resp.json()
    assert body["total_traces"] == 1
    assert body["archetype_count"] == 1
    assert body["traces_by_tool"] == {"claude_code": 1}


@pytest.mark.asyncio
async def test_archetypes_list(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/archetypes")
    body = resp.json()
    assert body == [{"id": "a1", "label": "Code", "pinned": True, "trace_count": 1}]


@pytest.mark.asyncio
async def test_traces_filtered_by_archetype(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/traces", params={"archetype_id": "a1"})
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_trace_detail_found_and_not_found(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/traces/t1")
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-sonnet-5"

        missing = await c.get("/api/traces/does-not-exist")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_recommendations_list(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/recommendations")
    body = resp.json()
    assert len(body) == 1
    assert body[0]["candidate"] == "claude-haiku-4-5"
    assert body[0]["archetype_label"] == "Code"


@pytest.mark.asyncio
async def test_applied_policies_empty(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/applied-policies")
    assert resp.json() == []


@pytest.mark.asyncio
async def test_system_endpoint(client: httpx.AsyncClient) -> None:
    async with client as c:
        resp = await c.get("/api/system")
    body = resp.json()
    assert "metrics" in body
    assert body["paused"] is False
