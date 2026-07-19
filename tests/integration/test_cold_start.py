"""E12: fresh-install cold start. Empty state -> the dashboard API renders
empty-states, no crashes on an empty/nonexistent DB (plan.md §3.3).
"""

from pathlib import Path

import httpx
import pytest

from shadowtrace.common.config import Settings
from shadowtrace.dashboard.api import create_app


@pytest.fixture
def fresh_settings(tmp_path: Path) -> Settings:
    # deliberately no ensure_dirs(), no seeding — nothing has ever run here
    return Settings(home=tmp_path / "brand-new-install")


@pytest.mark.asyncio
async def test_healthz_and_readyz_before_anything_exists(fresh_settings: Settings) -> None:
    app = create_app(fresh_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dashboard"
    ) as client:
        health = await client.get("/healthz")
        assert health.status_code == 200

        ready = await client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["ready"] is False  # duckdb file doesn't exist yet


@pytest.mark.asyncio
async def test_api_endpoints_return_empty_states_not_errors(fresh_settings: Settings) -> None:
    app = create_app(fresh_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dashboard"
    ) as client:
        summary = await client.get("/api/summary")
        assert summary.status_code == 200
        assert summary.json() == {
            "total_traces": 0,
            "quarantined_traces": 0,
            "archetype_count": 0,
            "unassigned_traces": 0,
            "traces_by_tool": {},
            "traces_by_model": {},
        }

        for path in (
            "/api/archetypes",
            "/api/traces",
            "/api/recommendations",
            "/api/applied-policies",
        ):
            resp = await client.get(path)
            assert resp.status_code == 200, path
            assert resp.json() == [], path

        trace_detail = await client.get("/api/traces/does-not-exist")
        assert trace_detail.status_code == 404

        system = await client.get("/api/system")
        assert system.status_code == 200
        assert system.json()["paused"] is False


@pytest.mark.asyncio
async def test_first_api_call_implicitly_provisions_the_analytics_store(
    fresh_settings: Settings,
) -> None:
    assert not fresh_settings.duckdb_path.exists()
    app = create_app(fresh_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://dashboard"
    ) as client:
        await client.get("/api/summary")
    assert fresh_settings.duckdb_path.exists()
