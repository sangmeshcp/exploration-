"""Dashboard backend: FastAPI over DuckDB (plan.md M2.4/M4.6, D2).

Serves `/api/*` for the React app (dashboard/web/) plus `/healthz` /
`/readyz` for the System tab and local alerting (§4.5). Read-only over
the analytics store — all writes happen through the CLI / mining /
replay / recommend modules, never through this API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from shadowtrace.common.config import Settings, get_settings
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.db import duckdb_store


def _connect(settings: Settings) -> Any:
    return duckdb_store.connect(settings.duckdb_path)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="shadowtrace dashboard")
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        ready = settings.duckdb_path.exists()
        return {"ready": ready, "duckdb_path": str(settings.duckdb_path)}

    @app.get("/api/summary")
    async def summary() -> dict[str, Any]:
        conn = _connect(settings)
        total_traces = conn.execute("SELECT count(*) FROM traces").fetchone()[0]
        quarantined = conn.execute(
            "SELECT count(*) FROM traces WHERE quarantined = TRUE"
        ).fetchone()[0]
        archetype_count = conn.execute(
            "SELECT count(*) FROM archetypes WHERE merged_into IS NULL"
        ).fetchone()[0]
        assigned = conn.execute("SELECT count(*) FROM archetype_assignments").fetchone()[0]
        by_tool = dict(
            conn.execute(
                "SELECT coalesce(tool, 'unknown'), count(*) FROM traces GROUP BY 1"
            ).fetchall()
        )
        by_model = dict(
            conn.execute(
                "SELECT model, count(*) FROM traces GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
            ).fetchall()
        )
        return {
            "total_traces": total_traces,
            "quarantined_traces": quarantined,
            "archetype_count": archetype_count,
            "unassigned_traces": max(0, total_traces - assigned),
            "traces_by_tool": by_tool,
            "traces_by_model": by_model,
        }

    @app.get("/api/archetypes")
    async def archetypes() -> list[dict[str, Any]]:
        conn = _connect(settings)
        rows = conn.execute(
            "SELECT a.id, a.label, a.pinned, count(aa.trace_id) AS n "
            "FROM archetypes a LEFT JOIN archetype_assignments aa ON aa.archetype_id = a.id "
            "WHERE a.merged_into IS NULL "
            "GROUP BY a.id, a.label, a.pinned ORDER BY n DESC"
        ).fetchall()
        return [{"id": r[0], "label": r[1], "pinned": r[2], "trace_count": r[3]} for r in rows]

    @app.get("/api/traces")
    async def traces(
        archetype_id: str | None = None, limit: int = Query(default=50, le=500)
    ) -> list[dict[str, Any]]:
        conn = _connect(settings)
        if archetype_id:
            rows = conn.execute(
                "SELECT t.id, t.ts, t.source, t.tool, t.model, t.tokens_in, t.tokens_out, t.cost_usd "
                "FROM traces t JOIN archetype_assignments aa ON aa.trace_id = t.id "
                "WHERE aa.archetype_id = ? ORDER BY t.ts DESC LIMIT ?",
                [archetype_id, limit],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ts, source, tool, model, tokens_in, tokens_out, cost_usd "
                "FROM traces ORDER BY ts DESC LIMIT ?",
                [limit],
            ).fetchall()
        cols = ["id", "ts", "source", "tool", "model", "tokens_in", "tokens_out", "cost_usd"]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    @app.get("/api/traces/{trace_id}")
    async def trace_detail(trace_id: str) -> dict[str, Any]:
        conn = _connect(settings)
        row = conn.execute(
            "SELECT id, ts, source, tool, model, request_json, response_json, tokens_in, "
            "tokens_out, cost_usd, redaction_flags, quarantined FROM traces WHERE id = ?",
            [trace_id],
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="trace not found")
        cols = [
            "id",
            "ts",
            "source",
            "tool",
            "model",
            "request_json",
            "response_json",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "redaction_flags",
            "quarantined",
        ]
        return dict(zip(cols, row, strict=True))

    @app.get("/api/recommendations")
    async def recommendations() -> list[dict[str, Any]]:
        conn = _connect(settings)
        rows = conn.execute(
            "SELECT r.id, r.archetype_id, a.label, r.candidate, r.currency, r.value, r.pass_rate, "
            "r.pass_rate_ci_low, r.pass_rate_ci_high, r.expires_ts "
            "FROM recommendations r JOIN archetypes a ON a.id = r.archetype_id "
            "ORDER BY r.created_ts DESC"
        ).fetchall()
        cols = [
            "id",
            "archetype_id",
            "archetype_label",
            "candidate",
            "currency",
            "value",
            "pass_rate",
            "pass_rate_ci_low",
            "pass_rate_ci_high",
            "expires_ts",
        ]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    @app.get("/api/applied-policies")
    async def applied_policies() -> list[dict[str, Any]]:
        conn = _connect(settings)
        rows = conn.execute(
            "SELECT id, archetype_id, candidate, writer, applied_ts, reverted_ts, revert_reason "
            "FROM applied_policies ORDER BY applied_ts DESC"
        ).fetchall()
        cols = [
            "id",
            "archetype_id",
            "candidate",
            "writer",
            "applied_ts",
            "reverted_ts",
            "revert_reason",
        ]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    @app.get("/api/system")
    async def system() -> dict[str, Any]:
        return {
            "metrics": REGISTRY.snapshot(),
            "duckdb_path": str(settings.duckdb_path),
            "sqlite_path": str(settings.sqlite_path),
            "paused": settings.is_paused(),
        }

    web_dist = Path(__file__).parent / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return app
