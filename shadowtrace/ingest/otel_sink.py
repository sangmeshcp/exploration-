"""Minimal OTLP/HTTP-JSON metrics receiver for Claude Code's built-in
telemetry (plan.md M1.6).

This is intentionally a simplified subset of the OTLP metrics envelope —
just enough structure (resourceMetrics -> scopeMetrics -> metrics ->
dataPoints) to pull out token-count data points and cross-check them
against transcript-derived counts (§4.5 SLO: capture completeness >= 99%).
It is not a general-purpose OTLP collector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY

logger = get_logger("ingest.otel_sink")


@dataclass
class TokenMetricPoint:
    metric_name: str
    value: float
    model: str | None
    session_id: str | None
    ts: int


def parse_otlp_metrics_json(payload: dict[str, Any]) -> list[TokenMetricPoint]:
    points: list[TokenMetricPoint] = []
    for resource_metric in payload.get("resourceMetrics", []):
        for scope_metric in resource_metric.get("scopeMetrics", []):
            for metric in scope_metric.get("metrics", []):
                name = metric.get("name", "")
                if "token" not in name:
                    continue
                data = metric.get("sum") or metric.get("gauge") or {}
                for dp in data.get("dataPoints", []):
                    attrs = {
                        a["key"]: a.get("value", {}).get("stringValue")
                        for a in dp.get("attributes", [])
                    }
                    value = dp.get("asDouble", dp.get("asInt"))
                    if value is None:
                        continue
                    ts_nanos = dp.get("timeUnixNano")
                    ts_ms = int(int(ts_nanos) / 1_000_000) if ts_nanos else 0
                    points.append(
                        TokenMetricPoint(
                            metric_name=name,
                            value=float(value),
                            model=attrs.get("model"),
                            session_id=attrs.get("session.id"),
                            ts=ts_ms,
                        )
                    )
    return points


class OtelSink:
    """In-memory ledger of received token-metric points, for the
    transcript-vs-OTel daily cross-check."""

    def __init__(self) -> None:
        self.points: list[TokenMetricPoint] = []

    def ingest(self, payload: dict[str, Any]) -> int:
        try:
            new_points = parse_otlp_metrics_json(payload)
        except Exception:
            logger.exception("failed to parse OTLP metrics payload")
            return 0
        self.points.extend(new_points)
        return len(new_points)

    def total_tokens(self) -> float:
        return sum(p.value for p in self.points if "token" in p.metric_name)


def cross_check_completeness(transcript_tokens: float, otel_tokens: float) -> float:
    """Returns capture completeness as a ratio in [0, 1+]. 1.0 means the
    transcript-derived token count matches OTel's reported count exactly."""
    if otel_tokens <= 0:
        return 1.0 if transcript_tokens == 0 else 0.0
    return min(transcript_tokens / otel_tokens, 1.0)


def create_app(sink: OtelSink | None = None) -> FastAPI:
    sink = sink or OtelSink()
    app = FastAPI(title="shadowtrace otel-sink")
    app.state.sink = sink

    @app.post("/v1/metrics")
    async def receive_metrics(request: Request) -> dict[str, int]:
        payload = await request.json()
        count = sink.ingest(payload)
        REGISTRY.inc("otel_metric_points_total", value=count)
        return {"accepted": count}

    return app
