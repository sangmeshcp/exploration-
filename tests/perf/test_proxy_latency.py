"""Proxy latency budget (plan.md §3.4): p50 < 5ms, p99 < 15ms of proxy-
*added* overhead (time spent in `forward()` — request build + upstream
send/receive headers — around the fake upstream), not full request
latency including client-side test-harness overhead.

Measured through `ProxyConfig.transport=httpx.ASGITransport`, i.e.
in-process against the fake upstream — this isolates the proxy's own
processing cost from real network variance, so it's a regression gate on
`proxy_added_latency_ms` (catches a 2x-plus slowdown in the proxy's own
code), not a literal reproduction of production network conditions.
"""

from pathlib import Path

import httpx
import pytest

from shadowtrace.common.config import Settings
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.proxy.server import ProxyConfig, create_app
from tests.fakes.anthropic_server import app as fake_upstream_app
from tests.fakes.anthropic_server import reset_call_log

N_REQUESTS = 200
P50_BUDGET_MS = 5.0
P99_BUDGET_MS = 15.0


@pytest.mark.perf
@pytest.mark.asyncio
async def test_proxy_added_latency_within_budget(tmp_path: Path) -> None:
    REGISTRY.reset()
    reset_call_log()
    settings = Settings(home=tmp_path / "shadowtrace")
    config = ProxyConfig(
        anthropic_base_url="http://fake-anthropic",
        settings=settings,
        transport=httpx.ASGITransport(app=fake_upstream_app),
    )
    proxy_app = create_app(config)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy"
    ) as client:
        for _ in range(N_REQUESTS):
            resp = await client.post(
                "/v1/messages",
                json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200

    p50 = REGISTRY.get_percentile("proxy_added_latency_ms", 0.5)
    p99 = REGISTRY.get_percentile("proxy_added_latency_ms", 0.99)

    assert p50 < P50_BUDGET_MS, f"p50 proxy overhead {p50:.2f}ms exceeds {P50_BUDGET_MS}ms budget"
    assert p99 < P99_BUDGET_MS, f"p99 proxy overhead {p99:.2f}ms exceeds {P99_BUDGET_MS}ms budget"
