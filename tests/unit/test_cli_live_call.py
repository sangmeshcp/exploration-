"""Verifies cli.py's `_live_call` request/response wiring against mocked
HTTP endpoints (respx) — the actual network call is real code, but no
real network is touched in CI."""

import httpx
import pytest
import respx

from shadowtrace.cli import _live_call
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import RateLimitError
from shadowtrace.replay.sampler import SampleRow

OLLAMA_CANDIDATE = Candidate(
    name="ollama-llama3.1-8b",
    provider="ollama",
    price_per_mtok_in=0,
    price_per_mtok_out=0,
    local=True,
)
ANTHROPIC_CANDIDATE = Candidate(
    name="claude-haiku-4-5", provider="anthropic", price_per_mtok_in=1.0, price_per_mtok_out=5.0
)
UNKNOWN_CANDIDATE = Candidate(
    name="mystery-model", provider="mystery", price_per_mtok_in=0, price_per_mtok_out=0
)

SAMPLE = SampleRow(
    trace_id="t1",
    archetype_id="arch-1",
    ts=1000,
    request_json='{"role": "user", "content": "hello"}',
    response_json='{"content": "frontier answer"}',
)


@pytest.mark.asyncio
@respx.mock
async def test_live_call_ollama_success() -> None:
    route = respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200, json={"response": "hi there", "prompt_eval_count": 12, "eval_count": 7}
        )
    )
    result = await _live_call(OLLAMA_CANDIDATE, SAMPLE)
    assert route.called
    assert result.response_text == "hi there"
    assert result.tokens_in == 12
    assert result.tokens_out == 7

    sent_body = respx.calls.last.request.content
    assert b"ollama-llama3.1-8b" in sent_body


@pytest.mark.asyncio
@respx.mock
async def test_live_call_ollama_429_raises_rate_limit_error() -> None:
    respx.post("http://localhost:11434/api/generate").mock(return_value=httpx.Response(429))
    with pytest.raises(RateLimitError):
        await _live_call(OLLAMA_CANDIDATE, SAMPLE)


@pytest.mark.asyncio
@respx.mock
async def test_live_call_anthropic_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-anthropic-key-not-real")
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-haiku-4-5",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )
    )
    result = await _live_call(ANTHROPIC_CANDIDATE, SAMPLE)
    assert route.called
    assert result.response_text == "hello world"
    assert result.tokens_in == 20
    assert result.tokens_out == 10

    sent_headers = respx.calls.last.request.headers
    assert sent_headers["x-api-key"] == "fake-test-anthropic-key-not-real"
    assert sent_headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
@respx.mock
async def test_live_call_anthropic_uses_replay_specific_key_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-subscription-key-not-real")
    monkeypatch.setenv("SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY", "fake-metered-replay-key-not-real")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
    )
    await _live_call(ANTHROPIC_CANDIDATE, SAMPLE)
    assert respx.calls.last.request.headers["x-api-key"] == "fake-metered-replay-key-not-real"


@pytest.mark.asyncio
async def test_live_call_anthropic_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        await _live_call(ANTHROPIC_CANDIDATE, SAMPLE)


@pytest.mark.asyncio
@respx.mock
async def test_live_call_anthropic_429_raises_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-anthropic-key-not-real")
    respx.post("https://api.anthropic.com/v1/messages").mock(return_value=httpx.Response(429))
    with pytest.raises(RateLimitError):
        await _live_call(ANTHROPIC_CANDIDATE, SAMPLE)


@pytest.mark.asyncio
async def test_live_call_unknown_provider_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await _live_call(UNKNOWN_CANDIDATE, SAMPLE)
