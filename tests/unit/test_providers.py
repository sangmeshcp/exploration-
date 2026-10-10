import json

import httpx
import pytest
import respx

from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.providers import live_call
from shadowtrace.replay.runner import RateLimitError
from shadowtrace.replay.sampler import SampleRow

TOOL_SAMPLE = SampleRow(
    trace_id="t1",
    archetype_id="a1",
    ts=1,
    request_json=json.dumps(
        {
            "system": "you are a coding agent",
            "messages": [{"role": "user", "content": "what port?"}],
        }
    ),
    response_json=json.dumps(
        {"content": [{"type": "tool_use", "name": "read_file", "input": {"path": "c.yaml"}}]}
    ),
)

VLLM = Candidate("vllm-q", "vllm", 0, 0, local=True, model="Qwen/Qwen2.5-14B-Instruct")
OPENROUTER = Candidate("or-llama", "openrouter", 0.4, 0.4, model="meta-llama/llama-3.1-70b")
HAIKU = Candidate("claude-haiku-5-5", "anthropic", 0.1, 0.5)

_OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "c.yaml"}'}}
                ],
            }
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 5},
}


@respx.mock
async def test_vllm_is_called_openai_style_with_system_and_inferred_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHADOWTRACE_VLLM_URL", "http://gpu-box:9000/")
    route = respx.post("http://gpu-box:9000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_RESPONSE)
    )
    result = await live_call(VLLM, TOOL_SAMPLE)
    assert route.called
    sent = json.loads(respx.calls.last.request.content)
    assert sent["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert sent["messages"][0] == {"role": "system", "content": "you are a coding agent"}
    assert sent["tools"][0]["function"]["name"] == "read_file"
    assert "authorization" not in respx.calls.last.request.headers
    assert result.tool_calls == [{"name": "read_file", "input": {"path": "c.yaml"}}]
    assert (result.tokens_in, result.tokens_out) == (30, 5)


@respx.mock
async def test_openrouter_needs_key_and_sends_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await live_call(OPENROUTER, TOOL_SAMPLE)

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key-not-real")
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_RESPONSE)
    )
    await live_call(OPENROUTER, TOOL_SAMPLE)
    assert (
        respx.calls.last.request.headers["authorization"] == "Bearer fake-openrouter-key-not-real"
    )


@respx.mock
async def test_openai_compatible_429_is_a_rate_limit() -> None:
    respx.post("http://localhost:8000/v1/chat/completions").mock(return_value=httpx.Response(429))
    with pytest.raises(RateLimitError):
        await live_call(VLLM, TOOL_SAMPLE)


@respx.mock
async def test_anthropic_gets_real_messages_system_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-anthropic-key-not-real")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "a"}}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )
    )
    result = await live_call(HAIKU, TOOL_SAMPLE)
    sent = json.loads(respx.calls.last.request.content)
    assert sent["messages"] == [{"role": "user", "content": "what port?"}]
    assert sent["system"] == "you are a coding agent"
    assert sent["tools"][0]["name"] == "read_file"
    assert result.tool_calls == [{"name": "read_file", "input": {"path": "a"}}]


@respx.mock
async def test_ollama_returns_tool_calls() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}],
                }
            },
        )
    )
    result = await live_call(Candidate("o", "ollama", 0, 0, local=True), TOOL_SAMPLE)
    assert result.tool_calls == [{"name": "read_file", "input": {"path": "a"}}]
    assert result.tokens_in == 0
