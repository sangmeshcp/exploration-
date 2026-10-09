"""Live replay calls to candidate providers (the `CallFn` behind `shadow run`).

Replay needs a metered key, separate from the subscription the primary
capture lane runs on (plan.md D1.4). Endpoints and keys:

  anthropic   SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY, else ANTHROPIC_API_KEY
  ollama      SHADOWTRACE_OLLAMA_URL (default http://localhost:11434)
  vllm        SHADOWTRACE_VLLM_URL (default http://localhost:8000), OpenAI-compatible
  openrouter  OPENROUTER_API_KEY, OpenAI-compatible

Every call sends the stored conversation as real messages (plus the system
prompt and the tools the frontier had or used) and returns the candidate's
text and tool calls, so graders compare like with like.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.messages import (
    parse_response,
    replay_tools,
    request_messages,
    system_prompt,
    to_openai_tools,
)
from shadowtrace.replay.runner import CallResult, RateLimitError
from shadowtrace.replay.sampler import SampleRow

MAX_OUTPUT_TOKENS = 4096
TIMEOUT_S = 120.0


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    if resp.status_code == 429:
        raise RateLimitError(f"{provider} rate limited")
    resp.raise_for_status()


async def _anthropic(candidate: Candidate, sample: SampleRow) -> CallResult:
    api_key = os.environ.get("SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY (or SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY) is required to "
            "replay against anthropic candidates"
        )
    body: dict[str, Any] = {
        "model": candidate.provider_model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": request_messages(sample.request_json),
    }
    system = system_prompt(sample.request_json)
    if system:
        body["system"] = system
    tools = replay_tools(sample.request_json, parse_response(sample.response_json).tool_calls)
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
    _raise_for_status(resp, "anthropic")
    data = resp.json()
    parsed = parse_response(data)
    usage = data.get("usage", {})
    return CallResult(
        response_text=parsed.text,
        tokens_in=usage.get("input_tokens", 0),
        tokens_out=usage.get("output_tokens", 0),
        latency_ms=0,
        tool_calls=parsed.tool_calls,
    )


def _openai_messages(sample: SampleRow) -> list[dict[str, str]]:
    messages = request_messages(sample.request_json)
    system = system_prompt(sample.request_json)
    return ([{"role": "system", "content": system}] if system else []) + messages


async def _ollama(candidate: Candidate, sample: SampleRow) -> CallResult:
    base = os.environ.get("SHADOWTRACE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    body: dict[str, Any] = {
        "model": candidate.provider_model,
        "messages": _openai_messages(sample),
        "stream": False,
    }
    tools = replay_tools(sample.request_json, parse_response(sample.response_json).tool_calls)
    if tools:
        body["tools"] = to_openai_tools(tools)
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(f"{base}/api/chat", json=body)
    _raise_for_status(resp, "ollama")
    data = resp.json()
    message = data.get("message", {}) or {}
    tool_calls = [
        {
            "name": (c.get("function") or {}).get("name", ""),
            "input": (c.get("function") or {}).get("arguments") or {},
        }
        for c in message.get("tool_calls") or []
    ]
    return CallResult(
        response_text=str(message.get("content") or ""),
        tokens_in=data.get("prompt_eval_count", 0),
        tokens_out=data.get("eval_count", 0),
        latency_ms=0,
        tool_calls=tool_calls,
    )


async def _openai_compatible(
    candidate: Candidate, sample: SampleRow, url: str, api_key: str | None
) -> CallResult:
    body: dict[str, Any] = {
        "model": candidate.provider_model,
        "messages": _openai_messages(sample),
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    tools = replay_tools(sample.request_json, parse_response(sample.response_json).tool_calls)
    if tools:
        body["tools"] = to_openai_tools(tools)
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(url, headers=headers, json=body)
    _raise_for_status(resp, candidate.provider)
    data = resp.json()
    parsed = parse_response(data)
    usage = data.get("usage", {}) or {}
    return CallResult(
        response_text=parsed.text,
        tokens_in=usage.get("prompt_tokens", 0),
        tokens_out=usage.get("completion_tokens", 0),
        latency_ms=0,
        tool_calls=parsed.tool_calls,
    )


async def live_call(candidate: Candidate, sample: SampleRow) -> CallResult:
    if candidate.provider == "anthropic":
        return await _anthropic(candidate, sample)
    if candidate.provider == "ollama":
        return await _ollama(candidate, sample)
    if candidate.provider == "vllm":
        base = os.environ.get("SHADOWTRACE_VLLM_URL", "http://localhost:8000").rstrip("/")
        return await _openai_compatible(candidate, sample, f"{base}/v1/chat/completions", None)
    if candidate.provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to replay against openrouter")
        return await _openai_compatible(
            candidate, sample, "https://openrouter.ai/api/v1/chat/completions", api_key
        )
    raise NotImplementedError(f"no live-call wiring for provider={candidate.provider!r} yet")
