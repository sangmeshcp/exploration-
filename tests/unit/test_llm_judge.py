from dataclasses import dataclass, field
from typing import Any

import pytest

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.graders import llm_judge
from shadowtrace.replay.graders.llm_judge import (
    AnthropicJudge,
    JudgeError,
    judge_cost_usd,
    judge_from_env,
    parse_verdict,
)


@dataclass
class _Block:
    type: str
    text: str = ""


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    content: list[_Block]
    usage: _Usage
    stop_reason: str = "end_turn"


@dataclass
class _Messages:
    response: _Response
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return self.response


@dataclass
class _Client:
    messages: _Messages


def _client(text: str, stop_reason: str = "end_turn") -> _Client:
    return _Client(
        _Messages(
            _Response(
                [_Block("thinking"), _Block("text", text)],
                _Usage(1_000_000, 100_000),
                stop_reason,
            )
        )
    )


def test_parse_verdict_uses_last_verdict_word() -> None:
    assert parse_verdict("PASS") is True
    assert parse_verdict("Not a FAIL... final answer: pass") is True
    assert parse_verdict("fail") is False
    with pytest.raises(JudgeError):
        parse_verdict("looks okay to me")


def test_unknown_judge_model_is_priced_at_the_most_expensive_rate() -> None:
    assert judge_cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.6)
    assert judge_cost_usd("mystery", 1_000_000, 0) == pytest.approx(4.0)


async def test_anthropic_judge_request_shape_verdict_and_cost() -> None:
    REGISTRY.reset()
    client = _client("PASS")
    judge = AnthropicJudge(client=client)
    verdict = await judge("rubric text", "frontier", "candidate")

    assert verdict.passed
    assert verdict.cost_usd == pytest.approx(0.1 + 0.05)
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"] == {"effort": "low"}
    prompt = call["messages"][0]["content"]
    assert "rubric text" in prompt and "frontier" in prompt and "candidate" in prompt
    assert REGISTRY.find("judge_spend_usd") == pytest.approx(0.15)


async def test_anthropic_judge_refusal_raises() -> None:
    with pytest.raises(JudgeError):
        await AnthropicJudge(client=_client("", stop_reason="refusal"))("r", "f", "c")


def test_judge_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY",
        "SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    assert judge_from_env() is None

    monkeypatch.setenv("SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY", "fake-judge-key-not-real")
    monkeypatch.setenv("SHADOWTRACE_JUDGE_MODEL", "claude-sonnet-5-5")
    judge = judge_from_env()
    assert judge is not None
    assert judge.model == "claude-sonnet-5-5"
    assert llm_judge.DEFAULT_JUDGE_MODEL == "claude-haiku-5-5"
