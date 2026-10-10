"""LLM-as-judge backed by a cheap Claude model, via the official Anthropic SDK
(`pip install '.[judge]'`).

Plugs into `judge.judge()` / the `judge` scorer as a `judge_fn`. Returns a
`JudgeVerdict` carrying what the call cost, so judge spend counts against
`shadow run --budget` like candidate spend does.

The judge model should come from a different family than the candidates
under test where possible (review R4: a judge shares its family's biases).
`human_review.py` measures how far the judge can be trusted by comparing
its verdicts against human labels.
"""

from __future__ import annotations

import os
import re
from typing import Any

from shadowtrace.common.instrument import genai_attributes, instrumented
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.graders.judge import JUDGE_PROMPT_TEMPLATE, JudgeVerdict

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"

# $ per million tokens (input, output). Unknown models are costed at the
# most expensive known rate so an unrecognized judge can't under-report spend.
JUDGE_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b")


class JudgeError(RuntimeError):
    """The judge produced no usable verdict (refusal or unparseable output)."""


def parse_verdict(text: str) -> bool:
    matches = _VERDICT_RE.findall(text.upper())
    if not matches:
        raise JudgeError(f"no PASS/FAIL in judge output: {text[:80]!r}")
    return bool(matches[-1] == "PASS")


def judge_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = JUDGE_PRICES.get(model, max(JUDGE_PRICES.values()))
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


class AnthropicJudge:
    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        client: Any | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    async def __call__(
        self, rubric: str, frontier_answer: str, candidate_answer: str
    ) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            rubric=rubric, frontier_answer=frontier_answer, candidate_answer=candidate_answer
        )
        with instrumented(
            "judge.call",
            component="replay",
            attributes=genai_attributes(provider="anthropic", request_model=self.model),
        ) as span:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt}],
            )
            usage = response.usage
            cost = judge_cost_usd(self.model, usage.input_tokens, usage.output_tokens)
            span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
            span.set_attribute("cost_usd", cost)
            REGISTRY.inc("judge_spend_usd", value=cost, labels={"model": self.model})
            if response.stop_reason == "refusal":
                raise JudgeError("judge model refused")
            text = "".join(
                str(getattr(block, "text", ""))
                for block in response.content
                if block.type == "text"
            )
            return JudgeVerdict(passed=parse_verdict(text), cost_usd=cost)


def judge_from_env() -> AnthropicJudge | None:
    """An `AnthropicJudge` when a judge key is configured, else None (callers
    fall back to the local heuristic). `SHADOWTRACE_JUDGE_MODEL` overrides
    the model; the key comes from SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY, then
    the replay key, then ANTHROPIC_API_KEY."""
    api_key = (
        os.environ.get("SHADOWTRACE_JUDGE_ANTHROPIC_API_KEY")
        or os.environ.get("SHADOWTRACE_REPLAY_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    model = os.environ.get("SHADOWTRACE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    return AnthropicJudge(model=model, client=anthropic.AsyncAnthropic(api_key=api_key))
