"""Pairwise judge: candidate answer vs. the stored frontier answer (free
baseline) (plan.md M3.4).

Circularity risk (review R4): a judge — however implemented — shares
family bias with the frontier baseline it's comparing against. That's
mitigated by a human spot-review queue downstream (M4.5), not by this
module. `JUDGE_PROMPT_VERSION` and the per-verdict hash exist so every
stored verdict is traceable to exactly which rubric text produced it,
which the spot-review queue and any later re-grading depend on.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

JUDGE_PROMPT_VERSION = "v1"

JUDGE_PROMPT_TEMPLATE = """\
You are grading whether a candidate model's answer is an acceptable \
substitute for a frontier model's answer to the same request, under this rubric:

{rubric}

Frontier answer (reference, known-good):
{frontier_answer}

Candidate answer (to grade):
{candidate_answer}

Respond with exactly one word: PASS or FAIL.
"""

JudgeFn = Callable[[str, str, str], Awaitable[bool]]

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


async def heuristic_judge(rubric: str, frontier_answer: str, candidate_answer: str) -> bool:
    """Free, fully-local fallback judge: token-overlap similarity against
    the frontier answer. Deliberately crude — it exists so replay runs,
    tests, and CI never require live API access to produce a verdict. Real
    deployments wire a cheap LLM call in via `judge_fn`."""
    frontier_tokens = set(_WORD_RE.findall(frontier_answer.lower()))
    candidate_tokens = set(_WORD_RE.findall(candidate_answer.lower()))
    if not frontier_tokens:
        return not candidate_tokens
    overlap = len(frontier_tokens & candidate_tokens) / len(frontier_tokens)
    return overlap >= 0.5


@dataclass
class JudgeResult:
    passed: bool
    prompt_version: str
    result_hash: str


def _stamp_hash(rubric: str, frontier_answer: str, candidate_answer: str, passed: bool) -> str:
    payload = f"{JUDGE_PROMPT_VERSION}|{rubric}|{frontier_answer}|{candidate_answer}|{passed}"
    return hashlib.sha256(payload.encode()).hexdigest()


async def judge(
    rubric: str,
    frontier_answer: str,
    candidate_answer: str,
    judge_fn: JudgeFn | None = None,
) -> JudgeResult:
    fn = judge_fn or heuristic_judge
    passed = await fn(rubric, frontier_answer, candidate_answer)
    return JudgeResult(
        passed=passed,
        prompt_version=JUDGE_PROMPT_VERSION,
        result_hash=_stamp_hash(rubric, frontier_answer, candidate_answer, passed),
    )
