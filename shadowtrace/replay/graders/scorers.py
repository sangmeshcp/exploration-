"""Composable evaluation scorers for LLM and agentic replay.

A scorer looks at one `EvalCase` (the request, the stored frontier answer and
the candidate's answer, each with any tool calls) and returns a `Score` in
[0, 1], or None when it doesn't apply to that case (a JSON-format check on a
prose answer, a trajectory check when nobody called a tool). An `EvalSuite`
runs a set of scorers and folds the gating ones into a single pass/fail
`Grade`, keeping every individual score so results can be broken down per
scorer, per archetype and per candidate afterwards.

Built-in scorers, cheapest first:

  format      deterministic: if the frontier answer was JSON / parseable
              Python / a unified diff, the candidate's must be too
  completion  task success: the candidate answered (not empty, not a refusal
              the frontier didn't make)
  trajectory  tool use: did the candidate pick the same tools, in the same
              order, with the same argument keys, as the frontier
  grounding   hallucination proxy: share of the candidate's checkable
              specifics (numbers, paths, identifiers, URLs, quoted strings)
              that appear in the request or the frontier answer
  judge       LLM-as-judge (or the local heuristic) pairwise vs. frontier

Custom scorers are plain async functions `(EvalCase) -> Score | None`, so a
team-specific check (a schema, a regex, a unit test run) slots in next to
these without touching the runner.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.graders import deterministic
from shadowtrace.replay.graders.judge import JudgeFn, judge
from shadowtrace.replay.messages import parse_response, request_text

if TYPE_CHECKING:
    from shadowtrace.replay.runner import CallResult
    from shadowtrace.replay.sampler import SampleRow

logger = get_logger("replay.scorers")


@dataclass
class Score:
    name: str
    value: float
    passed: bool
    detail: str = ""
    cost_usd: float = 0.0  # what producing this score spent (an LLM judge call)


@dataclass
class Grade:
    passed: bool
    scores: list[Score] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.scores)

    def score(self, name: str) -> Score | None:
        return next((s for s in self.scores if s.name == name), None)


@dataclass
class EvalCase:
    request_text: str
    reference_text: str
    candidate_text: str
    reference_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    candidate_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    archetype_id: str = ""


Scorer = Callable[[EvalCase], Awaitable[Score | None]]


def eval_case_from(sample: SampleRow, result: CallResult) -> EvalCase:
    reference = parse_response(sample.response_json)
    return EvalCase(
        request_text=request_text(sample.request_json),
        reference_text=reference.text,
        candidate_text=result.response_text,
        reference_tool_calls=reference.tool_calls,
        candidate_tool_calls=list(result.tool_calls),
        archetype_id=sample.archetype_id,
    )


# --- format -----------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)


def _python_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _FENCE_RE.finditer(text)]


def _all_parse(blocks: list[str]) -> bool:
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError:
            return False
    return True


async def format_scorer(case: EvalCase) -> Score | None:
    ref = case.reference_text.strip()
    cand = case.candidate_text.strip()
    if ref and deterministic.is_valid_json(ref):
        ok = deterministic.is_valid_json(cand)
        return Score("format", 1.0 if ok else 0.0, ok, "expected JSON")
    if deterministic.is_well_formed_unified_diff(ref):
        ok = deterministic.is_well_formed_unified_diff(cand)
        return Score("format", 1.0 if ok else 0.0, ok, "expected unified diff")
    ref_blocks = _python_blocks(ref)
    if ref_blocks and _all_parse(ref_blocks):
        cand_blocks = _python_blocks(cand)
        ok = bool(cand_blocks) and _all_parse(cand_blocks)
        return Score("format", 1.0 if ok else 0.0, ok, "expected parseable python block")
    return None


# --- completion -------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"^\s*(i'?m sorry|sorry|i can(?:'|no)t|i cannot|i am unable|i'm unable|i won'?t|as an ai)\b",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text[:200]))


async def completion_scorer(case: EvalCase) -> Score | None:
    if not case.candidate_text.strip() and not case.candidate_tool_calls:
        return Score("completion", 0.0, False, "empty answer")
    if is_refusal(case.candidate_text) and not is_refusal(case.reference_text):
        return Score("completion", 0.0, False, "refused where the frontier answered")
    return Score("completion", 1.0, True)


# --- trajectory -------------------------------------------------------------


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _key_jaccard(a: dict[str, Any], b: dict[str, Any]) -> float:
    ka, kb = set(a), set(b)
    if not ka and not kb:
        return 1.0
    return len(ka & kb) / len(ka | kb)


def trajectory_similarity(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> tuple[float, str]:
    """0.7 x tool-sequence similarity (LCS over tool names) + 0.3 x mean
    argument-key overlap on positions where both called the same tool.
    Tool choice dominates: calling the right tool with a slightly different
    argument set is a much smaller miss than calling the wrong tool."""
    ref_names = [str(c.get("name", "")) for c in reference]
    cand_names = [str(c.get("name", "")) for c in candidate]
    total = len(ref_names) + len(cand_names)
    seq_sim = 2 * _lcs_len(ref_names, cand_names) / total if total else 1.0
    pairs = [
        _key_jaccard(r.get("input") or {}, c.get("input") or {})
        for r, c in zip(reference, candidate, strict=False)
        if r.get("name") == c.get("name")
    ]
    arg_sim = sum(pairs) / len(pairs) if pairs else (1.0 if total == 0 else 0.0)
    detail = f"expected {ref_names} got {cand_names}"
    return 0.7 * seq_sim + 0.3 * arg_sim, detail


def make_trajectory_scorer(threshold: float = 0.7) -> Scorer:
    async def trajectory_scorer(case: EvalCase) -> Score | None:
        if not case.reference_tool_calls and not case.candidate_tool_calls:
            return None
        value, detail = trajectory_similarity(case.reference_tool_calls, case.candidate_tool_calls)
        return Score("trajectory", value, value >= threshold, detail)

    return trajectory_scorer


# --- grounding --------------------------------------------------------------

_SPECIFIC_PATTERNS = [
    re.compile(r"https?://[^\s)>\]]+"),
    re.compile(r"(?:[\w.-]+/)+[\w.-]+"),  # paths
    re.compile(r"\b[\w-]+\.(?:py|js|ts|tsx|go|rs|java|json|ya?ml|toml|md|sql|sh|txt|cfg|ini)\b"),
    re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b"),  # snake_case
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),  # camelCase
    re.compile(r"\b\d+(?:\.\d+)+\b|\b\d{2,}\b"),  # versions, multi-digit numbers
    re.compile(r"`([^`\n]{2,60})`"),  # inline code
]


def checkable_specifics(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in _SPECIFIC_PATTERNS:
        for match in pattern.finditer(text):
            token = (match.group(1) if match.groups() else match.group(0)).strip().rstrip(".,;:")
            if len(token) >= 2:
                found.add(token)
    return found


def make_grounding_scorer(threshold: float = 0.8, min_specifics: int = 2) -> Scorer:
    async def grounding_scorer(case: EvalCase) -> Score | None:
        specifics = checkable_specifics(case.candidate_text)
        if len(specifics) < min_specifics:
            return None
        sources = (case.request_text + "\n" + case.reference_text).lower()
        unsupported = sorted(s for s in specifics if s.lower() not in sources)
        value = 1 - len(unsupported) / len(specifics)
        detail = f"unsupported: {unsupported[:5]}" if unsupported else ""
        return Score("grounding", value, value >= threshold, detail)

    return grounding_scorer


# --- judge ------------------------------------------------------------------

DEFAULT_RUBRIC = (
    "Is the candidate answer an acceptable substitute for the frontier answer? "
    "It must address the same request, be factually consistent with it, and "
    "not invent files, APIs or values the frontier answer and request don't support."
)


def make_judge_scorer(judge_fn: JudgeFn | None = None, rubric: str = DEFAULT_RUBRIC) -> Scorer:
    async def judge_scorer(case: EvalCase) -> Score | None:
        candidate = case.candidate_text
        if case.candidate_tool_calls:
            candidate += "\n[tool calls] " + json.dumps(case.candidate_tool_calls, sort_keys=True)
        reference = case.reference_text
        if case.reference_tool_calls:
            reference += "\n[tool calls] " + json.dumps(case.reference_tool_calls, sort_keys=True)
        try:
            result = await judge(rubric, reference, candidate, judge_fn=judge_fn)
        except Exception as exc:
            # Fail closed: an unanswered judge call must not let a candidate
            # through on the remaining (cheaper, weaker) scorers alone.
            REGISTRY.inc("judge_errors_total", labels={"error": type(exc).__name__})
            logger.warning("judge call failed: %s: %s", type(exc).__name__, exc)
            return Score("judge", 0.0, False, f"judge_error:{type(exc).__name__}")
        return Score(
            "judge",
            1.0 if result.passed else 0.0,
            result.passed,
            f"{result.prompt_version}:{result.result_hash[:12]}",
            cost_usd=result.cost_usd,
        )

    return judge_scorer


# --- suite ------------------------------------------------------------------


@dataclass
class EvalSuite:
    """Runs scorers in order. Gating scorers decide pass/fail; the rest are
    recorded for analysis only (useful while a new scorer is being
    calibrated against human review before it's trusted to gate)."""

    scorers: list[tuple[Scorer, bool]]

    async def grade(self, case: EvalCase) -> Grade:
        scores: list[Score] = []
        passed = True
        for scorer, gating in self.scorers:
            score = await scorer(case)
            if score is None:
                continue
            scores.append(score)
            if gating and not score.passed:
                passed = False
        return Grade(passed=passed, scores=scores)

    async def __call__(self, sample: SampleRow, result: CallResult) -> Grade:
        return await self.grade(eval_case_from(sample, result))


def default_suite(judge_fn: JudgeFn | None = None) -> EvalSuite:
    return EvalSuite(
        scorers=[
            (format_scorer, True),
            (completion_scorer, True),
            (make_trajectory_scorer(), True),
            (make_grounding_scorer(), False),
            (make_judge_scorer(judge_fn), True),
        ]
    )
