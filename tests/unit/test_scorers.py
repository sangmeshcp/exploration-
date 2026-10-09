import json
from pathlib import Path

import pytest

from shadowtrace.common.metrics import REGISTRY
from shadowtrace.replay.graders.judge import JudgeVerdict
from shadowtrace.replay.graders.scorers import (
    EvalCase,
    EvalSuite,
    Score,
    checkable_specifics,
    completion_scorer,
    default_suite,
    eval_case_from,
    format_scorer,
    is_refusal,
    make_grounding_scorer,
    make_judge_scorer,
    make_trajectory_scorer,
    trajectory_similarity,
)
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import CallResult, ReplayRunner, RunnerConfig
from shadowtrace.replay.sampler import SampleRow


def _case(**kwargs: object) -> EvalCase:
    base: dict[str, object] = {
        "request_text": "q",
        "reference_text": "answer",
        "candidate_text": "answer",
    }
    base.update(kwargs)
    return EvalCase(**base)  # type: ignore[arg-type]


async def test_format_scorer_requires_same_structure_as_frontier() -> None:
    ok = await format_scorer(_case(reference_text='{"a": 1}', candidate_text='{"b": 2}'))
    bad = await format_scorer(_case(reference_text='{"a": 1}', candidate_text="sure: a=1"))
    assert ok is not None and ok.passed
    assert bad is not None and not bad.passed and bad.value == 0.0

    diff = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    diff_score = await format_scorer(_case(reference_text=diff, candidate_text="changed a to b"))
    assert diff_score is not None and not diff_score.passed

    py_ref = "Here:\n```python\nx = 1\n```"
    good_py = await format_scorer(
        _case(reference_text=py_ref, candidate_text="```python\ny = 2\n```")
    )
    broken_py = await format_scorer(
        _case(reference_text=py_ref, candidate_text="```python\ndef (:\n```")
    )
    assert good_py is not None and good_py.passed
    assert broken_py is not None and not broken_py.passed

    assert await format_scorer(_case(reference_text="just prose")) is None


async def test_completion_scorer_catches_empty_and_unmatched_refusals() -> None:
    empty = await completion_scorer(_case(candidate_text="  "))
    assert empty is not None and not empty.passed
    tool_only = await completion_scorer(
        _case(candidate_text="", candidate_tool_calls=[{"name": "x", "input": {}}])
    )
    assert tool_only is not None and tool_only.passed

    refused = await completion_scorer(_case(candidate_text="I can't help with that."))
    assert refused is not None and not refused.passed
    both_refuse = await completion_scorer(
        _case(reference_text="Sorry, I cannot do that.", candidate_text="I cannot do that.")
    )
    assert both_refuse is not None and both_refuse.passed
    assert is_refusal("As an AI, I won't")
    assert not is_refusal("Sorting the list in place")


def test_trajectory_similarity_weights_tool_choice_over_arguments() -> None:
    ref = [
        {"name": "read_file", "input": {"path": "a"}},
        {"name": "run_tests", "input": {}},
    ]
    same, _ = trajectory_similarity(ref, ref)
    assert same == pytest.approx(1.0)

    other_args, _ = trajectory_similarity(
        ref, [{"name": "read_file", "input": {"file": "a"}}, {"name": "run_tests", "input": {}}]
    )
    assert other_args == pytest.approx(0.7 + 0.3 * 0.5)

    wrong_tool, detail = trajectory_similarity(ref, [{"name": "delete_file", "input": {}}])
    assert wrong_tool == pytest.approx(0.0)
    assert "delete_file" in detail

    assert trajectory_similarity([], [])[0] == pytest.approx(1.0)


async def test_trajectory_scorer_only_applies_when_tools_are_involved() -> None:
    scorer = make_trajectory_scorer(threshold=0.7)
    assert await scorer(_case()) is None
    missing = await scorer(_case(reference_tool_calls=[{"name": "grep", "input": {}}]))
    assert missing is not None and not missing.passed


def test_checkable_specifics_extracts_paths_ids_numbers() -> None:
    found = checkable_specifics(
        "Edit `config.yaml` at src/server/app.py, set max_retries to 30, see https://x.io/a."
    )
    assert {"config.yaml", "src/server/app.py", "max_retries", "30", "https://x.io/a"} <= found


async def test_grounding_scorer_flags_invented_specifics() -> None:
    scorer = make_grounding_scorer(threshold=0.8)
    grounded = await scorer(
        _case(
            request_text="why does src/app.py fail on port 8080?",
            reference_text="src/app.py binds 8080 twice",
            candidate_text="src/app.py binds port 8080 twice",
        )
    )
    assert grounded is not None and grounded.passed and grounded.value == 1.0

    invented = await scorer(
        _case(
            request_text="why does src/app.py fail?",
            reference_text="it binds the port twice",
            candidate_text="set retry_limit in lib/net/socket.py to 9000",
        )
    )
    assert invented is not None and not invented.passed
    assert "socket.py" in invented.detail

    assert await scorer(_case(candidate_text="looks fine to me")) is None


async def test_judge_scorer_passes_cost_through_and_fails_closed_on_error() -> None:
    async def paid_judge(rubric: str, frontier: str, candidate: str) -> JudgeVerdict:
        assert "[tool calls]" in candidate
        return JudgeVerdict(passed=True, cost_usd=0.002)

    score = await make_judge_scorer(paid_judge)(
        _case(candidate_tool_calls=[{"name": "grep", "input": {"p": "x"}}])
    )
    assert score is not None and score.passed and score.cost_usd == pytest.approx(0.002)
    assert score.detail.startswith("v1:")

    async def broken_judge(rubric: str, frontier: str, candidate: str) -> bool:
        raise TimeoutError("judge down")

    REGISTRY.reset()
    failed = await make_judge_scorer(broken_judge)(_case())
    assert failed is not None and not failed.passed
    assert failed.detail == "judge_error:TimeoutError"
    assert REGISTRY.find("judge_errors_total") == 1


async def test_suite_gates_only_on_gating_scorers() -> None:
    async def always_fail(case: EvalCase) -> Score:
        return Score("advisory", 0.0, False)

    async def not_applicable(case: EvalCase) -> None:
        return None

    advisory = EvalSuite([(always_fail, False), (not_applicable, True)])
    grade = await advisory.grade(_case())
    assert grade.passed
    assert [s.name for s in grade.scores] == ["advisory"]
    assert grade.score("advisory") is not None and grade.score("missing") is None

    gating = EvalSuite([(always_fail, True)])
    assert not (await gating.grade(_case())).passed


async def test_default_suite_end_to_end_on_a_tool_use_sample(tmp_path: Path) -> None:
    sample = SampleRow(
        trace_id="t1",
        archetype_id="agentic",
        ts=1,
        request_json=json.dumps({"role": "user", "content": "what port does config.yaml set?"}),
        response_json=json.dumps(
            {
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "name": "read_file", "input": {"path": "config.yaml"}},
                ]
            }
        ),
    )
    good = CallResult("Let me check.", 1, 1, 0, [{"name": "read_file", "input": {"path": "x"}}])
    bad = CallResult("Probably 8080.", 1, 1, 0, [])

    suite = default_suite()
    case = eval_case_from(sample, good)
    assert case.reference_tool_calls == [{"name": "read_file", "input": {"path": "config.yaml"}}]
    good_grade = await suite(sample, good)
    bad_grade = await suite(sample, bad)
    assert good_grade.passed
    assert not bad_grade.passed
    trajectory = bad_grade.score("trajectory")
    assert trajectory is not None and not trajectory.passed


async def test_runner_records_scores_and_counts_grader_cost_in_budget(tmp_path: Path) -> None:
    async def paid_judge(rubric: str, frontier: str, candidate: str) -> JudgeVerdict:
        return JudgeVerdict(passed=True, cost_usd=0.5)

    async def call(candidate: Candidate, sample: SampleRow) -> CallResult:
        return CallResult("answer", tokens_in=0, tokens_out=0, latency_ms=0)

    samples = [
        SampleRow(f"t{i}", "a1", i, '{"role": "user", "content": "q"}', '"answer"')
        for i in range(5)
    ]
    runner = ReplayRunner(
        [Candidate("local", "ollama", 0.0, 0.0, local=True)],
        EvalSuite([(make_judge_scorer(paid_judge), True)]),
        call,
        RunnerConfig(budget_usd=1.0, checkpoint_path=tmp_path / "cp.json"),
    )
    result = await runner.run("r", samples)
    # a $0 local candidate still spends the judge's $0.50 per grade: stops after two
    assert len(result.results) == 2
    assert result.stopped_reason == "budget_exhausted"
    assert result.results[0].response_text == "answer"
    assert result.results[0].latency_ms >= 0
    assert [s.name for s in result.results[0].scores] == ["judge"]
