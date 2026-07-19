import pytest

from shadowtrace.replay.graders.deterministic import (
    all_pass,
    is_valid_json,
    is_valid_python,
    is_well_formed_unified_diff,
    run_deterministic_checks,
    within_length,
)
from shadowtrace.replay.graders.judge import heuristic_judge, judge


def test_is_valid_json() -> None:
    assert is_valid_json('{"a": 1}') is True
    assert is_valid_json("not json") is False


def test_is_valid_python() -> None:
    assert is_valid_python("def f():\n    return 1\n") is True
    assert is_valid_python("def f(:\n") is False


def test_is_well_formed_unified_diff() -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-old\n+new\n"
    assert is_well_formed_unified_diff(diff) is True
    assert is_well_formed_unified_diff("just some text") is False


def test_within_length_check() -> None:
    check = within_length(min_chars=5, max_chars=10)
    assert check("hello") is True
    assert check("hi") is False
    assert check("way too long for this check") is False


def test_run_deterministic_checks_and_all_pass() -> None:
    verdicts = run_deterministic_checks(
        '{"ok": true}', {"json": is_valid_json, "length": within_length(1, 1000)}
    )
    assert all_pass(verdicts) is True

    verdicts2 = run_deterministic_checks("not json", {"json": is_valid_json})
    assert all_pass(verdicts2) is False


@pytest.mark.asyncio
async def test_heuristic_judge_high_overlap_passes() -> None:
    frontier = "The capital of France is Paris."
    candidate = "Paris is the capital of France."
    assert await heuristic_judge("rubric", frontier, candidate) is True


@pytest.mark.asyncio
async def test_heuristic_judge_low_overlap_fails() -> None:
    frontier = "The capital of France is Paris."
    candidate = "I like turtles and skateboarding."
    assert await heuristic_judge("rubric", frontier, candidate) is False


@pytest.mark.asyncio
async def test_judge_result_is_hash_stamped_and_deterministic() -> None:
    r1 = await judge("rubric", "answer a", "answer a")
    r2 = await judge("rubric", "answer a", "answer a")
    assert r1.result_hash == r2.result_hash
    assert r1.prompt_version == "v1"

    r3 = await judge("different rubric", "answer a", "answer a")
    assert r3.result_hash != r1.result_hash


@pytest.mark.asyncio
async def test_judge_accepts_custom_judge_fn() -> None:
    async def always_fail(rubric: str, frontier: str, candidate: str) -> bool:
        return False

    result = await judge("rubric", "x", "x", judge_fn=always_fail)
    assert result.passed is False
