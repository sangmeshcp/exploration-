import json
from pathlib import Path

import pytest

from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import CallResult, RateLimitError, ReplayRunner, RunnerConfig
from shadowtrace.replay.sampler import ReplayChain, SampleRow


def _samples(n: int) -> list[SampleRow]:
    return [
        SampleRow(
            trace_id=f"t{i}",
            archetype_id="arch-1",
            ts=1000 + i,
            request_json="{}",
            response_json="frontier answer",
        )
        for i in range(n)
    ]


CANDIDATE = Candidate(
    name="claude-haiku-4-5", provider="anthropic", price_per_mtok_in=1.0, price_per_mtok_out=5.0
)


async def _fixed_cost_call(candidate: Candidate, sample: SampleRow) -> CallResult:
    # 1,000,000 in / 200,000 out -> cost = 1*1.0 + 0.2*5.0 = $2.00 per call
    return CallResult(response_text="ok", tokens_in=1_000_000, tokens_out=200_000, latency_ms=5)


async def _always_pass_grader(sample: SampleRow, result: CallResult) -> bool:
    return True


@pytest.mark.asyncio
async def test_runner_stops_within_one_call_of_budget_cap(tmp_path: Path) -> None:
    config = RunnerConfig(budget_usd=3.0, checkpoint_path=tmp_path / "cp.json", concurrency=1)
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, _fixed_cost_call, config)
    result = await runner.run("run-1", _samples(5))
    # $2.00/call, $3.00 cap: 1st call ($2.00, under cap) proceeds, 2nd call
    # would start with spend=$2.00 < $3.00 so it also proceeds ($4.00 total),
    # 3rd call sees spend=$4.00 >= $3.00 and stops.
    assert result.stopped_reason == "budget_exhausted"
    assert len(result.results) == 2
    assert result.spend_usd == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_runner_completes_under_budget(tmp_path: Path) -> None:
    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json", concurrency=1)
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, _fixed_cost_call, config)
    result = await runner.run("run-2", _samples(3))
    assert result.stopped_reason == "completed"
    assert len(result.results) == 3
    assert all(r.verdict == "pass" for r in result.results)


@pytest.mark.asyncio
async def test_runner_resume_from_checkpoint_does_not_double_spend(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "cp.json"
    config = RunnerConfig(budget_usd=3.0, checkpoint_path=checkpoint_path, concurrency=1)
    runner1 = ReplayRunner([CANDIDATE], _always_pass_grader, _fixed_cost_call, config)
    result1 = await runner1.run("run-3", _samples(5))
    assert result1.stopped_reason == "budget_exhausted"
    spend_after_first = result1.spend_usd

    # "resume" with a fresh runner pointed at the same checkpoint + higher budget
    config2 = RunnerConfig(budget_usd=100.0, checkpoint_path=checkpoint_path, concurrency=1)
    runner2 = ReplayRunner([CANDIDATE], _always_pass_grader, _fixed_cost_call, config2)
    result2 = await runner2.run("run-3", _samples(5))

    assert result2.stopped_reason == "completed"
    # already-completed trace/candidate pairs from run 1 must not be re-billed
    assert result2.spend_usd == pytest.approx(spend_after_first + 3 * 2.0)
    assert len(result2.results) == 3  # only the 3 remaining samples


@pytest.mark.asyncio
async def test_runner_retries_on_rate_limit_then_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    async def flaky_call(candidate: Candidate, sample: SampleRow) -> CallResult:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("slow down")
        return CallResult(response_text="ok", tokens_in=1000, tokens_out=1000, latency_ms=5)

    config = RunnerConfig(
        budget_usd=100.0, checkpoint_path=tmp_path / "cp.json", concurrency=1, backoff_base_s=0.001
    )
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, flaky_call, config)
    result = await runner.run("run-4", _samples(1))
    assert result.stopped_reason == "completed"
    assert len(result.results) == 1
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_runner_gives_up_after_max_retries(tmp_path: Path) -> None:
    async def always_rate_limited(candidate: Candidate, sample: SampleRow) -> CallResult:
        raise RateLimitError("nope")

    config = RunnerConfig(
        budget_usd=100.0,
        checkpoint_path=tmp_path / "cp.json",
        concurrency=1,
        max_retries=2,
        backoff_base_s=0.001,
    )
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, always_rate_limited, config)
    result = await runner.run("run-5", _samples(1))
    assert result.results == []  # call failed after exhausting retries, no result recorded


def _chain(session_id: str, n_steps: int) -> ReplayChain:
    steps = [
        SampleRow(
            trace_id=f"{session_id}-{i}",
            archetype_id="arch-1",
            ts=1000 + i,
            request_json=json.dumps({"role": "user", "content": f"original user turn {i}"}),
            response_json=json.dumps({"content": f"original frontier answer {i}"}),
            session_id=session_id,
        )
        for i in range(n_steps)
    ]
    return ReplayChain(archetype_id="arch-1", session_id=session_id, steps=steps)


@pytest.mark.asyncio
async def test_run_chain_carries_candidate_outputs_forward_not_frontier(tmp_path: Path) -> None:
    call_log: list[dict[str, object]] = []

    async def recording_call(candidate: Candidate, sample: SampleRow) -> CallResult:
        call_log.append(json.loads(sample.request_json))
        step_index = len(call_log) - 1
        return CallResult(
            response_text=f"candidate output {step_index}",
            tokens_in=100,
            tokens_out=50,
            latency_ms=1,
        )

    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, recording_call, config)
    chain = _chain("sess-a", 3)

    result = await runner.run_chain("run-9", chain, CANDIDATE, chain_budget_usd=100.0)

    assert result is not None
    assert result.verdict == "pass"
    assert result.n_steps == 3
    assert result.confidence == "chain"
    assert len(call_log) == 3

    assert call_log[0]["messages"] == [{"role": "user", "content": "original user turn 0"}]
    # step 1 carries the *candidate's* own output from step 0 forward, not
    # the original frontier answer — this is the E9 assertion (downstream
    # calls use candidate outputs, checked here via the fake-upstream
    # request log rather than a real fake HTTP server).
    assert call_log[1]["messages"] == [
        {"role": "user", "content": "original user turn 0"},
        {"role": "assistant", "content": "candidate output 0"},
        {"role": "user", "content": "original user turn 1"},
    ]
    assert "original frontier answer" not in json.dumps(call_log)


@pytest.mark.asyncio
async def test_run_chain_respects_per_chain_budget_sub_cap(tmp_path: Path) -> None:
    config = RunnerConfig(budget_usd=1000.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, _fixed_cost_call, config)
    chain = _chain("sess-b", 5)

    result = await runner.run_chain("run-10", chain, CANDIDATE, chain_budget_usd=3.0)

    assert result is not None
    # $2/call (per _fixed_cost_call), $3 sub-cap: step0 (spend=0<3) proceeds
    # -> spend=$2; step1 (spend=2<3) proceeds -> spend=$4; step2 sees
    # spend=$4>=3 and stops — the chain's own budget, independent of the
    # much larger overall run budget.
    assert result.n_steps == 2
    assert result.cost_usd == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_run_chain_grades_only_final_step(tmp_path: Path) -> None:
    graded: list[SampleRow] = []

    async def recording_grader(sample: SampleRow, result: CallResult) -> bool:
        graded.append(sample)
        return True

    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], recording_grader, _fixed_cost_call, config)
    chain = _chain("sess-c", 3)

    await runner.run_chain("run-11", chain, CANDIDATE, chain_budget_usd=100.0)

    assert len(graded) == 1  # grader called once, not once per step
    assert graded[0].trace_id == "sess-c-2"  # the final step


@pytest.mark.asyncio
async def test_run_chain_returns_none_if_first_call_fails(tmp_path: Path) -> None:
    async def always_fails(candidate: Candidate, sample: SampleRow) -> CallResult:
        raise RuntimeError("boom")

    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], _always_pass_grader, always_fails, config)
    chain = _chain("sess-d", 2)

    result = await runner.run_chain("run-12", chain, CANDIDATE, chain_budget_usd=100.0)
    assert result is None
