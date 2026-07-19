"""Verifies ReplayRunner actually emits the span tree the plan describes:
`shadow_run` root -> per-call `replay_call` spans (model/tokens/cost
attrs) -> `grader` spans, and for chain replay `shadow_run_chain` ->
`chain_step` spans -> `grader` (plan.md §4.5: "one trace answers why did
this run cost $2.40 and stop early on archetype X").
"""

import json
from pathlib import Path

import pytest

from shadowtrace.common.tracing import build_tracer_provider
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.runner import CallResult, ReplayRunner, RunnerConfig
from shadowtrace.replay.sampler import ReplayChain, SampleRow

CANDIDATE = Candidate(
    name="claude-haiku-4-5", provider="anthropic", price_per_mtok_in=1.0, price_per_mtok_out=5.0
)


async def _call(candidate: Candidate, sample: SampleRow) -> CallResult:
    return CallResult(response_text="ok", tokens_in=100, tokens_out=50, latency_ms=2)


async def _grader(sample: SampleRow, result: CallResult) -> bool:
    return True


def _read_spans(traces_path: Path) -> list[dict[str, object]]:
    if not traces_path.exists():
        return []
    return [json.loads(line) for line in traces_path.read_text().strip().split("\n") if line]


@pytest.mark.asyncio
async def test_run_emits_shadow_run_root_span_with_replay_call_and_grader_children(
    tmp_path: Path,
) -> None:
    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], _grader, _call, config, tracer=tracer)
    samples = [
        SampleRow(
            trace_id="t1", archetype_id="arch-1", ts=1000, request_json="{}", response_json="{}"
        )
    ]
    await runner.run("run-trace-1", samples)

    spans = _read_spans(traces_path)
    names = [s["name"] for s in spans]
    assert names.count("shadow_run") == 1
    assert names.count("replay_call") == 1
    assert names.count("grader") == 1

    root = next(s for s in spans if s["name"] == "shadow_run")
    call = next(s for s in spans if s["name"] == "replay_call")
    grader = next(s for s in spans if s["name"] == "grader")

    # parent/child chain: grader -> replay_call -> shadow_run, all one trace
    assert call["parent_span_id"] == root["span_id"]
    assert grader["parent_span_id"] == call["span_id"]
    assert root["trace_id"] == call["trace_id"] == grader["trace_id"]

    assert root["attributes"]["run_id"] == "run-trace-1"
    assert root["attributes"]["stopped_reason"] == "completed"
    assert call["attributes"]["candidate"] == "claude-haiku-4-5"
    assert call["attributes"]["cost_usd"] > 0
    assert grader["attributes"]["verdict"] == "pass"


@pytest.mark.asyncio
async def test_run_chain_emits_chain_span_tree(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.jsonl"
    provider = build_tracer_provider(traces_path)
    tracer = provider.get_tracer("test")

    config = RunnerConfig(budget_usd=100.0, checkpoint_path=tmp_path / "cp.json")
    runner = ReplayRunner([CANDIDATE], _grader, _call, config, tracer=tracer)
    chain = ReplayChain(
        archetype_id="arch-1",
        session_id="sess-1",
        steps=[
            SampleRow(
                trace_id=f"t{i}",
                archetype_id="arch-1",
                ts=1000 + i,
                request_json=json.dumps({"role": "user", "content": f"turn {i}"}),
                response_json="{}",
                session_id="sess-1",
            )
            for i in range(2)
        ],
    )
    await runner.run_chain("run-trace-2", chain, CANDIDATE, chain_budget_usd=100.0)

    spans = _read_spans(traces_path)
    names = [s["name"] for s in spans]
    assert names.count("shadow_run_chain") == 1
    assert names.count("chain_step") == 2
    assert names.count("grader") == 1

    root = next(s for s in spans if s["name"] == "shadow_run_chain")
    assert root["attributes"]["session_id"] == "sess-1"
    assert root["attributes"]["n_steps_executed"] == 2
    steps = [s for s in spans if s["name"] == "chain_step"]
    assert all(s["parent_span_id"] == root["span_id"] for s in steps)
