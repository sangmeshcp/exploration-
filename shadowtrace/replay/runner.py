"""Budget-capped shadow replay executor: `shadow run --budget 3.00
[--archetype X]` (plan.md M3.3).

Concurrency note: budget enforcement is exact ("stops within one call of
the cap") at concurrency=1, since the spend check and the call happen
serially. At concurrency>1, up to `concurrency` calls may already be
in-flight when the cap is first reached, so the guarantee weakens to
"stops within `concurrency` calls of the cap" — documented here rather
than hidden, since a caller setting a tight budget should know which
regime they're in.

`run()` replays each sampled trace as an independent single call — fine
for one-shot archetypes, but for agentic/multi-turn archetypes replaying
each step in isolation against the *original* frontier context is a lie:
a real agentic replay needs the candidate's own prior turns feeding
forward, since that's what actually happens when you switch a live agent
loop to a different model. `run_chain()` (plan.md M3.6) does that: it
re-executes an ordered session step by step, substituting the candidate's
own previous responses for what the frontier originally said at each
prior turn, and grades only the final step. Restricted by the caller to
the top-2 archetypes by volume (review R6 — chain cost multiplies with
chain length, so this is opt-in, not automatic) and bounded by its own
`chain_budget_usd` sub-cap, independent of `run()`'s single-call budget.
Unlike `run()`, chain replay is not checkpointed/resumable — a single
chain is short-lived and re-running it from scratch is cheap enough that
resumability wasn't worth the added complexity here.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.tracing import get_default_tracer
from shadowtrace.common.ulid import new_ulid
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.sampler import ReplayChain, SampleRow

logger = get_logger("replay.runner")


class RateLimitError(Exception):
    """Raised by a CallFn to signal a provider rate limit; the runner will
    back off and retry rather than treating it as a hard failure."""


@dataclass
class CallResult:
    response_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class CallFn(Protocol):
    async def __call__(self, candidate: Candidate, sample: SampleRow) -> CallResult: ...


class GraderFn(Protocol):
    async def __call__(self, sample: SampleRow, result: CallResult) -> bool: ...


@dataclass
class ReplayResult:
    id: str
    archetype_id: str
    candidate: str
    source_trace_id: str
    verdict: str  # "pass" | "fail"
    cost_usd: float
    latency_ms: int
    grader: str
    run_id: str
    ts: int


@dataclass
class ChainReplayResult:
    id: str
    archetype_id: str
    session_id: str
    candidate: str
    verdict: str  # "pass" | "fail"
    cost_usd: float
    n_steps: int  # steps actually executed — may be < len(chain.steps) if the sub-cap was hit
    run_id: str
    ts: int
    confidence: str = "chain"  # vs. "single_call" elsewhere — see module docstring


def _extract_user_content(request_json_raw: str) -> Any:
    try:
        data = json.loads(request_json_raw)
    except (json.JSONDecodeError, TypeError):
        return request_json_raw
    if isinstance(data, dict):
        return data.get("content", data)
    return data


@dataclass
class Checkpoint:
    run_id: str
    spend_usd: float = 0.0
    completed: set[tuple[str, str]] = field(default_factory=set)  # (trace_id, candidate)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "spend_usd": self.spend_usd,
                "completed": [list(k) for k in self.completed],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Checkpoint:
        data = json.loads(text)
        return cls(
            run_id=data["run_id"],
            spend_usd=data["spend_usd"],
            completed={tuple(k) for k in data["completed"]},
        )


@dataclass
class RunnerConfig:
    budget_usd: float
    checkpoint_path: Path
    concurrency: int = 1
    max_retries: int = 3
    backoff_base_s: float = 0.1
    grader_name: str = "judge"


@dataclass
class RunResult:
    results: list[ReplayResult]
    spend_usd: float
    stopped_reason: str  # "completed" | "budget_exhausted"


class ReplayRunner:
    def __init__(
        self,
        ladder: list[Candidate],
        grader: GraderFn,
        call_fn: CallFn,
        config: RunnerConfig,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.ladder = ladder
        self.grader = grader
        self.call_fn = call_fn
        self.config = config
        self.tracer = tracer or get_default_tracer()

    def _load_checkpoint(self, run_id: str) -> Checkpoint:
        path = self.config.checkpoint_path
        if path.exists():
            try:
                cp = Checkpoint.from_json(path.read_text())
                if cp.run_id == run_id:
                    return cp
            except (json.JSONDecodeError, KeyError):
                logger.warning("ignoring corrupt checkpoint at %s", path)
        return Checkpoint(run_id=run_id)

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_path.write_text(checkpoint.to_json())
        REGISTRY.set_gauge("checkpoint_age_seconds", 0.0)

    async def _call_with_backoff(self, candidate: Candidate, sample: SampleRow) -> CallResult:
        attempt = 0
        while True:
            try:
                return await self.call_fn(candidate, sample)
            except RateLimitError:
                attempt += 1
                REGISTRY.inc(
                    "provider_errors_total", labels={"provider": candidate.provider, "code": "429"}
                )
                if attempt > self.config.max_retries:
                    raise
                await asyncio.sleep(self.config.backoff_base_s * (2 ** (attempt - 1)))

    async def run(self, run_id: str, samples: list[SampleRow]) -> RunResult:
        with self.tracer.start_as_current_span(
            "shadow_run",
            attributes={
                "run_id": run_id,
                "budget_usd": self.config.budget_usd,
                "n_samples": len(samples),
            },
        ) as root_span:
            checkpoint = self._load_checkpoint(run_id)
            results: list[ReplayResult] = []
            stopped_reason = "completed"
            lock = asyncio.Lock()

            work_items = [
                (sample, candidate)
                for sample in samples
                for candidate in self.ladder
                if (sample.trace_id, candidate.name) not in checkpoint.completed
            ]

            async def process(sample: SampleRow, candidate: Candidate) -> ReplayResult | None:
                nonlocal stopped_reason
                async with lock:
                    if checkpoint.spend_usd >= self.config.budget_usd:
                        stopped_reason = "budget_exhausted"
                        return None
                with self.tracer.start_as_current_span(
                    "replay_call",
                    attributes={
                        "candidate": candidate.name,
                        "archetype_id": sample.archetype_id,
                        "trace_id": sample.trace_id,
                    },
                ) as call_span:
                    try:
                        call_result = await self._call_with_backoff(candidate, sample)
                    except Exception:
                        logger.exception(
                            "replay call failed for %s/%s after retries",
                            sample.trace_id,
                            candidate.name,
                        )
                        call_span.set_status(Status(StatusCode.ERROR))
                        return None
                    call_span.set_attribute("tokens_in", call_result.tokens_in)
                    call_span.set_attribute("tokens_out", call_result.tokens_out)

                    with self.tracer.start_as_current_span("grader") as grader_span:
                        passed = await self.grader(sample, call_result)
                        grader_span.set_attribute("verdict", "pass" if passed else "fail")

                    cost = candidate.cost_usd(call_result.tokens_in, call_result.tokens_out)
                    call_span.set_attribute("cost_usd", cost)
                    result = ReplayResult(
                        id=new_ulid(),
                        archetype_id=sample.archetype_id,
                        candidate=candidate.name,
                        source_trace_id=sample.trace_id,
                        verdict="pass" if passed else "fail",
                        cost_usd=cost,
                        latency_ms=call_result.latency_ms,
                        grader=self.config.grader_name,
                        run_id=run_id,
                        ts=int(time.time() * 1000),
                    )
                    async with lock:
                        checkpoint.spend_usd += cost
                        checkpoint.completed.add((sample.trace_id, candidate.name))
                        self._save_checkpoint(checkpoint)
                        REGISTRY.inc(
                            "replay_spend_usd", value=cost, labels={"candidate": candidate.name}
                        )
                        REGISTRY.inc(
                            "replay_calls_total",
                            labels={"candidate": candidate.name, "archetype": sample.archetype_id},
                        )
                    return result

            # Work is dispatched in batches of `concurrency`. At concurrency=1
            # each batch is a single call, giving the exact "stop within one
            # call of cap" bound; larger batches trade that precision for
            # throughput, per the module docstring.
            i = 0
            while i < len(work_items) and stopped_reason != "budget_exhausted":
                batch = work_items[i : i + self.config.concurrency]
                batch_results = await asyncio.gather(*(process(s, c) for s, c in batch))
                results.extend(r for r in batch_results if r is not None)
                i += len(batch)

            root_span.set_attribute("spend_usd", checkpoint.spend_usd)
            root_span.set_attribute("stopped_reason", stopped_reason)
            root_span.set_attribute("n_results", len(results))
            return RunResult(
                results=results, spend_usd=checkpoint.spend_usd, stopped_reason=stopped_reason
            )

    async def run_chain(
        self,
        run_id: str,
        chain: ReplayChain,
        candidate: Candidate,
        chain_budget_usd: float,
    ) -> ChainReplayResult | None:
        with self.tracer.start_as_current_span(
            "shadow_run_chain",
            attributes={
                "run_id": run_id,
                "session_id": chain.session_id,
                "archetype_id": chain.archetype_id,
                "candidate": candidate.name,
                "chain_budget_usd": chain_budget_usd,
                "n_steps_available": len(chain.steps),
            },
        ) as root_span:
            chain_spend = 0.0
            executed_steps = 0
            carried_history: list[dict[str, Any]] = []
            last_result: CallResult | None = None
            last_step: SampleRow | None = None

            for step_index, step in enumerate(chain.steps):
                if chain_spend >= chain_budget_usd:
                    break
                carried_history.append(
                    {"role": "user", "content": _extract_user_content(step.request_json)}
                )
                synthetic_sample = SampleRow(
                    trace_id=step.trace_id,
                    archetype_id=step.archetype_id,
                    ts=step.ts,
                    request_json=json.dumps({"messages": list(carried_history)}),
                    response_json=step.response_json,
                    session_id=step.session_id,
                )
                with self.tracer.start_as_current_span(
                    "chain_step", attributes={"step_index": step_index, "trace_id": step.trace_id}
                ) as step_span:
                    try:
                        call_result = await self._call_with_backoff(candidate, synthetic_sample)
                    except Exception:
                        logger.exception(
                            "chain replay call failed session=%s step=%s",
                            chain.session_id,
                            step.trace_id,
                        )
                        step_span.set_status(Status(StatusCode.ERROR))
                        break
                    step_cost = candidate.cost_usd(call_result.tokens_in, call_result.tokens_out)
                    step_span.set_attribute("cost_usd", step_cost)
                    chain_spend += step_cost
                    carried_history.append(
                        {"role": "assistant", "content": call_result.response_text}
                    )
                    last_result = call_result
                    last_step = step
                    executed_steps += 1

            if last_result is None or last_step is None:
                root_span.set_status(Status(StatusCode.ERROR))
                return None

            final_sample = SampleRow(
                trace_id=last_step.trace_id,
                archetype_id=last_step.archetype_id,
                ts=last_step.ts,
                request_json=json.dumps({"messages": carried_history}),
                response_json=last_step.response_json,
                session_id=last_step.session_id,
            )
            with self.tracer.start_as_current_span("grader") as grader_span:
                passed = await self.grader(final_sample, last_result)
                grader_span.set_attribute("verdict", "pass" if passed else "fail")

            REGISTRY.inc(
                "replay_spend_usd", value=chain_spend, labels={"candidate": candidate.name}
            )
            REGISTRY.inc(
                "replay_calls_total",
                labels={"candidate": candidate.name, "archetype": chain.archetype_id},
            )

            root_span.set_attribute("spend_usd", chain_spend)
            root_span.set_attribute("n_steps_executed", executed_steps)
            root_span.set_attribute("verdict", "pass" if passed else "fail")

            return ChainReplayResult(
                id=new_ulid(),
                archetype_id=chain.archetype_id,
                session_id=chain.session_id,
                candidate=candidate.name,
                verdict="pass" if passed else "fail",
                cost_usd=chain_spend,
                n_steps=executed_steps,
                run_id=run_id,
                ts=int(time.time() * 1000),
            )
