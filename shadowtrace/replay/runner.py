"""Budget-capped shadow replay executor: `shadow run --budget 3.00
[--archetype X]` (plan.md M3.3).

Concurrency note: budget enforcement is exact ("stops within one call of
the cap") at concurrency=1, since the spend check and the call happen
serially. At concurrency>1, up to `concurrency` calls may already be
in-flight when the cap is first reached, so the guarantee weakens to
"stops within `concurrency` calls of the cap" — documented here rather
than hidden, since a caller setting a tight budget should know which
regime they're in.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid
from shadowtrace.replay.ladder import Candidate
from shadowtrace.replay.sampler import SampleRow

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
    ) -> None:
        self.ladder = ladder
        self.grader = grader
        self.call_fn = call_fn
        self.config = config

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
            try:
                call_result = await self._call_with_backoff(candidate, sample)
            except Exception:
                logger.exception(
                    "replay call failed for %s/%s after retries", sample.trace_id, candidate.name
                )
                return None
            passed = await self.grader(sample, call_result)
            cost = candidate.cost_usd(call_result.tokens_in, call_result.tokens_out)
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
                REGISTRY.inc("replay_spend_usd", value=cost, labels={"candidate": candidate.name})
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

        return RunResult(
            results=results, spend_usd=checkpoint.spend_usd, stopped_reason=stopped_reason
        )
