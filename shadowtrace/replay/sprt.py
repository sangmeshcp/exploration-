"""Wald's Sequential Probability Ratio Test, per (archetype, candidate)
(plan.md M3.5).

H1 ("pass"): true pass rate >= p0 (default: the per-archetype floor).
H0 ("fail"): true pass rate <= p1 (default: floor - 0.10).
alpha = P(accept H1 | H0 true) = false-positive rate (default 0.05).
beta  = P(accept H0 | H1 true) = false-negative rate (default 0.05).

Stop states: "pass" / "fail" (a boundary was crossed) or "inconclusive"
(budget/sample cap hit with neither boundary crossed — plan.md calls this
"budget-exhausted(inconclusive)").
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Verdict = str  # "pass" | "fail" | "inconclusive"


@dataclass(frozen=True)
class SPRTConfig:
    alpha: float = 0.05
    beta: float = 0.05
    p0: float = 0.90
    p1: float = 0.80

    def __post_init__(self) -> None:
        if not (0.0 < self.p1 < self.p0 < 1.0):
            raise ValueError("require 0 < p1 < p0 < 1")
        if not (0.0 < self.alpha < 1.0 and 0.0 < self.beta < 1.0):
            raise ValueError("alpha and beta must be in (0, 1)")

    @property
    def upper_bound(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower_bound(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def _llr_increment(self, passed: bool) -> float:
        if passed:
            return math.log(self.p0 / self.p1)
        return math.log((1 - self.p0) / (1 - self.p1))


@dataclass
class SPRTState:
    config: SPRTConfig
    n: int = 0
    n_pass: int = 0
    log_likelihood_ratio: float = 0.0
    verdict: Verdict | None = None

    def update(self, passed: bool) -> Verdict | None:
        if self.verdict is not None:
            return self.verdict
        self.n += 1
        if passed:
            self.n_pass += 1
        self.log_likelihood_ratio += self.config._llr_increment(passed)
        if self.log_likelihood_ratio >= self.config.upper_bound:
            self.verdict = "pass"
        elif self.log_likelihood_ratio <= self.config.lower_bound:
            self.verdict = "fail"
        return self.verdict


def run_sprt(
    outcomes: Iterable[bool],
    config: SPRTConfig | None = None,
    max_samples: int | None = None,
) -> SPRTState:
    config = config or SPRTConfig()
    state = SPRTState(config)
    for outcome in outcomes:
        state.update(outcome)
        if state.verdict is not None:
            return state
        if max_samples is not None and state.n >= max_samples:
            break
    if state.verdict is None:
        state.verdict = "inconclusive"
    return state
