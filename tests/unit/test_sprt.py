import random

import pytest

from shadowtrace.replay.sprt import SPRTConfig, run_sprt


def test_config_rejects_invalid_p_ordering() -> None:
    with pytest.raises(ValueError):
        SPRTConfig(p0=0.5, p1=0.9)


def test_two_passes_crosses_upper_bound_exactly() -> None:
    # p0=0.9, p1=0.1 -> per-pass LLR = ln(9) ≈ 2.197; bound = ln(0.95/0.05) ≈ 2.944.
    # One pass (2.197) doesn't cross; two passes (4.394) does.
    config = SPRTConfig(alpha=0.05, beta=0.05, p0=0.9, p1=0.1)
    state = run_sprt([True], config, max_samples=1)
    assert state.verdict == "inconclusive"
    state2 = run_sprt([True, True], config)
    assert state2.verdict == "pass"
    assert state2.n == 2


def test_two_fails_crosses_lower_bound_exactly() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.05, p0=0.9, p1=0.1)
    state = run_sprt([False, False], config)
    assert state.verdict == "fail"
    assert state.n == 2


def test_alternating_outcomes_with_symmetric_p_stays_inconclusive() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.05, p0=0.9, p1=0.1)
    state = run_sprt([True, False, True, False], config, max_samples=4)
    assert state.verdict == "inconclusive"
    assert state.n == 4


def test_verdict_is_sticky_after_decision() -> None:
    config = SPRTConfig(alpha=0.05, beta=0.05, p0=0.9, p1=0.1)
    state = run_sprt([True, True, True, True], config)
    assert state.verdict == "pass"
    assert state.n == 2  # stopped consuming after the boundary was crossed


def test_empirical_error_rates_within_theoretical_bounds() -> None:
    """SPRT's headline guarantee: for a truly-bad candidate (rate=p1), the
    probability of an incorrect "pass" verdict is <= alpha; for a
    truly-good candidate (rate=p0), the probability of an incorrect "fail"
    verdict is <= beta. Verify empirically over many simulated runs."""
    rng = random.Random(1234)
    config = SPRTConfig(alpha=0.05, beta=0.05, p0=0.90, p1=0.80)
    trials = 4000
    tolerance = 0.03  # binomial noise budget at n=4000

    false_pass = 0
    for _ in range(trials):
        outcomes = (rng.random() < config.p1 for _ in range(2000))
        state = run_sprt(outcomes, config, max_samples=2000)
        if state.verdict == "pass":
            false_pass += 1
    assert false_pass / trials <= config.alpha + tolerance

    false_fail = 0
    for _ in range(trials):
        outcomes = (rng.random() < config.p0 for _ in range(2000))
        state = run_sprt(outcomes, config, max_samples=2000)
        if state.verdict == "fail":
            false_fail += 1
    assert false_fail / trials <= config.beta + tolerance
