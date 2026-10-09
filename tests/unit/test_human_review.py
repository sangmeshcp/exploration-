from pathlib import Path

import pytest

from shadowtrace.db import duckdb_store
from shadowtrace.replay.graders import human_review
from tests.fakes.analytics import NOW, add_archetype, add_replay_result, add_trace


@pytest.fixture
def conn(tmp_path: Path):  # type: ignore[no-untyped-def]
    c = duckdb_store.connect(tmp_path / "a.duckdb")
    add_archetype(c, "a1", "Code")
    add_trace(c, "t1", NOW, "a1", request="what port?", response="8080")
    return c


def test_enqueue_picks_only_judge_decided_results_and_caps_per_archetype(conn) -> None:  # type: ignore[no-untyped-def]
    add_replay_result(conn, "decided", "a1", "c", True, NOW, "t1", {"format": True, "judge": True})
    add_replay_result(
        conn, "det_fail", "a1", "c", False, NOW, "t1", {"format": False, "judge": True}
    )
    add_replay_result(conn, "no_judge", "a1", "c", True, NOW, "t1", {"completion": True})
    add_replay_result(conn, "chain", "a1", "c", True, NOW, "t1", {"judge": True}, grader="chain")
    for i in range(5):
        add_replay_result(conn, f"more{i}", "a1", "c", True, NOW - i, "t1", {"judge": True})

    assert human_review.enqueue(conn, per_archetype=3) == 3
    queued = {r[0] for r in conn.execute("SELECT replay_result_id FROM human_reviews").fetchall()}
    assert "decided" in queued
    assert not queued & {"det_fail", "no_judge", "chain"}
    assert human_review.enqueue(conn, per_archetype=3) == 0  # cap already reached

    item = human_review.pending(conn, limit=1)[0]
    assert item.archetype_label == "Code"
    assert item.request == "what port?"
    assert item.frontier_answer == "8080"
    assert item.candidate_answer == "candidate answer"


def test_record_and_agreement_stats(conn) -> None:  # type: ignore[no-untyped-def]
    # judge verdict, human verdict
    pairs = [(True, True), (True, True), (True, False), (False, False), (False, True)]
    for i, (judge_pass, _) in enumerate(pairs):
        add_replay_result(conn, f"r{i}", "a1", "c", judge_pass, NOW, "t1", {"judge": judge_pass})
    human_review.enqueue(conn, per_archetype=10)
    for i, (_, human_pass) in enumerate(pairs):
        human_review.record(conn, f"r{i}", passed=human_pass, note="checked")

    stats = human_review.agreement(conn)
    assert stats.n == 5
    assert stats.agreement == pytest.approx(3 / 5)
    assert stats.false_pass_rate == pytest.approx(1 / 3)
    assert stats.false_fail_rate == pytest.approx(1 / 2)
    assert stats.cohen_kappa == pytest.approx(human_review.cohen_kappa(pairs))
    assert human_review.pending(conn) == []
    assert human_review.agreement(conn, archetype_id="other").n == 0

    with pytest.raises(KeyError):
        human_review.record(conn, "not-queued", passed=True)


def test_cohen_kappa_reference_values() -> None:
    assert human_review.cohen_kappa([]) is None
    perfect = [(True, True), (False, False)]
    assert human_review.cohen_kappa(perfect) == pytest.approx(1.0)
    assert human_review.cohen_kappa([(True, True), (True, True)]) == pytest.approx(1.0)
    # judge always passes, human splits evenly: agreement is pure chance
    assert human_review.cohen_kappa([(True, True), (True, False)]) == pytest.approx(0.0)
    # classic 2x2: a=20 b=5 c=10 d=15 -> po=0.7, pe=0.5, kappa=0.4
    table = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert human_review.cohen_kappa(table) == pytest.approx(0.4)
