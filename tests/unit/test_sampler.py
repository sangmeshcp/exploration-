from pathlib import Path
from typing import Any

from shadowtrace.db import duckdb_store
from shadowtrace.replay.sampler import sample_chains, sample_traces, top_archetypes_by_volume


def _seed(
    conn, n: int, archetype_id: str = "arch-1", quarantined_ids: set[int] | None = None
) -> None:  # type: ignore[no-untyped-def]
    quarantined_ids = quarantined_ids or set()
    now = 1_700_000_000_000
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?)",
        [archetype_id, "Test Archetype", False, now, now],
    )
    for i in range(n):
        trace_id = f"trace-{archetype_id}-{i}"
        conn.execute(
            "INSERT INTO traces (id, ts, source, model, request_json, response_json, quarantined) "
            "VALUES (?, ?, 'proxy', 'x', '{}', '{}', ?)",
            [trace_id, now + i, i in quarantined_ids],
        )
        conn.execute(
            "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
            "VALUES (?, ?, 0.1, ?)",
            [trace_id, archetype_id, now],
        )


def test_sample_traces_respects_n_per_archetype(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 20)
    samples = sample_traces(conn, n_per_archetype=5, seed=1)
    assert len(samples["arch-1"]) == 5


def test_sample_traces_excludes_quarantined(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 10, quarantined_ids={0, 1, 2})
    samples = sample_traces(conn, n_per_archetype=10, seed=1)
    assert len(samples["arch-1"]) == 7  # 10 - 3 quarantined


def test_sample_traces_deterministic_for_same_seed(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 30)
    a = sample_traces(conn, n_per_archetype=6, seed=42)
    b = sample_traces(conn, n_per_archetype=6, seed=42)
    assert [s.trace_id for s in a["arch-1"]] == [s.trace_id for s in b["arch-1"]]


def test_sample_traces_different_seeds_can_differ(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 30)
    a = sample_traces(conn, n_per_archetype=6, seed=1)
    b = sample_traces(conn, n_per_archetype=6, seed=2)
    assert [s.trace_id for s in a["arch-1"]] != [s.trace_id for s in b["arch-1"]]


def test_sample_traces_prioritizes_recency(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 10)
    samples = sample_traces(conn, n_per_archetype=3, seed=1)
    ids = {s.trace_id for s in samples["arch-1"]}
    # the most recent trace (highest suffix index) must always be included
    assert "trace-arch-1-9" in ids


def test_sample_traces_filters_by_archetype_id(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 5, archetype_id="arch-1")
    _seed(conn, 5, archetype_id="arch-2")
    samples = sample_traces(conn, n_per_archetype=10, seed=1, archetype_id="arch-1")
    assert set(samples.keys()) == {"arch-1"}


def test_sample_traces_excludes_merged_archetypes(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 5, archetype_id="arch-old")
    conn.execute("UPDATE archetypes SET merged_into = 'arch-new' WHERE id = 'arch-old'")
    samples = sample_traces(conn, n_per_archetype=10, seed=1)
    assert samples == {}


def _seed_sessions(
    conn: Any, archetype_id: str, sessions: dict[str, int], now: int = 1_700_000_000_000
) -> None:
    conn.execute(
        "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?)",
        [archetype_id, "Test Archetype", False, now, now],
    )
    for session_id, n_steps in sessions.items():
        for i in range(n_steps):
            trace_id = f"trace-{session_id}-{i}"
            conn.execute(
                "INSERT INTO traces (id, ts, source, model, request_json, response_json, "
                "quarantined, session_id) VALUES (?, ?, 'transcript', 'x', ?, '{}', FALSE, ?)",
                [trace_id, now + i, f'{{"role": "user", "content": "step {i}"}}', session_id],
            )
            conn.execute(
                "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
                "VALUES (?, ?, 0.1, ?)",
                [trace_id, archetype_id, now],
            )


def test_top_archetypes_by_volume(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed(conn, 10, archetype_id="big")
    _seed(conn, 3, archetype_id="small")
    _seed(conn, 6, archetype_id="medium")
    assert top_archetypes_by_volume(conn, n=2) == ["big", "medium"]


def test_sample_chains_groups_multi_step_sessions_in_order(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed_sessions(conn, "arch-1", {"sess-a": 3, "sess-b": 2})
    chains = sample_chains(conn, ["arch-1"], seed=1)
    assert len(chains) == 2
    sess_a = next(c for c in chains if c.session_id == "sess-a")
    assert [s.trace_id for s in sess_a.steps] == [
        "trace-sess-a-0",
        "trace-sess-a-1",
        "trace-sess-a-2",
    ]


def test_sample_chains_excludes_single_step_sessions(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed_sessions(conn, "arch-1", {"sess-solo": 1, "sess-multi": 3})
    chains = sample_chains(conn, ["arch-1"], seed=1)
    assert [c.session_id for c in chains] == ["sess-multi"]


def test_sample_chains_respects_max_chains_and_steps(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    _seed_sessions(conn, "arch-1", {f"sess-{i}": 4 for i in range(5)})
    chains = sample_chains(
        conn, ["arch-1"], max_chains_per_archetype=2, max_steps_per_chain=2, seed=1
    )
    assert len(chains) == 2
    assert all(len(c.steps) == 2 for c in chains)
