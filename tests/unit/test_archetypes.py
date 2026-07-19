from pathlib import Path

from shadowtrace.db import duckdb_store
from shadowtrace.mining import archetypes
from shadowtrace.mining.cluster import cluster_embeddings
from shadowtrace.mining.embed import HashingEmbedder

CODE_QUESTIONS = [
    "how do I write a for loop in python",
    "python for loop syntax example please",
    "write a for loop in python for me",
]
WEATHER_QUESTIONS = [
    "what is the weather forecast for tomorrow in paris",
    "will it rain tomorrow in paris",
    "paris weather forecast tomorrow please",
]


def _cluster_all(texts: list[str]) -> tuple[list[str], list[list[float]], list[int]]:
    trace_ids = [f"trace-{i}" for i in range(len(texts))]
    vectors = HashingEmbedder(dims=256).embed(texts)
    labels = cluster_embeddings(vectors, min_cluster_size=2, similarity_threshold=0.35)
    return trace_ids, vectors, labels


def test_assign_archetypes_creates_new_archetypes(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    texts = CODE_QUESTIONS + WEATHER_QUESTIONS
    trace_ids, vectors, labels = _cluster_all(texts)

    result = archetypes.assign_archetypes(conn, trace_ids, texts, vectors, labels)

    assert len(result.new_archetype_ids) == 2
    assert len(result.trace_to_archetype) == 6
    rows = conn.execute("SELECT count(*) FROM archetypes").fetchone()[0]
    assert rows == 2


def test_reclustering_reuses_stable_archetype_ids(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    texts = CODE_QUESTIONS + WEATHER_QUESTIONS
    trace_ids, vectors, labels = _cluster_all(texts)
    first = archetypes.assign_archetypes(conn, trace_ids, texts, vectors, labels)
    first_ids = set(first.trace_to_archetype.values())

    # re-cluster the same corpus (as a nightly re-cluster job would)
    trace_ids2, vectors2, labels2 = _cluster_all(texts)
    second = archetypes.assign_archetypes(conn, trace_ids2, texts, vectors2, labels2)

    assert len(second.new_archetype_ids) == 0
    assert len(second.matched_archetype_ids) == 2
    assert set(second.trace_to_archetype.values()) == first_ids
    total_archetypes = conn.execute("SELECT count(*) FROM archetypes").fetchone()[0]
    assert total_archetypes == 2  # no duplicates created


def test_rename_persists(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    trace_ids, vectors, labels = _cluster_all(CODE_QUESTIONS + WEATHER_QUESTIONS)
    result = archetypes.assign_archetypes(
        conn, trace_ids, CODE_QUESTIONS + WEATHER_QUESTIONS, vectors, labels
    )
    archetype_id = result.new_archetype_ids[0]

    archetypes.rename_archetype(conn, archetype_id, "Python Loops")
    label = conn.execute("SELECT label FROM archetypes WHERE id = ?", [archetype_id]).fetchone()[0]
    assert label == "Python Loops"


def test_merge_reassigns_trace_assignments_and_survives_rename(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    texts = CODE_QUESTIONS + WEATHER_QUESTIONS
    trace_ids, vectors, labels = _cluster_all(texts)
    result = archetypes.assign_archetypes(conn, trace_ids, texts, vectors, labels)
    source_id, target_id = result.new_archetype_ids

    archetypes.merge_archetypes(conn, source_id, target_id)

    remaining = conn.execute("SELECT DISTINCT archetype_id FROM archetype_assignments").fetchall()
    assert {r[0] for r in remaining} == {target_id}
    merged_into = conn.execute(
        "SELECT merged_into FROM archetypes WHERE id = ?", [source_id]
    ).fetchone()[0]
    assert merged_into == target_id


def test_merge_is_a_noop_for_self_merge(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    trace_ids, vectors, labels = _cluster_all(CODE_QUESTIONS)
    result = archetypes.assign_archetypes(conn, trace_ids, CODE_QUESTIONS, vectors, [0, 0, 0])
    archetype_id = result.new_archetype_ids[0]
    archetypes.merge_archetypes(conn, archetype_id, archetype_id)  # must not raise or corrupt
    merged_into = conn.execute(
        "SELECT merged_into FROM archetypes WHERE id = ?", [archetype_id]
    ).fetchone()[0]
    assert merged_into is None


def test_set_pinned(tmp_path: Path) -> None:
    conn = duckdb_store.connect(tmp_path / "a.duckdb")
    trace_ids, vectors, labels = _cluster_all(CODE_QUESTIONS)
    result = archetypes.assign_archetypes(conn, trace_ids, CODE_QUESTIONS, vectors, [0, 0, 0])
    archetype_id = result.new_archetype_ids[0]

    archetypes.set_pinned(conn, archetype_id, True)
    pinned = conn.execute("SELECT pinned FROM archetypes WHERE id = ?", [archetype_id]).fetchone()[
        0
    ]
    assert pinned is True
