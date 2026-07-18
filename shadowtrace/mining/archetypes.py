"""Archetype persistence + CRUD (plan.md M2.3).

Cluster labels from mining/cluster.py are just integers that can shuffle
between runs — this module is what makes archetype *identity* stable
across re-clustering: each new cluster's medoid embedding is matched
against existing archetypes' stored medoid (by looking up that trace's
embedding in the current batch); a close-enough match reuses the old
archetype id (and survives any rename/merge/pin done on it), otherwise a
new archetype is created.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from shadowtrace.common.logging import get_logger
from shadowtrace.common.metrics import REGISTRY
from shadowtrace.common.ulid import new_ulid
from shadowtrace.mining.cluster import LabelFn, keyword_label, medoid_index
from shadowtrace.mining.embed import cosine_similarity

logger = get_logger("mining.archetypes")

STABLE_MATCH_SIMILARITY_THRESHOLD = 0.75


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ArchetypeAssignmentResult:
    trace_to_archetype: dict[str, str]
    new_archetype_ids: list[str]
    matched_archetype_ids: list[str]


def assign_archetypes(
    conn: Any,
    trace_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    labels: list[int],
    label_fn: LabelFn | None = None,
) -> ArchetypeAssignmentResult:
    now = _now_ms()
    trace_id_to_vec = dict(zip(trace_ids, embeddings, strict=True))

    clusters: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(i)

    existing = conn.execute(
        "SELECT id, medoid_trace_id FROM archetypes WHERE merged_into IS NULL"
    ).fetchall()
    used_existing_ids: set[str] = set()

    result = ArchetypeAssignmentResult({}, [], [])

    for _lbl, idxs in clusters.items():
        cluster_vectors = [embeddings[i] for i in idxs]
        m_idx = medoid_index(cluster_vectors)
        medoid_trace_id = trace_ids[idxs[m_idx]]
        medoid_vec = cluster_vectors[m_idx]

        matched_id: str | None = None
        best_sim = -1.0
        for arch_id, arch_medoid_trace_id in existing:
            if arch_id in used_existing_ids:
                continue
            old_vec = trace_id_to_vec.get(arch_medoid_trace_id)
            if old_vec is None:
                continue
            sim = cosine_similarity(medoid_vec, old_vec)
            if sim >= STABLE_MATCH_SIMILARITY_THRESHOLD and sim > best_sim:
                matched_id, best_sim = arch_id, sim

        if matched_id is not None:
            archetype_id = matched_id
            used_existing_ids.add(matched_id)
            conn.execute(
                "UPDATE archetypes SET medoid_trace_id = ?, updated_ts = ? WHERE id = ?",
                [medoid_trace_id, now, archetype_id],
            )
            result.matched_archetype_ids.append(archetype_id)
        else:
            archetype_id = new_ulid()
            exemplar_texts = [texts[i] for i in idxs[:5]]
            label = keyword_label(exemplar_texts) if label_fn is None else label_fn(exemplar_texts)
            conn.execute(
                "INSERT INTO archetypes (id, label, pinned, created_ts, updated_ts, medoid_trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [archetype_id, label, False, now, now, medoid_trace_id],
            )
            result.new_archetype_ids.append(archetype_id)

        for i in idxs:
            trace_id = trace_ids[i]
            distance = 1.0 - cosine_similarity(embeddings[i], medoid_vec)
            conn.execute(
                "INSERT INTO archetype_assignments (trace_id, archetype_id, distance, assigned_ts) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (trace_id) DO UPDATE SET "
                "archetype_id = excluded.archetype_id, distance = excluded.distance, "
                "assigned_ts = excluded.assigned_ts",
                [trace_id, archetype_id, distance, now],
            )
            result.trace_to_archetype[trace_id] = archetype_id

    unassigned = len(trace_ids) - len(result.trace_to_archetype)
    if trace_ids:
        REGISTRY.set_gauge("unassigned_traces_ratio", unassigned / len(trace_ids))
    active_count = conn.execute(
        "SELECT count(*) FROM archetypes WHERE merged_into IS NULL"
    ).fetchone()[0]
    REGISTRY.set_gauge("archetype_count", active_count)

    return result


def rename_archetype(conn: Any, archetype_id: str, new_label: str) -> None:
    conn.execute(
        "UPDATE archetypes SET label = ?, updated_ts = ? WHERE id = ?",
        [new_label, _now_ms(), archetype_id],
    )


def set_pinned(conn: Any, archetype_id: str, pinned: bool) -> None:
    conn.execute(
        "UPDATE archetypes SET pinned = ?, updated_ts = ? WHERE id = ?",
        [pinned, _now_ms(), archetype_id],
    )


def merge_archetypes(conn: Any, source_id: str, target_id: str) -> None:
    """Merge `source_id` into `target_id`. `source_id` is kept as a row
    (with `merged_into` set) for audit rather than deleted, and future
    re-clusters will never match against it again since assign_archetypes
    only considers `merged_into IS NULL` rows as match candidates."""
    if source_id == target_id:
        return
    now = _now_ms()
    conn.execute(
        "UPDATE archetype_assignments SET archetype_id = ?, assigned_ts = ? WHERE archetype_id = ?",
        [target_id, now, source_id],
    )
    conn.execute(
        "UPDATE archetypes SET merged_into = ?, updated_ts = ? WHERE id = ?",
        [target_id, now, source_id],
    )
    logger.info("merged archetype %s into %s", source_id, target_id)
