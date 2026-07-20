"""HDBSCAN clustering + auto-labeling (plan.md M2.2).

Same two-backend pattern as embed.py: real HDBSCAN when the `ml` extra is
installed, otherwise a dependency-free union-find fallback over a cosine
similarity threshold. Cluster labels use HDBSCAN's convention: -1 means
"noise" / unassigned.

Auto-labeling defaults to a fully-local keyword extractor (the
"fully-local labeling" feature flag from the plan) rather than a live LLM
call, so clustering never needs network access or an API key. A real
labeler (one cheap Haiku call over k exemplars) can be plugged in via
`label_fn` without changing call sites.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from shadowtrace.common.logging import get_logger
from shadowtrace.mining.embed import cosine_similarity

logger = get_logger("mining.cluster")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "please",
    "can",
    "you",
    "i",
    "me",
    "my",
    "this",
    "that",
    "with",
    "it",
    "be",
    "how",
    "do",
    "does",
    "what",
    "why",
    "we",
    "us",
    "your",
    "our",
}

DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.5


def _union_find_cluster(
    vectors: list[list[float]],
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[int]:
    n = len(vectors)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(vectors[i], vectors[j]) >= similarity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    labels = [-1] * n
    next_label = 0
    for members in groups.values():
        if len(members) >= min_cluster_size:
            for m in members:
                labels[m] = next_label
            next_label += 1
    return labels


def _hdbscan_cluster(vectors: list[list[float]], min_cluster_size: int) -> list[int]:
    import hdbscan
    import numpy as np

    arr = np.array(vectors)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(arr)
    return [int(x) for x in labels]


def cluster_embeddings(
    vectors: list[list[float]],
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[int]:
    """Returns one cluster label per input vector. -1 = unassigned/noise."""
    if not vectors:
        return []
    if len(vectors) <= min_cluster_size:
        # Real HDBSCAN's kd-tree query needs at least min_samples + 1
        # (= min_cluster_size + 1 by default) training points and raises
        # ValueError below that — hit in practice when a capture store has
        # many traces but only a handful with usable prompt text. The
        # union-find fallback handles arbitrarily small inputs, so use it.
        return _union_find_cluster(vectors, min_cluster_size, similarity_threshold)
    try:
        return _hdbscan_cluster(vectors, min_cluster_size)
    except ImportError:
        logger.warning(
            "hdbscan not installed; using local cosine-threshold clustering fallback "
            "(pip install '.[ml]' for HDBSCAN)"
        )
        return _union_find_cluster(vectors, min_cluster_size, similarity_threshold)


def keyword_label(exemplars: list[str], top_k: int = 3) -> str:
    counts: Counter[str] = Counter()
    for text in exemplars:
        for token in _TOKEN_RE.findall(text.lower()):
            if len(token) > 2 and token not in _STOPWORDS:
                counts[token] += 1
    if not counts:
        return "Unlabeled"
    top = [w for w, _ in counts.most_common(top_k)]
    return " ".join(w.capitalize() for w in top)


LabelFn = Callable[[list[str]], str]


def label_cluster(exemplars: list[str], label_fn: LabelFn | None = None) -> str:
    fn = label_fn or keyword_label
    return fn(exemplars)


def medoid_index(vectors: list[list[float]]) -> int:
    """Index of the vector with the highest average similarity to the rest
    of the cluster — used both for labeling exemplars and for archetype
    identity matching across re-clusters (mining/archetypes.py)."""
    if len(vectors) == 1:
        return 0
    best_idx, best_score = 0, -1.0
    for i, v in enumerate(vectors):
        score = sum(cosine_similarity(v, other) for j, other in enumerate(vectors) if j != i)
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx
