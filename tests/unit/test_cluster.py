import pytest

from shadowtrace.mining.cluster import (
    _hdbscan_cluster,
    cluster_embeddings,
    keyword_label,
    medoid_index,
)
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

try:
    import hdbscan as _hdbscan_pkg  # noqa: F401

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def test_cluster_embeddings_groups_similar_topics() -> None:
    embedder = HashingEmbedder(dims=256)
    texts = CODE_QUESTIONS + WEATHER_QUESTIONS
    vectors = embedder.embed(texts)

    labels = cluster_embeddings(vectors, min_cluster_size=2, similarity_threshold=0.35)

    assert len(labels) == len(texts)
    code_labels = set(labels[:3])
    weather_labels = set(labels[3:])
    # each topic forms its own cluster, and the two topics never share one
    assert len(code_labels) == 1 and -1 not in code_labels
    assert len(weather_labels) == 1 and -1 not in weather_labels
    assert code_labels != weather_labels


def test_cluster_embeddings_empty_input() -> None:
    assert cluster_embeddings([]) == []


def test_keyword_label_picks_significant_words() -> None:
    label = keyword_label(CODE_QUESTIONS)
    assert "Loop" in label or "Python" in label


def test_keyword_label_empty_returns_placeholder() -> None:
    assert keyword_label([]) == "Unlabeled"


def test_medoid_index_single_vector() -> None:
    assert medoid_index([[1.0, 0.0]]) == 0


def test_medoid_index_picks_most_central() -> None:
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    idx = medoid_index(vectors)
    assert idx in (0, 1)  # the two similar ones are more central than the outlier


@pytest.mark.skipif(not HAS_HDBSCAN, reason="requires the `ml` extra (pip install '.[ml]')")
def test_real_hdbscan_backend_separates_topics() -> None:
    """Exercises the actual HDBSCAN backend (not the union-find fallback)
    end to end — skipped unless `hdbscan` is installed, since it's not a
    core dependency (see README's ml-extra fallback design)."""
    topics = {
        "code": [
            "how do I write a for loop in python",
            "python for loop syntax example",
            "write a for loop in python",
            "for loop python help",
            "python loop syntax question",
        ],
        "weather": [
            "what is the weather in paris tomorrow",
            "will it rain in paris this weekend",
            "paris weather forecast",
            "is it going to rain in paris",
            "weather forecast paris tomorrow",
        ],
        "sql": [
            "write a sql query to find duplicates",
            "sql query for duplicate rows",
            "how to find duplicate rows in sql",
            "sql duplicate row query help",
            "find duplicates using sql query",
        ],
    }
    texts = []
    boundaries: dict[str, tuple[int, int]] = {}
    for topic, prompts in topics.items():
        start = len(texts)
        texts += prompts * 3  # repeat for enough density for HDBSCAN's min_cluster_size
        boundaries[topic] = (start, len(texts))

    vectors = HashingEmbedder(dims=128).embed(texts)
    labels = _hdbscan_cluster(vectors, min_cluster_size=5)  # real backend, no fallback

    label_sets = {
        topic: {lbl for lbl in labels[start:end] if lbl != -1}
        for topic, (start, end) in boundaries.items()
    }
    assert label_sets["code"], "expected at least one non-noise cluster for the code topic"
    assert label_sets["weather"], "expected at least one non-noise cluster for the weather topic"
    assert label_sets["sql"], "expected at least one non-noise cluster for the sql topic"
    # HDBSCAN may split a topic into several tight sub-clusters, but must
    # never merge two different topics into the same cluster label.
    assert label_sets["code"].isdisjoint(label_sets["weather"])
    assert label_sets["code"].isdisjoint(label_sets["sql"])
    assert label_sets["weather"].isdisjoint(label_sets["sql"])

    # cluster_embeddings() must actually dispatch to this backend when
    # hdbscan is importable, not silently use the fallback.
    dispatched_labels = cluster_embeddings(vectors, min_cluster_size=5)
    assert dispatched_labels == labels
