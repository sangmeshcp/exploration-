from shadowtrace.mining.cluster import cluster_embeddings, keyword_label, medoid_index
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
