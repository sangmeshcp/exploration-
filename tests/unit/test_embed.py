from shadowtrace.mining.embed import HashingEmbedder, cosine_similarity, get_default_embedder


def test_hashing_embedder_is_deterministic() -> None:
    e = HashingEmbedder(dims=64)
    v1 = e.embed(["how do I write a for loop in python"])[0]
    v2 = e.embed(["how do I write a for loop in python"])[0]
    assert v1 == v2


def test_hashing_embedder_vectors_are_normalized() -> None:
    e = HashingEmbedder(dims=64)
    vecs = e.embed(["some text here", "other words entirely"])
    for v in vecs:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6


def test_similar_texts_score_higher_than_dissimilar() -> None:
    e = HashingEmbedder(dims=128)
    a, b, c = e.embed(
        [
            "how do I write a for loop in python",
            "how do I write a for loop in python please",
            "what is the weather forecast for tomorrow in paris",
        ]
    )
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


def test_cosine_similarity_handles_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_get_default_embedder_falls_back_without_ml_extra() -> None:
    embedder = get_default_embedder()
    assert isinstance(embedder, HashingEmbedder)
