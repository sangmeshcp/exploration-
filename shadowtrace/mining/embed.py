"""Local embeddings of prompt text (plan.md M2.1, D5).

Two backends behind one `Embedder` protocol:

- `SentenceTransformerEmbedder` — the real thing (D5: local
  sentence-transformers, privacy default). Requires the `ml` extra
  (`pip install '.[ml]'`); not a core dependency so the base install and
  CI stay fast and network-independent.
- `HashingEmbedder` — a deterministic, dependency-free hashed
  bag-of-words fallback. Semantically weak but stable and fully local;
  used automatically when `sentence-transformers` isn't installed, so
  clustering (mining/cluster.py) and its tests never require a model
  download.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from shadowtrace.common.logging import get_logger

logger = get_logger("mining.embed")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    dims: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # renamed from get_sentence_embedding_dimension in newer
        # sentence-transformers releases; prefer the new name so users on
        # them don't see a FutureWarning, but keep working on older ones
        get_dims = getattr(self._model, "get_embedding_dimension", None)
        if get_dims is None:
            get_dims = self._model.get_sentence_embedding_dimension
        self.dims = get_dims()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()  # type: ignore[no-any-return]


def get_default_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; using local hashing embedder fallback "
            "(pip install '.[ml]' for semantic embeddings)"
        )
        return HashingEmbedder()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
