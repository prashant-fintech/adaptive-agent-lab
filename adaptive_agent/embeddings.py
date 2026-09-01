"""Local embeddings via sentence-transformers.

The model loads lazily on first use. Vectors are L2-normalised, so cosine
similarity is a plain dot product.
"""

import numpy as np

from adaptive_agent import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, normalize_embeddings=True)


def cosine_scores(query: str, texts: list[str]) -> np.ndarray:
    """Similarity of `query` against each entry of `texts`, in order."""
    vectors = embed_texts([query] + texts)
    return vectors[1:] @ vectors[0]
