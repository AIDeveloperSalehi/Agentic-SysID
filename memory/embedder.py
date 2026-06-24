"""
Lazy-loading sentence-transformer wrapper.

The model is loaded once on first use and cached for the process lifetime.
Using all-MiniLM-L6-v2: 384-dim, ~80 MB, fast, strong on technical text.
"""
from __future__ import annotations

from typing import List
import numpy as np

MODEL_NAME  = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

_model = None


def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts → (N, 384) float32 array, L2-normalised."""
    return _load().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
