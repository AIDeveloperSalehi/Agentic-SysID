"""
RetrievalService — unified interface for agents.

Two-stage retrieval:
  1. ANN vector search (fast, top-K*4 candidates)
  2. Cross-encoder reranking (accurate, on the candidates)

Falls back to vector-only if the cross-encoder is unavailable.
Both EpisodicStore and DocumentStore share the same local Qdrant instance.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from memory.document_store import DocumentStore
from memory.episodic_store import EpisodicStore
from memory.schemas import DocumentChunk, RunSummary

log = logging.getLogger(__name__)

_reranker      = None
_reranker_tried = False


def _get_reranker():
    global _reranker, _reranker_tried
    if _reranker_tried:
        return _reranker
    _reranker_tried = True
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        log.debug("CrossEncoder reranker loaded.")
    except Exception as exc:
        log.info("CrossEncoder unavailable (%s) — using vector scores only.", exc)
        _reranker = None
    return _reranker


class RetrievalService:
    """
    Unified retrieval for agents.

    Usage
    -----
    svc = RetrievalService(data_dir="data")

    # Inject formatted context into an LLM agent:
    context = svc.query("pendulum with Coulomb friction, 2nd order", top_k=3)

    # Get raw RunSummary objects for a deterministic agent (e.g. Estimator):
    runs = svc.query_runs_only("pendulum 2nd order", top_k=3)

    # Ingest a reference document once:
    svc.documents.ingest_file("references/pendulum_dynamics.md", system_type="pendulum")
    """

    def __init__(self, data_dir: str = "data"):
        from pathlib import Path
        from qdrant_client import QdrantClient
        qdrant_path = f"{data_dir}/memory/qdrant"
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=qdrant_path)
        self.episodic  = EpisodicStore(client)
        self.documents = DocumentStore(client)

    # ── Main query methods ─────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 3,
        filter_system_order: Optional[int] = None,
        include_docs: bool = True,
    ) -> str:
        """
        Search episodic runs + reference docs. Returns formatted string for agent context.
        """
        fetch_k = max(top_k * 4, 12)
        runs = self.episodic.query(text, top_k=fetch_k, filter_system_order=filter_system_order)
        docs = self.documents.query(text, top_k=fetch_k) if include_docs else []

        if not runs and not docs:
            return "No relevant prior runs or reference documents found."

        reranker = _get_reranker()
        runs = _rerank(reranker, text, runs, top_k, key=lambda r: r.to_embedding_text())
        docs = _rerank(reranker, text, docs, top_k, key=lambda d: d.to_embedding_text()) if docs else []

        return _format(runs, docs)

    def query_runs_only(
        self,
        text: str,
        top_k: int = 3,
        filter_system_order: Optional[int] = None,
    ) -> List[RunSummary]:
        """Return raw RunSummary objects — used by deterministic agents."""
        fetch_k = max(top_k * 4, 12)
        runs    = self.episodic.query(text, top_k=fetch_k, filter_system_order=filter_system_order)
        reranker = _get_reranker()
        return _rerank(reranker, text, runs, top_k, key=lambda r: r.to_embedding_text())

    # ── Convenience ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "episodic_runs":   self.episodic.count(),
            "document_chunks": self.documents.count(),
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rerank(reranker, query: str, items, top_k: int, key) -> list:
    if not items:
        return []
    if reranker is None:
        return items[:top_k]
    pairs = [(query, key(item)) for item in items]
    try:
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_k]]
    except Exception as exc:
        log.warning("Reranking failed: %s", exc)
        return items[:top_k]


def _format(runs: List[RunSummary], docs: List[DocumentChunk]) -> str:
    parts = []
    if runs:
        parts.append("=== PRIOR IDENTIFICATION RUNS ===")
        for i, r in enumerate(runs, 1):
            parts.append(f"\n[Run {i}]")
            parts.append(r.to_context_string())
    if docs:
        parts.append("\n=== REFERENCE DOCUMENTS ===")
        for i, d in enumerate(docs, 1):
            parts.append(f"\n[Doc {i}]")
            parts.append(d.to_context_string())
    return "\n".join(parts)
