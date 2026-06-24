"""
Document store — persists reference document chunks in a local Qdrant collection.
Chunks by paragraph boundaries (natural for physics/math documents).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from memory.embedder import VECTOR_SIZE, embed_one
from memory.schemas import DocumentChunk

log = logging.getLogger(__name__)
COLLECTION = "reference_docs"


def _point_id(chunk: DocumentChunk) -> str:
    key = f"doc:{chunk.source}:{chunk.text[:80]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class DocumentStore:

    def __init__(self, client: QdrantClient):
        self._client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            log.debug("Created Qdrant collection '%s'", COLLECTION)

    def ingest_file(self, path: str, system_type: str = "") -> int:
        text = Path(path).read_text(encoding="utf-8")
        return self.ingest_text(text, source=Path(path).name, system_type=system_type)

    def ingest_text(
        self,
        text: str,
        source: str,
        system_type: str = "",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> int:
        chunks = _paragraph_chunk(text, source, system_type, chunk_size, overlap)
        return self.ingest_chunks(chunks)

    def ingest_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        points = []
        for chunk in chunks:
            vector = embed_one(chunk.to_embedding_text()).tolist()
            points.append(PointStruct(
                id=_point_id(chunk),
                vector=vector,
                payload=chunk.model_dump(),
            ))
        self._client.upsert(collection_name=COLLECTION, points=points)
        log.debug("Ingested %d chunks from '%s'", len(points), chunks[0].source)
        return len(points)

    def query(self, text: str, top_k: int = 5) -> List[DocumentChunk]:
        vector = embed_one(text).tolist()
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
        ).points
        results = []
        for h in hits:
            try:
                results.append(DocumentChunk.model_validate(h.payload))
            except Exception as exc:
                log.warning("Skipping malformed DocumentChunk payload: %s", exc)
        return results

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION).count


# ── Chunking ───────────────────────────────────────────────────────────────────

def _paragraph_chunk(
    text: str,
    source: str,
    system_type: str,
    chunk_size: int,
    overlap: int,
) -> List[DocumentChunk]:
    """
    Split by blank lines (paragraph boundaries) first, then merge small paragraphs
    and split oversized ones.  This preserves equation blocks as atomic units.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[DocumentChunk] = []
    current_parts: List[str] = []
    current_len  = 0
    section      = ""

    for para in paragraphs:
        first_line = para.split("\n")[0].strip()
        if first_line.startswith("#") or (first_line.isupper() and len(first_line) < 80):
            section = first_line.lstrip("#").strip()

        if current_len + len(para) > chunk_size and current_parts:
            chunks.append(DocumentChunk(
                source=source, section=section,
                text="\n\n".join(current_parts), system_type=system_type,
            ))
            # keep last part for overlap
            current_parts = current_parts[-1:] if overlap > 0 else []
            current_len   = len(current_parts[0]) if current_parts else 0

        current_parts.append(para)
        current_len += len(para)

    if current_parts:
        chunks.append(DocumentChunk(
            source=source, section=section,
            text="\n\n".join(current_parts), system_type=system_type,
        ))

    return chunks or [DocumentChunk(source=source, text=text[:chunk_size], system_type=system_type)]
