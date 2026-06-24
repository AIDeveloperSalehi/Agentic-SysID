"""
Episodic store — persists RunSummary objects in a local Qdrant collection.
No server required: data lives at data/memory/qdrant on disk.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)

from memory.embedder import VECTOR_SIZE, embed_one
from memory.schemas import RunSummary

log = logging.getLogger(__name__)
COLLECTION = "episodic_runs"


def _point_id(run_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"run:{run_id}"))


def make_client(path: str) -> QdrantClient:
    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


class EpisodicStore:

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

    def add_run(self, summary: RunSummary) -> None:
        vector = embed_one(summary.to_embedding_text()).tolist()
        self._client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=_point_id(summary.run_id),
                vector=vector,
                payload=summary.model_dump(),
            )],
        )
        log.debug("Stored episodic run: %s (%s)", summary.run_id, summary.plant_name)

    def query(
        self,
        text: str,
        top_k: int = 5,
        filter_system_order: Optional[int] = None,
    ) -> List[RunSummary]:
        vector = embed_one(text).tolist()
        qdrant_filter = None
        if filter_system_order is not None:
            qdrant_filter = Filter(must=[
                FieldCondition(key="system_order", match=MatchValue(value=filter_system_order))
            ])
        hits = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points
        results = []
        for h in hits:
            try:
                results.append(RunSummary.model_validate(h.payload))
            except Exception as exc:
                log.warning("Skipping malformed RunSummary payload: %s", exc)
        return results

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION).count
