"""Milvus staging and publication operations for ingestion chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import Chunk, DocumentLifecycle
from semikb.rag_retrieval.encoders import HybridEmbedding
from semikb.rag_retrieval.milvus_schema import chunk_to_milvus_row, collection_name
from semikb.storage.clients import StorageClientFactory


def _alias_names(result: Any) -> set[str]:
    """Normalize pymilvus alias responses across client versions."""
    if isinstance(result, dict):
        aliases = result.get("aliases", [])
    else:
        aliases = result
    return {str(alias) for alias in aliases or []}


class MilvusChunkRepository:
    """Writes vectors to a physical index version before alias publication."""

    def __init__(self, factory: StorageClientFactory, settings: Settings) -> None:
        self._factory = factory
        self._settings = settings

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[HybridEmbedding],
        *,
        lifecycle: DocumentLifecycle,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every Milvus chunk requires one embedding.")
        if not chunks:
            raise ValueError("At least one chunk is required for Milvus staging.")
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            stored_chunk = chunk.model_copy(update={"lifecycle": lifecycle})
            rows.append(
                chunk_to_milvus_row(
                    stored_chunk,
                    embedding.dense,
                    embedding.sparse,
                    embedding_dim=self._settings.embedding_dim,
                )
            )
        physical_collection = collection_name(chunks[0].index_version)
        with self._factory.milvus() as client:
            result = client.upsert(physical_collection, rows)
            upsert_count = int(result.get("upsert_count", result.get("insert_count", 0)))
            if upsert_count != len(rows):
                raise RuntimeError(
                    f"Milvus acknowledged {upsert_count} of {len(rows)} staged chunks."
                )
            stored = client.query(
                physical_collection,
                ids=[chunk.chunk_id for chunk in chunks],
                output_fields=["chunk_id", "lifecycle"],
                consistency_level="Strong",
            )
            if len(stored) != len(chunks) or any(
                item.get("lifecycle") != lifecycle.value for item in stored
            ):
                raise RuntimeError("Milvus read-back verification failed after upsert.")

    def delete_chunks(self, index_version: str, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._factory.milvus() as client:
            client.delete(collection_name(index_version), ids=list(chunk_ids))

    def verify_chunks_absent(self, index_version: str, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._factory.milvus() as client:
            remaining = client.query(
                collection_name(index_version),
                ids=list(chunk_ids),
                output_fields=["chunk_id"],
                consistency_level="Strong",
            )
        if remaining:
            raise RuntimeError("Milvus still contains withdrawn chunk projections.")

    def activate_alias(self, index_version: str) -> None:
        physical_collection = collection_name(index_version)
        alias = "semikb_chunks_active"
        with self._factory.milvus() as client:
            if alias in _alias_names(
                client.list_aliases(collection_name=physical_collection)
            ):
                return
            if alias not in _alias_names(client.list_aliases()):
                client.create_alias(physical_collection, alias)
                return
            client.alter_alias(physical_collection, alias)
