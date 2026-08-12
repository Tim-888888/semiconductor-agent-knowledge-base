"""Rebuild published chunks into a new embedding index without deleting the old index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from semikb.config import Settings
from semikb.contracts.models import Chunk, DocumentLifecycle
from semikb.rag_retrieval.encoders import HybridEncoder, create_hybrid_encoder
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.storage.clients import StorageClientFactory
from semikb.storage.milvus_chunks import MilvusChunkRepository
from semikb.storage.provisioning import provision_milvus_index


@dataclass(frozen=True, slots=True)
class EmbeddingMigrationPlan:
    source_collection: str
    target_collection: str
    target_index_version: str
    embedding_version: str
    embedding_dim: int
    published_documents: int
    published_chunks: int
    target_exists: bool
    alias_switch_required: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EmbeddingIndexMigrator:
    def __init__(
        self,
        settings: Settings,
        target_index_version: str,
        *,
        encoder: HybridEncoder | None = None,
        factory: StorageClientFactory | None = None,
    ) -> None:
        if settings.demo_mode:
            raise ValueError("Embedding index migration requires DEMO_MODE=false.")
        if not target_index_version or target_index_version == settings.milvus_index_version:
            raise ValueError("Target index version must differ from the configured source version.")
        self.settings = settings.model_copy(
            update={"milvus_index_version": target_index_version}
        )
        self.target_index_version = target_index_version
        self.factory = factory or StorageClientFactory(self.settings)
        self.encoder = encoder or create_hybrid_encoder(self.settings)
        self.vectors = MilvusChunkRepository(self.factory, self.settings)

    def plan(self) -> EmbeddingMigrationPlan:
        source_collection = self._active_collection()
        target_collection = collection_name(self.target_index_version)
        with self.factory.mongodb() as client:
            database = client[self.settings.mongodb_database]
            document_count = database.document_catalog.count_documents(
                {"lifecycle": DocumentLifecycle.PUBLISHED.value}
            )
            chunk_count = database.chunk_catalog.count_documents(
                {"lifecycle": DocumentLifecycle.PUBLISHED.value}
            )
        with self.factory.milvus() as client:
            target_exists = client.has_collection(target_collection)
        return EmbeddingMigrationPlan(
            source_collection=source_collection,
            target_collection=target_collection,
            target_index_version=self.target_index_version,
            embedding_version=self.settings.embedding_version,
            embedding_dim=self.settings.embedding_dim,
            published_documents=document_count,
            published_chunks=chunk_count,
            target_exists=target_exists,
            alias_switch_required=source_collection != target_collection,
        )

    def apply(self) -> dict[str, object]:
        plan = self.plan()
        if plan.published_documents <= 0 or plan.published_chunks <= 0:
            raise RuntimeError("No published documents and chunks are available to rebuild.")
        if not plan.alias_switch_required:
            raise RuntimeError("The target collection is already active.")

        provision_result = provision_milvus_index(
            self.settings,
            self.target_index_version,
        )
        chunks = self._load_published_chunks()
        if len(chunks) != plan.published_chunks:
            raise RuntimeError("Published Chunk count changed while building the migration plan.")
        embeddings = self._encode_chunks(chunks)
        self.vectors.upsert_chunks(
            chunks,
            embeddings,
            lifecycle=DocumentLifecycle.PUBLISHED,
        )
        self._verify_rows(chunks)
        self._verify_dense_probe(chunks, embeddings)

        alias_switched = False
        try:
            self.vectors.activate_alias(self.target_index_version)
            alias_switched = True
            mongo_counts = self._update_catalog_versions(plan)
            if self._active_collection() != plan.target_collection:
                raise RuntimeError("Milvus alias verification failed after migration.")
        except Exception:
            if alias_switched:
                self._restore_alias(plan.source_collection)
            raise

        return {
            "status": "applied",
            "plan": plan.as_dict(),
            "provision": provision_result,
            "encoded_chunks": len(chunks),
            "catalog_updates": mongo_counts,
            "active_collection": self._active_collection(),
            "rollback_collection_retained": plan.source_collection,
        }

    def _load_published_chunks(self) -> list[Chunk]:
        with self.factory.mongodb() as client:
            records = list(
                client[self.settings.mongodb_database]
                .chunk_catalog.find(
                    {"lifecycle": DocumentLifecycle.PUBLISHED.value},
                    {"_id": 0},
                )
                .sort("chunk_id", 1)
            )
        return [
            Chunk.model_validate(record).model_copy(
                update={
                    "embedding_version": self.settings.embedding_version,
                    "index_version": self.target_index_version,
                }
            )
            for record in records
        ]

    def _encode_chunks(self, chunks: Sequence[Chunk]):
        embeddings = []
        batch_size = self.settings.embedding_batch_size
        texts = [chunk.chunk_text for chunk in chunks]
        for offset in range(0, len(texts), batch_size):
            embeddings.extend(self.encoder.encode(texts[offset : offset + batch_size]))
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding provider omitted one or more published chunks.")
        return embeddings

    def _verify_rows(self, chunks: Sequence[Chunk]) -> None:
        target = collection_name(self.target_index_version)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        with self.factory.milvus() as client:
            rows = client.query(
                target,
                ids=chunk_ids,
                output_fields=["chunk_id", "lifecycle", "index_version"],
                consistency_level="Strong",
            )
        if len(rows) != len(chunks):
            raise RuntimeError("Milvus target collection is missing rebuilt rows.")
        if any(
            row.get("lifecycle") != DocumentLifecycle.PUBLISHED.value
            or row.get("index_version") != self.target_index_version
            for row in rows
        ):
            raise RuntimeError("Milvus target collection failed metadata verification.")

    def _verify_dense_probe(self, chunks, embeddings) -> None:
        target = collection_name(self.target_index_version)
        with self.factory.milvus() as client:
            results = client.search(
                target,
                data=[embeddings[0].dense],
                anns_field="dense_vector",
                filter='lifecycle == "published"',
                limit=min(3, len(chunks)),
                output_fields=["chunk_id"],
                search_params={"metric_type": "IP", "params": {"ef": 96}},
                consistency_level="Strong",
            )
        hit_ids = {
            str(hit.get("chunk_id") or hit.get("id"))
            for hit in (results[0] if results else [])
        }
        if chunks[0].chunk_id not in hit_ids:
            raise RuntimeError("Dense self-retrieval probe failed on the target collection.")

    def _update_catalog_versions(self, plan: EmbeddingMigrationPlan) -> dict[str, int]:
        selector = {"lifecycle": DocumentLifecycle.PUBLISHED.value}
        values = {
            "embedding_version": self.settings.embedding_version,
            "index_version": self.target_index_version,
        }
        with self.factory.mongodb() as client:
            database = client[self.settings.mongodb_database]
            document_result = database.document_catalog.update_many(selector, {"$set": values})
            chunk_result = database.chunk_catalog.update_many(selector, {"$set": values})
            if document_result.matched_count != plan.published_documents:
                raise RuntimeError("Document catalog count changed during migration.")
            if chunk_result.matched_count != plan.published_chunks:
                raise RuntimeError("Chunk catalog count changed during migration.")
            database.index_releases.update_many(
                {"status": "active", "index_version": {"$ne": self.target_index_version}},
                {"$set": {"status": "inactive", "deactivated_at": datetime.now(UTC)}},
            )
            database.index_releases.update_one(
                {"index_version": self.target_index_version},
                {
                    "$set": {
                        "status": "active",
                        "alias": "semikb_chunks_active",
                        "collection": plan.target_collection,
                        "embedding_version": self.settings.embedding_version,
                        "embedding_dim": self.settings.embedding_dim,
                        "normalization": "l2",
                        "sparse_encoder_version": self.settings.sparse_encoder_version,
                        "migration_source_collection": plan.source_collection,
                        "published_at": datetime.now(UTC),
                        "published_chunks": plan.published_chunks,
                    }
                },
                upsert=True,
            )
        return {
            "documents": document_result.matched_count,
            "chunks": chunk_result.matched_count,
        }

    def _active_collection(self) -> str:
        with self.factory.milvus() as client:
            alias = client.describe_alias("semikb_chunks_active")
        collection = str(alias.get("collection_name", ""))
        if not collection:
            raise RuntimeError("semikb_chunks_active alias is missing.")
        return collection

    def _restore_alias(self, source_collection: str) -> None:
        with self.factory.milvus() as client:
            client.alter_alias(source_collection, "semikb_chunks_active")
