"""Build and publish a governed embedding index without overwriting the active index."""

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
    source_index_version: str
    source_embedding_version: str
    target_collection: str
    target_index_version: str
    target_embedding_version: str
    target_output_type: str
    target_sparse_encoder_version: str
    embedding_dim: int
    published_documents: int
    published_chunks: int
    target_exists: bool
    alias_switch_required: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EmbeddingIndexMigrator:
    """Separates candidate construction from the explicitly approved alias switch."""

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
        self.source_settings = settings
        self.settings = settings.model_copy(
            update={"milvus_index_version": target_index_version}
        )
        self.target_index_version = target_index_version
        self.factory = factory or StorageClientFactory(self.settings)
        self.encoder = encoder
        self.vectors = MilvusChunkRepository(self.factory, self.settings)

    def plan(self) -> EmbeddingMigrationPlan:
        source_collection = self._active_collection()
        target_collection = collection_name(self.target_index_version)
        selector = {"lifecycle": DocumentLifecycle.PUBLISHED.value}
        with self.factory.mongodb() as client:
            database = client[self.settings.mongodb_database]
            document_count = database.document_catalog.count_documents(selector)
            chunk_count = database.chunk_catalog.count_documents(selector)
            source_index_version = _single_catalog_value(
                database.document_catalog.distinct("index_version", selector),
                database.chunk_catalog.distinct("index_version", selector),
                "index_version",
            )
            source_embedding_version = _single_catalog_value(
                database.document_catalog.distinct("embedding_version", selector),
                database.chunk_catalog.distinct("embedding_version", selector),
                "embedding_version",
            )
        expected_source_collection = collection_name(source_index_version)
        if source_collection != expected_source_collection:
            raise RuntimeError(
                "Milvus active alias and MongoDB published index version disagree."
            )
        with self.factory.milvus() as client:
            target_exists = client.has_collection(target_collection)
        return EmbeddingMigrationPlan(
            source_collection=source_collection,
            source_index_version=source_index_version,
            source_embedding_version=source_embedding_version,
            target_collection=target_collection,
            target_index_version=self.target_index_version,
            target_embedding_version=self.settings.embedding_version,
            target_output_type=self.settings.embedding_output_type,
            target_sparse_encoder_version=self.settings.sparse_encoder_version,
            embedding_dim=self.settings.embedding_dim,
            published_documents=document_count,
            published_chunks=chunk_count,
            target_exists=target_exists,
            alias_switch_required=source_collection != target_collection,
        )

    def build(self) -> dict[str, object]:
        """Create and validate a candidate Collection without changing the active alias."""

        plan = self.plan()
        self._validate_migration_scope(plan)
        provision_result = provision_milvus_index(
            self.settings,
            self.target_index_version,
        )
        chunks = self._load_published_chunks()
        self._verify_catalog_snapshot(plan, chunks)
        embeddings = self._encode_chunks(chunks)
        self.vectors.upsert_chunks(
            chunks,
            embeddings,
            lifecycle=DocumentLifecycle.PUBLISHED,
        )
        self._verify_rows(chunks)
        self._verify_dense_probe(chunks)
        self._record_candidate(plan)
        if self._active_collection() != plan.source_collection:
            raise RuntimeError("Candidate build unexpectedly changed the active Milvus alias.")
        return {
            "status": "candidate_built",
            "plan": plan.as_dict(),
            "provision": provision_result,
            "encoded_chunks": len(chunks),
            "active_collection": self._active_collection(),
            "candidate_collection": plan.target_collection,
        }

    def publish(self) -> dict[str, object]:
        """Publish a previously built candidate after external shadow evaluation passes."""

        plan = self.plan()
        self._validate_migration_scope(plan)
        if not plan.target_exists:
            raise RuntimeError("Target Collection does not exist; build the candidate first.")
        self._verify_candidate_release(plan)
        chunks = self._load_published_chunks()
        self._verify_catalog_snapshot(plan, chunks)
        self._verify_rows(chunks)
        self._verify_dense_probe(chunks)

        alias_switched = False
        try:
            self.vectors.activate_alias(self.target_index_version)
            alias_switched = True
            mongo_counts = self._update_catalog_versions(plan)
            if self._active_collection() != plan.target_collection:
                raise RuntimeError("Milvus alias verification failed after migration.")
        except Exception as exc:
            rollback_errors: list[str] = []
            if alias_switched:
                try:
                    self._restore_alias(plan.source_collection)
                except Exception as rollback_exc:
                    rollback_errors.append(f"alias:{type(rollback_exc).__name__}")
            try:
                self._restore_catalog_versions(plan)
            except Exception as rollback_exc:
                rollback_errors.append(f"catalog:{type(rollback_exc).__name__}")
            if rollback_errors:
                raise RuntimeError(
                    "Embedding index publish failed and rollback requires manual repair: "
                    + ", ".join(rollback_errors)
                ) from exc
            raise

        return {
            "status": "published",
            "plan": plan.as_dict(),
            "catalog_updates": mongo_counts,
            "active_collection": self._active_collection(),
            "rollback_collection_retained": plan.source_collection,
        }

    @staticmethod
    def _validate_migration_scope(plan: EmbeddingMigrationPlan) -> None:
        if plan.published_documents <= 0 or plan.published_chunks <= 0:
            raise RuntimeError("No published documents and chunks are available to rebuild.")
        if not plan.alias_switch_required:
            raise RuntimeError("The target Collection is already active.")

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

    def _verify_catalog_snapshot(
        self,
        plan: EmbeddingMigrationPlan,
        chunks: Sequence[Chunk],
    ) -> None:
        if len(chunks) != plan.published_chunks:
            raise RuntimeError("Published Chunk count changed during migration.")
        current = self.plan()
        if (
            current.source_collection != plan.source_collection
            or current.source_index_version != plan.source_index_version
            or current.source_embedding_version != plan.source_embedding_version
            or current.published_documents != plan.published_documents
            or current.published_chunks != plan.published_chunks
        ):
            raise RuntimeError("Published catalog changed during migration.")

    def _encode_chunks(self, chunks: Sequence[Chunk]):
        encoder = self.encoder or create_hybrid_encoder(self.settings)
        embeddings = []
        batch_size = self.settings.embedding_batch_size
        texts = [chunk.chunk_text for chunk in chunks]
        for offset in range(0, len(texts), batch_size):
            embeddings.extend(encoder.encode(texts[offset : offset + batch_size]))
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
            counts = client.query(
                target,
                filter="",
                output_fields=["count(*)"],
                consistency_level="Strong",
            )
        row_count = int(counts[0].get("count(*)", 0)) if counts else 0
        if len(rows) != len(chunks) or row_count != len(chunks):
            raise RuntimeError("Milvus target Collection has missing or unexpected rows.")
        if any(
            row.get("lifecycle") != DocumentLifecycle.PUBLISHED.value
            or row.get("index_version") != self.target_index_version
            for row in rows
        ):
            raise RuntimeError("Milvus target Collection failed metadata verification.")

    def _verify_dense_probe(self, chunks: Sequence[Chunk]) -> None:
        target = collection_name(self.target_index_version)
        with self.factory.milvus() as client:
            rows = client.query(
                target,
                ids=[chunks[0].chunk_id],
                output_fields=["chunk_id", "dense_vector"],
                consistency_level="Strong",
            )
            if len(rows) != 1 or not rows[0].get("dense_vector"):
                raise RuntimeError("Candidate Dense vector could not be read back.")
            results = client.search(
                target,
                data=[rows[0]["dense_vector"]],
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
            raise RuntimeError("Dense self-retrieval probe failed on the target Collection.")

    def _record_candidate(self, plan: EmbeddingMigrationPlan) -> None:
        with self.factory.mongodb() as client:
            client[self.settings.mongodb_database].index_releases.update_one(
                {"index_version": self.target_index_version},
                {
                    "$set": {
                        "status": "candidate",
                        "collection": plan.target_collection,
                        "embedding_version": self.settings.embedding_version,
                        "embedding_dim": self.settings.embedding_dim,
                        "embedding_output_type": self.settings.embedding_output_type,
                        "normalization": "dense_l2_sparse_provider_raw",
                        "sparse_encoder_version": self.settings.sparse_encoder_version,
                        "migration_source_collection": plan.source_collection,
                        "built_at": datetime.now(UTC),
                        "published_chunks": plan.published_chunks,
                    }
                },
                upsert=True,
            )

    def _verify_candidate_release(self, plan: EmbeddingMigrationPlan) -> None:
        with self.factory.mongodb() as client:
            candidate = client[self.settings.mongodb_database].index_releases.find_one(
                {
                    "index_version": self.target_index_version,
                    "status": "candidate",
                    "collection": plan.target_collection,
                    "embedding_version": plan.target_embedding_version,
                    "embedding_output_type": plan.target_output_type,
                    "sparse_encoder_version": plan.target_sparse_encoder_version,
                    "published_chunks": plan.published_chunks,
                }
            )
        if candidate is None:
            raise RuntimeError("Target Collection has no matching candidate release record.")

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
            now = datetime.now(UTC)
            database.index_releases.update_one(
                {"index_version": plan.source_index_version},
                {
                    "$set": {"status": "inactive", "deactivated_at": now},
                    "$unset": {"alias": ""},
                },
            )
            database.index_releases.update_one(
                {"index_version": self.target_index_version},
                {
                    "$set": {
                        "status": "active",
                        "alias": "semikb_chunks_active",
                        "published_at": now,
                    },
                    "$unset": {"deactivated_at": ""},
                },
            )
        return {
            "documents": document_result.matched_count,
            "chunks": chunk_result.matched_count,
        }

    def _restore_catalog_versions(self, plan: EmbeddingMigrationPlan) -> None:
        selector = {"lifecycle": DocumentLifecycle.PUBLISHED.value}
        source_values = {
            "embedding_version": plan.source_embedding_version,
            "index_version": plan.source_index_version,
        }
        with self.factory.mongodb() as client:
            database = client[self.settings.mongodb_database]
            database.document_catalog.update_many(selector, {"$set": source_values})
            database.chunk_catalog.update_many(selector, {"$set": source_values})
            database.index_releases.update_one(
                {"index_version": plan.source_index_version},
                {
                    "$set": {"status": "active", "alias": "semikb_chunks_active"},
                    "$unset": {"deactivated_at": ""},
                },
            )
            database.index_releases.update_one(
                {"index_version": self.target_index_version},
                {
                    "$set": {
                        "status": "candidate",
                        "publish_rollback_at": datetime.now(UTC),
                    },
                    "$unset": {"published_at": "", "alias": ""},
                },
            )

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


def _single_catalog_value(
    document_values: Sequence[object],
    chunk_values: Sequence[object],
    field: str,
) -> str:
    documents = {str(value) for value in document_values if value}
    chunks = {str(value) for value in chunk_values if value}
    if len(documents) != 1 or documents != chunks:
        raise RuntimeError(f"Published MongoDB catalogs disagree on {field}.")
    return next(iter(documents))
