"""Production retrieval reads across Milvus, MongoDB, and private MinIO assets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    ApprovalStatus,
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    RetrievalConstraints,
    RetrievalTrace,
)
from semikb.storage.clients import StorageClientFactory
from semikb.storage.minio_artifacts import MinioArtifactRepository


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float
    rank: int


def _literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _in_filter(field: str, values: Sequence[str]) -> str:
    return f"{field} in [{', '.join(_literal(value) for value in values)}]"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def build_access_filter(
    actor_scope: ActorScope,
    constraints: RetrievalConstraints | None,
    *,
    now: datetime,
) -> tuple[str, dict[str, Any]]:
    """Build a Milvus expression from validated fields, never raw expression fragments."""

    point_in_time = constraints.as_of if constraints and constraints.as_of else now
    point_in_time = _utc(point_in_time)
    epoch = int(point_in_time.timestamp())
    clauses = [
        'approval_status == "approved"',
        'lifecycle == "published"',
        f"effective_at <= {epoch}",
        f"(expires_at == 0 or expires_at > {epoch})",
    ]
    metadata: dict[str, Any] = {
        "approval_status": "approved",
        "lifecycle": "published",
        "as_of": point_in_time.isoformat(),
    }
    if "admin" not in actor_scope.roles:
        if not actor_scope.access_scope_keys:
            clauses.append('chunk_id == "__no_access__"')
        else:
            clauses.append(_in_filter("access_scope_key", actor_scope.access_scope_keys))
        for field, values in (
            ("fab", actor_scope.fabs),
            ("product", actor_scope.products),
            ("tool_id", actor_scope.tool_ids),
        ):
            clauses.append(
                f"({field} == \"\" or {_in_filter(field, values)})"
                if values
                else f'{field} == ""'
            )
        metadata.update(
            {
                "access_scope_keys": actor_scope.access_scope_keys,
                "fabs": actor_scope.fabs,
                "products": actor_scope.products,
                "tool_ids": actor_scope.tool_ids,
            }
        )

    if constraints:
        for field in (
            "fab",
            "product",
            "process_layer",
            "tool_id",
            "chamber",
            "recipe_id",
            "recipe_version",
        ):
            value = getattr(constraints, field)
            if value:
                clauses.append(f"{field} == {_literal(value)}")
                metadata[field] = value
    return " and ".join(clauses), metadata


class ProductionRetrievalRepository:
    def __init__(self, settings: Settings, factory: StorageClientFactory | None = None) -> None:
        self._settings = settings
        self._factory = factory or StorageClientFactory(settings)
        self._artifacts = MinioArtifactRepository(self._factory)

    def vector_search(
        self,
        vector: list[float] | dict[int, float],
        *,
        vector_field: str,
        filter_expression: str,
        limit: int,
    ) -> list[VectorHit]:
        search_params = (
            {"metric_type": "IP", "params": {"ef": 96}}
            if vector_field == "dense_vector"
            else {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
        )
        with self._factory.milvus() as client:
            results = client.search(
                "semikb_chunks_active",
                data=[vector],
                anns_field=vector_field,
                filter=filter_expression,
                limit=limit,
                output_fields=["chunk_id"],
                search_params=search_params,
                consistency_level="Strong",
            )
        hits = results[0] if results else []
        return [
            VectorHit(
                chunk_id=str(hit.get("chunk_id") or hit.get("id")),
                score=float(hit.get("distance", 0.0)),
                rank=index + 1,
            )
            for index, hit in enumerate(hits)
        ]

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._factory.mongodb() as client:
            database = client[self._settings.mongodb_database]
            records = list(
                database.chunk_catalog.find(
                    {"chunk_id": {"$in": list(chunk_ids)}},
                    {"_id": 0},
                )
            )
            document_keys = {
                (str(record["document_id"]), str(record["revision"])) for record in records
            }
            document_records = list(
                database.document_catalog.find(
                    {
                        "$or": [
                            {"document_id": document_id, "revision": revision}
                            for document_id, revision in document_keys
                        ]
                    },
                    {"_id": 0, "document_id": 1, "revision": 1, "source_uri": 1, "title": 1},
                )
            ) if document_keys else []
        documents = {
            (str(item["document_id"]), str(item["revision"])): item
            for item in document_records
        }
        for record in records:
            document = documents.get((str(record["document_id"]), str(record["revision"])), {})
            record["metadata"] = {
                **record.get("metadata", {}),
                "source_uri": document.get("source_uri", ""),
                "document_title": document.get("title", ""),
            }
        return {
            chunk.chunk_id: chunk
            for chunk in (Chunk.model_validate(record) for record in records)
        }

    def save_trace(self, trace: RetrievalTrace) -> RetrievalTrace:
        with self._factory.mongodb() as client:
            client[self._settings.mongodb_database].retrieval_traces.replace_one(
                {"trace_id": trace.trace_id},
                trace.model_dump(mode="python"),
                upsert=True,
            )
        return trace

    def get_trace(
        self,
        trace_id: str,
        actor_scope: ActorScope | None = None,
    ) -> RetrievalTrace | None:
        selector: dict[str, Any] = {"trace_id": trace_id}
        if actor_scope is not None and "admin" not in actor_scope.roles:
            selector["actor_user_id"] = actor_scope.user_id
        with self._factory.mongodb() as client:
            record = client[self._settings.mongodb_database].retrieval_traces.find_one(
                selector,
                {"_id": 0},
            )
        return RetrievalTrace.model_validate(record) if record else None

    def list_traces(self, actor_scope: ActorScope | None = None) -> list[RetrievalTrace]:
        selector: dict[str, Any] = {}
        if actor_scope is not None and "admin" not in actor_scope.roles:
            selector["actor_user_id"] = actor_scope.user_id
        with self._factory.mongodb() as client:
            records = list(
                client[self._settings.mongodb_database]
                .retrieval_traces.find(selector, {"_id": 0})
                .sort("created_at", -1)
            )
        return [RetrievalTrace.model_validate(record) for record in records]

    def get_image(self, image_id: str) -> ImageAsset | None:
        with self._factory.mongodb() as client:
            record = client[self._settings.mongodb_database].image_assets.find_one(
                {"image_id": image_id},
                {"_id": 0},
            )
        return ImageAsset.model_validate(record) if record else None

    def get_document(self, document_id: str, revision: str) -> DocumentRevision | None:
        with self._factory.mongodb() as client:
            record = client[self._settings.mongodb_database].document_catalog.find_one(
                {"document_id": document_id, "revision": revision},
                {"_id": 0},
            )
        return DocumentRevision.model_validate(record) if record else None

    def asset_access(self, image_id: str, actor_scope: ActorScope) -> dict[str, str]:
        asset = self.get_image(image_id)
        if asset is None:
            raise KeyError(image_id)
        document = self.get_document(asset.document_id, asset.revision)
        if document is None:
            raise KeyError(image_id)
        document_probe = Chunk(
            chunk_id=f"asset-{asset.image_id}",
            document_id=asset.document_id,
            revision=asset.revision,
            chunk_text=asset.caption,
            page_or_section=asset.source_page,
            approval_status=document.approval_status,
            lifecycle=document.lifecycle,
            effective_at=document.effective_at,
            expires_at=document.expires_at,
            access_scope_key=document.access_scope_key,
            fab=document.fab,
            product=document.product,
            tool_id=document.tool_id,
        )
        asset_probe = document_probe.model_copy(
            update={
                "approval_status": asset.approval_status,
                "lifecycle": asset.lifecycle,
                "effective_at": asset.effective_at,
                "expires_at": asset.expires_at,
                "access_scope_key": asset.access_scope_key,
            }
        )
        current = datetime.now(UTC)
        if not self.is_accessible(
            document_probe,
            actor_scope,
            current,
            None,
        ) or not self.is_accessible(asset_probe, actor_scope, current, None):
            raise PermissionError(image_id)
        expires = timedelta(minutes=5)
        return {
            "image_id": image_id,
            "url": self._artifacts.presign_get(asset.object_ref, expires=expires),
            "expires_at": (datetime.now(UTC) + expires).isoformat(),
            "object_key": asset.object_ref.object_key,
        }

    @staticmethod
    def is_accessible(
        chunk: Chunk,
        actor_scope: ActorScope,
        current: datetime,
        constraints: RetrievalConstraints | None,
    ) -> bool:
        point_in_time = constraints.as_of if constraints and constraints.as_of else current
        point_in_time = _utc(point_in_time)
        if chunk.lifecycle is not DocumentLifecycle.PUBLISHED:
            return False
        if chunk.approval_status is not ApprovalStatus.APPROVED:
            return False
        if _utc(chunk.effective_at) > point_in_time:
            return False
        if chunk.expires_at and _utc(chunk.expires_at) <= point_in_time:
            return False
        if "admin" not in actor_scope.roles:
            if chunk.access_scope_key not in actor_scope.access_scope_keys:
                return False
            if chunk.fab and chunk.fab not in actor_scope.fabs:
                return False
            if chunk.product and chunk.product not in actor_scope.products:
                return False
            if chunk.tool_id and chunk.tool_id not in actor_scope.tool_ids:
                return False
        if constraints:
            for field in (
                "fab",
                "product",
                "process_layer",
                "tool_id",
                "chamber",
                "recipe_id",
                "recipe_version",
            ):
                expected = getattr(constraints, field)
                if expected and getattr(chunk, field) != expected:
                    return False
        return True
