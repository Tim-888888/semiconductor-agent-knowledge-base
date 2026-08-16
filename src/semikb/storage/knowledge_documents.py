"""Knowledge-document catalog and lifecycle operation persistence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from pymongo import ReturnDocument

from semikb.contracts.models import (
    ActorScope,
    AffectedRecordCounts,
    AuditEvent,
    Chunk,
    DocumentLifecycle,
    DocumentLifecycleOperationRecord,
    DocumentRevision,
    DocumentRevisionSelector,
    ImageAsset,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRevisionSummary,
    KnowledgeDocumentSummary,
    SourceManifest,
    TableAsset,
)
from semikb.storage.clients import StorageClientFactory
from semikb.storage.memory import DemoStore

_REVISION_COLLECTIONS = ("chunk_catalog", "image_assets", "table_assets")


class LifecycleOperationRequestConflictError(ValueError):
    """A request identifier was reused for a different lifecycle operation."""


class RevisionLifecycleConflictError(RuntimeError):
    """The revision changed while a lifecycle transition was being prepared."""


class RevisionBundle:
    """All governed records required to validate or rebuild one revision."""

    def __init__(
        self,
        document: DocumentRevision,
        chunks: list[Chunk],
        images: list[ImageAsset],
        tables: list[TableAsset],
    ) -> None:
        self.document = document
        self.chunks = chunks
        self.images = images
        self.tables = tables

    @property
    def counts(self) -> AffectedRecordCounts:
        return AffectedRecordCounts(
            documents=1,
            chunks=len(self.chunks),
            images=len(self.images),
            tables=len(self.tables),
            vectors=len(self.chunks),
        )


class KnowledgeDocumentRepository(Protocol):
    def list_documents(
        self,
        actor_scope: ActorScope,
        *,
        query: str | None,
        lifecycle: DocumentLifecycle | None,
        limit: int,
        offset: int,
    ) -> KnowledgeDocumentListResponse: ...

    def list_revisions(
        self,
        document_id: str,
        actor_scope: ActorScope,
    ) -> list[KnowledgeDocumentRevisionSummary]: ...

    def get_bundle(self, selector: DocumentRevisionSelector) -> RevisionBundle | None: ...

    def get_source_manifest(self, source_id: str, version: str) -> SourceManifest | None: ...

    def create_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord: ...

    def get_operation(self, operation_id: str) -> DocumentLifecycleOperationRecord | None: ...

    def get_operation_by_request_id(
        self,
        request_id: str,
    ) -> DocumentLifecycleOperationRecord | None: ...

    def save_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord: ...

    def claim_operation(self, operation_id: str, execution_id: str) -> bool: ...

    def release_operation(self, operation_id: str, execution_id: str) -> None: ...

    def block_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        actor_user_id: str,
        reason: str,
        expected_lifecycle: DocumentLifecycle | None,
    ) -> None: ...

    def publish_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        target_index_version: str,
    ) -> None: ...

    def append_audit(self, event: AuditEvent) -> AuditEvent: ...


def _clean(document: dict[str, object] | None) -> dict[str, object] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


def _clean_operation(document: dict[str, object] | None) -> dict[str, object] | None:
    cleaned = _clean(document)
    if cleaned is None:
        return None
    return {key: value for key, value in cleaned.items() if key != "execution_id"}


def _is_in_scope(document: DocumentRevision, actor_scope: ActorScope) -> bool:
    return "admin" in actor_scope.roles or (
        bool(actor_scope.access_scope_keys)
        and document.access_scope_key in actor_scope.access_scope_keys
    )


def _revision_summary(
    document: DocumentRevision,
    counts: AffectedRecordCounts,
) -> KnowledgeDocumentRevisionSummary:
    return KnowledgeDocumentRevisionSummary(
        document_id=document.document_id,
        revision=document.revision,
        title=document.title,
        document_type=document.document_type,
        approval_status=document.approval_status,
        lifecycle=document.lifecycle,
        effective_at=document.effective_at,
        expires_at=document.expires_at,
        source_id=document.source_id,
        source_manifest_version=document.source_manifest_version,
        dataset_version=document.dataset_version,
        source_uri=document.source_uri,
        source_license=document.source_license,
        source_license_status=document.source_license_status,
        redistribution_policy=document.redistribution_policy,
        access_scope_key=document.access_scope_key,
        counts=counts,
        created_at=document.created_at,
    )


def _document_list(
    documents: Sequence[DocumentRevision],
    *,
    query: str | None,
    lifecycle: DocumentLifecycle | None,
    limit: int,
    offset: int,
) -> KnowledgeDocumentListResponse:
    grouped: dict[str, list[DocumentRevision]] = defaultdict(list)
    normalized_query = (query or "").strip().casefold()
    for document in documents:
        if normalized_query and normalized_query not in (
            f"{document.document_id} {document.title} {document.document_type}"
        ).casefold():
            continue
        grouped[document.document_id].append(document)

    items: list[KnowledgeDocumentSummary] = []
    for document_id, revisions in grouped.items():
        revisions.sort(
            key=lambda item: (item.effective_at, item.created_at, item.revision),
            reverse=True,
        )
        if lifecycle is not None and not any(item.lifecycle is lifecycle for item in revisions):
            continue
        current = revisions[0]
        items.append(
            KnowledgeDocumentSummary(
                document_id=document_id,
                title=current.title,
                document_type=current.document_type,
                current_revision=current.revision,
                current_lifecycle=current.lifecycle,
                revision_count=len(revisions),
                source_id=current.source_id,
                dataset_version=current.dataset_version,
                updated_at=max(item.created_at for item in revisions),
            )
        )
    items.sort(key=lambda item: (item.updated_at, item.document_id), reverse=True)
    return KnowledgeDocumentListResponse(
        items=items[offset : offset + limit],
        total=len(items),
        limit=limit,
        offset=offset,
    )


def _operation_identity(operation: DocumentLifecycleOperationRecord) -> tuple[object, ...]:
    return (
        operation.action,
        operation.selector.document_id,
        operation.selector.revision,
        operation.actor_user_id,
        operation.reason,
        operation.target_index_version,
    )


class MongoKnowledgeDocumentRepository:
    """MongoDB authority for catalog visibility and lifecycle operation state."""

    def __init__(self, factory: StorageClientFactory, database_name: str) -> None:
        self._factory = factory
        self._database_name = database_name

    def list_documents(
        self,
        actor_scope: ActorScope,
        *,
        query: str | None,
        lifecycle: DocumentLifecycle | None,
        limit: int,
        offset: int,
    ) -> KnowledgeDocumentListResponse:
        selector: dict[str, object] = {}
        if "admin" not in actor_scope.roles:
            selector["access_scope_key"] = {"$in": actor_scope.access_scope_keys}
        if query and query.strip():
            pattern = re.escape(query.strip())
            selector["$or"] = [
                {"document_id": {"$regex": pattern, "$options": "i"}},
                {"title": {"$regex": pattern, "$options": "i"}},
                {"document_type": {"$regex": pattern, "$options": "i"}},
            ]
        with self._factory.mongodb() as client:
            records = list(
                client[self._database_name].document_catalog.find(selector, {"_id": 0})
            )
        documents = [DocumentRevision.model_validate(record) for record in records]
        return _document_list(
            documents,
            query=None,
            lifecycle=lifecycle,
            limit=limit,
            offset=offset,
        )

    def list_revisions(
        self,
        document_id: str,
        actor_scope: ActorScope,
    ) -> list[KnowledgeDocumentRevisionSummary]:
        selector: dict[str, object] = {"document_id": document_id}
        if "admin" not in actor_scope.roles:
            selector["access_scope_key"] = {"$in": actor_scope.access_scope_keys}
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            records = list(
                database.document_catalog.find(selector, {"_id": 0}).sort(
                    [("effective_at", -1), ("created_at", -1)]
                )
            )
            summaries = []
            for record in records:
                revision_selector = {
                    "document_id": record["document_id"],
                    "revision": record["revision"],
                }
                counts = AffectedRecordCounts(
                    documents=1,
                    chunks=database.chunk_catalog.count_documents(revision_selector),
                    images=database.image_assets.count_documents(revision_selector),
                    tables=database.table_assets.count_documents(revision_selector),
                    vectors=(
                        database.chunk_catalog.count_documents(revision_selector)
                        if record.get("lifecycle") == DocumentLifecycle.PUBLISHED.value
                        else 0
                    ),
                )
                summaries.append(
                    _revision_summary(DocumentRevision.model_validate(record), counts)
                )
        return summaries

    def get_bundle(self, selector: DocumentRevisionSelector) -> RevisionBundle | None:
        query = selector.model_dump(mode="python")
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            document_record = _clean(database.document_catalog.find_one(query))
            if document_record is None:
                return None
            chunks = list(database.chunk_catalog.find(query, {"_id": 0}))
            images = list(database.image_assets.find(query, {"_id": 0}))
            tables = list(database.table_assets.find(query, {"_id": 0}))
        return RevisionBundle(
            DocumentRevision.model_validate(document_record),
            [Chunk.model_validate(item) for item in chunks],
            [ImageAsset.model_validate(item) for item in images],
            [TableAsset.model_validate(item) for item in tables],
        )

    def get_source_manifest(self, source_id: str, version: str) -> SourceManifest | None:
        with self._factory.mongodb() as client:
            record = client[self._database_name].source_manifests.find_one(
                {"source_id": source_id, "manifest_version": version},
                {"_id": 0, "manifest_checksum": 0},
            )
        return SourceManifest.model_validate(record) if record else None

    def create_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        payload = operation.model_dump(mode="python")
        with self._factory.mongodb() as client:
            stored = client[
                self._database_name
            ].document_lifecycle_operations.find_one_and_update(
                {"request_id": operation.request_id},
                {"$setOnInsert": payload},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        existing = DocumentLifecycleOperationRecord.model_validate(_clean_operation(stored))
        if _operation_identity(existing) != _operation_identity(operation):
            raise LifecycleOperationRequestConflictError(operation.request_id)
        return existing

    def get_operation(self, operation_id: str) -> DocumentLifecycleOperationRecord | None:
        with self._factory.mongodb() as client:
            record = client[self._database_name].document_lifecycle_operations.find_one(
                {"operation_id": operation_id},
                {"_id": 0},
            )
        return (
            DocumentLifecycleOperationRecord.model_validate(_clean_operation(record))
            if record
            else None
        )

    def get_operation_by_request_id(
        self,
        request_id: str,
    ) -> DocumentLifecycleOperationRecord | None:
        with self._factory.mongodb() as client:
            record = client[self._database_name].document_lifecycle_operations.find_one(
                {"request_id": request_id},
                {"_id": 0},
            )
        return (
            DocumentLifecycleOperationRecord.model_validate(_clean_operation(record))
            if record
            else None
        )

    def save_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        with self._factory.mongodb() as client:
            result = client[self._database_name].document_lifecycle_operations.update_one(
                {"operation_id": operation.operation_id},
                {"$set": operation.model_dump(mode="python")},
            )
        if result.matched_count != 1:
            raise KeyError(operation.operation_id)
        return operation

    def claim_operation(self, operation_id: str, execution_id: str) -> bool:
        with self._factory.mongodb() as client:
            result = client[self._database_name].document_lifecycle_operations.update_one(
                {
                    "operation_id": operation_id,
                    "$or": [
                        {"execution_id": {"$exists": False}},
                        {"execution_id": execution_id},
                    ],
                },
                {"$set": {"execution_id": execution_id}},
            )
        return result.matched_count == 1

    def release_operation(self, operation_id: str, execution_id: str) -> None:
        with self._factory.mongodb() as client:
            client[self._database_name].document_lifecycle_operations.update_one(
                {"operation_id": operation_id, "execution_id": execution_id},
                {"$unset": {"execution_id": ""}},
            )

    def block_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        actor_user_id: str,
        reason: str,
        expected_lifecycle: DocumentLifecycle | None,
    ) -> None:
        query: dict[str, object] = selector.model_dump(mode="python")
        if expected_lifecycle is not None:
            query["lifecycle"] = expected_lifecycle.value
        now = datetime.now(UTC)
        values = {
            "lifecycle": DocumentLifecycle.WITHDRAWN.value,
            "lifecycle_updated_at": now,
            "last_lifecycle_operation_id": operation_id,
            "withdrawn_at": now,
            "withdrawn_by": actor_user_id,
            "withdrawn_reason": reason,
        }
        revision_query = selector.model_dump(mode="python")
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            result = database.document_catalog.update_one(query, {"$set": values})
            if result.matched_count != 1:
                raise RevisionLifecycleConflictError(
                    f"{selector.document_id}:{selector.revision}"
                )
            child_values = {
                "lifecycle": DocumentLifecycle.WITHDRAWN.value,
                "lifecycle_updated_at": now,
                "last_lifecycle_operation_id": operation_id,
            }
            for collection_name in _REVISION_COLLECTIONS:
                database[collection_name].update_many(
                    revision_query,
                    {"$set": child_values},
                )

    def publish_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        target_index_version: str,
    ) -> None:
        query = {
            **selector.model_dump(mode="python"),
            "lifecycle": DocumentLifecycle.WITHDRAWN.value,
        }
        now = datetime.now(UTC)
        values = {
            "lifecycle": DocumentLifecycle.PUBLISHED.value,
            "index_version": target_index_version,
            "lifecycle_updated_at": now,
            "last_lifecycle_operation_id": operation_id,
            "restored_at": now,
        }
        unset = {
            "withdrawn_at": "",
            "withdrawn_by": "",
            "withdrawn_reason": "",
        }
        revision_query = selector.model_dump(mode="python")
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            result = database.document_catalog.update_one(
                query,
                {"$set": values, "$unset": unset},
            )
            if result.matched_count != 1:
                raise RevisionLifecycleConflictError(
                    f"{selector.document_id}:{selector.revision}"
                )
            child_values: dict[str, object] = {
                "lifecycle": DocumentLifecycle.PUBLISHED.value,
                "lifecycle_updated_at": now,
                "last_lifecycle_operation_id": operation_id,
            }
            for collection_name in _REVISION_COLLECTIONS:
                collection_values = (
                    {**child_values, "index_version": target_index_version}
                    if collection_name == "chunk_catalog"
                    else child_values
                )
                database[collection_name].update_many(
                    revision_query,
                    {"$set": collection_values},
                )

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        with self._factory.mongodb() as client:
            client[self._database_name].audit_events.update_one(
                {"event_id": event.event_id},
                {"$setOnInsert": event.model_dump(mode="python")},
                upsert=True,
            )
        return event


class DemoKnowledgeDocumentRepository:
    """In-memory lifecycle repository used by tests and the standalone demo mode."""

    def __init__(self, store: DemoStore) -> None:
        self._store = store
        self._operations: dict[str, DocumentLifecycleOperationRecord] = {}
        self._request_ids: dict[str, str] = {}
        self._operation_claims: dict[str, str] = {}

    def list_documents(
        self,
        actor_scope: ActorScope,
        *,
        query: str | None,
        lifecycle: DocumentLifecycle | None,
        limit: int,
        offset: int,
    ) -> KnowledgeDocumentListResponse:
        documents = [
            item for item in self._store.documents.values() if _is_in_scope(item, actor_scope)
        ]
        return _document_list(
            documents,
            query=query,
            lifecycle=lifecycle,
            limit=limit,
            offset=offset,
        )

    def list_revisions(
        self,
        document_id: str,
        actor_scope: ActorScope,
    ) -> list[KnowledgeDocumentRevisionSummary]:
        summaries: list[KnowledgeDocumentRevisionSummary] = []
        for document in self._store.documents.values():
            if document.document_id != document_id or not _is_in_scope(document, actor_scope):
                continue
            bundle = self.get_bundle(
                DocumentRevisionSelector(
                    document_id=document.document_id,
                    revision=document.revision,
                )
            )
            if bundle:
                counts = bundle.counts.model_copy(
                    update={
                        "vectors": (
                            len(bundle.chunks)
                            if document.lifecycle is DocumentLifecycle.PUBLISHED
                            else 0
                        )
                    }
                )
                summaries.append(_revision_summary(document, counts))
        summaries.sort(key=lambda item: (item.effective_at, item.created_at), reverse=True)
        return summaries

    def get_bundle(self, selector: DocumentRevisionSelector) -> RevisionBundle | None:
        document = self._store.documents.get((selector.document_id, selector.revision))
        if document is None:
            return None
        return RevisionBundle(
            document,
            [
                item
                for item in self._store.chunks.values()
                if item.document_id == selector.document_id and item.revision == selector.revision
            ],
            [
                item
                for item in self._store.images.values()
                if item.document_id == selector.document_id and item.revision == selector.revision
            ],
            [
                item
                for item in self._store.tables.values()
                if item.document_id == selector.document_id and item.revision == selector.revision
            ],
        )

    def get_source_manifest(self, source_id: str, version: str) -> SourceManifest | None:
        return None

    def create_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        existing_id = self._request_ids.get(operation.request_id)
        if existing_id:
            existing = self._operations[existing_id]
            if _operation_identity(existing) != _operation_identity(operation):
                raise LifecycleOperationRequestConflictError(operation.request_id)
            return existing
        self._operations[operation.operation_id] = operation
        self._request_ids[operation.request_id] = operation.operation_id
        return operation

    def get_operation(self, operation_id: str) -> DocumentLifecycleOperationRecord | None:
        return self._operations.get(operation_id)

    def get_operation_by_request_id(
        self,
        request_id: str,
    ) -> DocumentLifecycleOperationRecord | None:
        operation_id = self._request_ids.get(request_id)
        return self._operations.get(operation_id) if operation_id else None

    def save_operation(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        if operation.operation_id not in self._operations:
            raise KeyError(operation.operation_id)
        self._operations[operation.operation_id] = operation
        return operation

    def claim_operation(self, operation_id: str, execution_id: str) -> bool:
        if operation_id not in self._operations:
            raise KeyError(operation_id)
        owner = self._operation_claims.get(operation_id)
        if owner not in {None, execution_id}:
            return False
        self._operation_claims[operation_id] = execution_id
        return True

    def release_operation(self, operation_id: str, execution_id: str) -> None:
        if self._operation_claims.get(operation_id) == execution_id:
            self._operation_claims.pop(operation_id, None)

    def block_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        actor_user_id: str,
        reason: str,
        expected_lifecycle: DocumentLifecycle | None,
    ) -> None:
        bundle = self.get_bundle(selector)
        if bundle is None or (
            expected_lifecycle is not None
            and bundle.document.lifecycle is not expected_lifecycle
        ):
            raise RevisionLifecycleConflictError(
                f"{selector.document_id}:{selector.revision}"
            )
        bundle.document.lifecycle = DocumentLifecycle.WITHDRAWN
        for record in [*bundle.chunks, *bundle.images, *bundle.tables]:
            record.lifecycle = DocumentLifecycle.WITHDRAWN

    def publish_revision(
        self,
        selector: DocumentRevisionSelector,
        *,
        operation_id: str,
        target_index_version: str,
    ) -> None:
        bundle = self.get_bundle(selector)
        if bundle is None or bundle.document.lifecycle is not DocumentLifecycle.WITHDRAWN:
            raise RevisionLifecycleConflictError(
                f"{selector.document_id}:{selector.revision}"
            )
        bundle.document.lifecycle = DocumentLifecycle.PUBLISHED
        bundle.document.index_version = target_index_version
        for chunk in bundle.chunks:
            chunk.lifecycle = DocumentLifecycle.PUBLISHED
            chunk.index_version = target_index_version
        for record in [*bundle.images, *bundle.tables]:
            record.lifecycle = DocumentLifecycle.PUBLISHED

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        return self._store.append_audit(event)
