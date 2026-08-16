"""Controlled withdrawal and restoration of governed knowledge revisions."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    ApprovalStatus,
    AuditEvent,
    Chunk,
    CompensationStatus,
    DocumentLifecycle,
    DocumentLifecycleAction,
    DocumentLifecycleOperationRecord,
    DocumentLifecycleOperationStatus,
    DocumentRevisionSelector,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRevisionSummary,
    ObjectRef,
    RestoreDocumentRevisionRequest,
    WithdrawDocumentRevisionRequest,
)
from semikb.rag_ingestion.publication_governance import (
    PublicationGovernanceError,
    validate_publication_governance,
)
from semikb.rag_retrieval.encoders import HybridEmbedding, HybridEncoder
from semikb.storage.knowledge_documents import (
    KnowledgeDocumentRepository,
    RevisionBundle,
)


class LifecycleValidationError(ValueError):
    """A revision cannot safely enter the requested lifecycle state."""


class LifecycleStateConflictError(RuntimeError):
    """A revision is not in the state required for the requested action."""


class ArtifactReader(Protocol):
    def load_object(self, object_ref: ObjectRef) -> bytes: ...


class VectorProjectionRepository(Protocol):
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[HybridEmbedding],
        *,
        lifecycle: DocumentLifecycle,
    ) -> None: ...

    def delete_chunks(self, index_version: str, chunk_ids: Sequence[str]) -> None: ...

    def verify_chunks_absent(self, index_version: str, chunk_ids: Sequence[str]) -> None: ...

    def activate_alias(self, index_version: str) -> None: ...


class NoopVectorProjectionRepository:
    """Demo-mode projection; in-memory retrieval reads lifecycle state directly."""

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[HybridEmbedding],
        *,
        lifecycle: DocumentLifecycle,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every restored chunk requires one embedding.")

    def delete_chunks(self, index_version: str, chunk_ids: Sequence[str]) -> None:
        return None

    def verify_chunks_absent(self, index_version: str, chunk_ids: Sequence[str]) -> None:
        return None

    def activate_alias(self, index_version: str) -> None:
        return None


_TERMINAL_STATUSES = {
    DocumentLifecycleOperationStatus.WITHDRAWN,
    DocumentLifecycleOperationStatus.RESTORED,
    DocumentLifecycleOperationStatus.FAILED,
}


class KnowledgeDocumentLifecycleService:
    """Keeps MongoDB authoritative while treating Milvus as a rebuildable projection."""

    def __init__(
        self,
        settings: Settings,
        repository: KnowledgeDocumentRepository,
        vectors: VectorProjectionRepository,
        artifacts: ArtifactReader,
        encoder: HybridEncoder,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.vectors = vectors
        self.artifacts = artifacts
        self.encoder = encoder

    def list_documents(
        self,
        actor_scope: ActorScope,
        *,
        query: str | None = None,
        lifecycle: DocumentLifecycle | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> KnowledgeDocumentListResponse:
        return self.repository.list_documents(
            actor_scope,
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
        return self.repository.list_revisions(document_id, actor_scope)

    def get_operation(self, operation_id: str) -> DocumentLifecycleOperationRecord | None:
        return self.repository.get_operation(operation_id)

    def request_withdrawal(
        self,
        selector: DocumentRevisionSelector,
        request: WithdrawDocumentRevisionRequest,
        actor_scope: ActorScope,
    ) -> DocumentLifecycleOperationRecord:
        existing = self.repository.get_operation_by_request_id(request.request_id)
        if existing is not None:
            return self._validate_idempotent_request(
                existing,
                action=DocumentLifecycleAction.WITHDRAW,
                selector=selector,
                actor_user_id=actor_scope.user_id,
                reason=request.reason,
                target_index_version=None,
            )
        bundle = self._authorized_bundle(selector, actor_scope)
        if bundle.document.lifecycle is not DocumentLifecycle.PUBLISHED:
            raise LifecycleStateConflictError(
                f"Only a published revision can be withdrawn; current state is "
                f"{bundle.document.lifecycle.value}."
            )
        candidate = DocumentLifecycleOperationRecord(
            request_id=request.request_id,
            action=DocumentLifecycleAction.WITHDRAW,
            status=DocumentLifecycleOperationStatus.REQUESTED,
            selector=selector,
            actor_user_id=actor_scope.user_id,
            reason=request.reason,
            before_lifecycle=bundle.document.lifecycle,
            affected=bundle.counts,
        )
        operation = self.repository.create_operation(candidate)
        if operation.operation_id != candidate.operation_id:
            return self._validate_idempotent_request(
                operation,
                action=DocumentLifecycleAction.WITHDRAW,
                selector=selector,
                actor_user_id=actor_scope.user_id,
                reason=request.reason,
                target_index_version=None,
            )
        self._audit(operation, "requested")
        operation = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.BLOCKING,
        )
        try:
            self.repository.block_revision(
                selector,
                operation_id=operation.operation_id,
                actor_user_id=actor_scope.user_id,
                reason=request.reason,
                expected_lifecycle=DocumentLifecycle.PUBLISHED,
            )
        except Exception:
            self._save(
                operation,
                status=DocumentLifecycleOperationStatus.FAILED,
                warning_code="MONGODB_BLOCKING_FAILED",
                completed=True,
            )
            raise
        return self._save(
            operation,
            status=DocumentLifecycleOperationStatus.VECTOR_CLEANUP,
            after_lifecycle=DocumentLifecycle.WITHDRAWN,
        )

    def request_restore(
        self,
        selector: DocumentRevisionSelector,
        request: RestoreDocumentRevisionRequest,
        actor_scope: ActorScope,
    ) -> DocumentLifecycleOperationRecord:
        target_index_version = request.target_index_version or self.settings.milvus_index_version
        if target_index_version != self.settings.milvus_index_version:
            raise LifecycleValidationError(
                "Restoration must target the configured active index version."
            )
        existing = self.repository.get_operation_by_request_id(request.request_id)
        if existing is not None:
            return self._validate_idempotent_request(
                existing,
                action=DocumentLifecycleAction.RESTORE,
                selector=selector,
                actor_user_id=actor_scope.user_id,
                reason=request.reason,
                target_index_version=target_index_version,
            )
        bundle = self._authorized_bundle(selector, actor_scope)
        if bundle.document.lifecycle is not DocumentLifecycle.WITHDRAWN:
            raise LifecycleStateConflictError(
                f"Only a withdrawn revision can be restored; current state is "
                f"{bundle.document.lifecycle.value}."
            )
        candidate = DocumentLifecycleOperationRecord(
            request_id=request.request_id,
            action=DocumentLifecycleAction.RESTORE,
            status=DocumentLifecycleOperationStatus.RESTORE_VALIDATING,
            selector=selector,
            actor_user_id=actor_scope.user_id,
            reason=request.reason,
            before_lifecycle=bundle.document.lifecycle,
            target_index_version=target_index_version,
            affected=bundle.counts,
        )
        operation = self.repository.create_operation(candidate)
        if operation.operation_id != candidate.operation_id:
            return self._validate_idempotent_request(
                operation,
                action=DocumentLifecycleAction.RESTORE,
                selector=selector,
                actor_user_id=actor_scope.user_id,
                reason=request.reason,
                target_index_version=target_index_version,
            )
        self._audit(operation, "requested")
        return operation

    def process(
        self,
        operation_id: str,
        *,
        execution_id: str | None = None,
    ) -> DocumentLifecycleOperationRecord:
        operation = self.repository.get_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation.status in _TERMINAL_STATUSES:
            return operation
        owner = execution_id or f"inline-{uuid.uuid4().hex}"
        if not self.repository.claim_operation(operation_id, owner):
            current = self.repository.get_operation(operation_id)
            if current is None:
                raise KeyError(operation_id)
            return current
        try:
            operation = self.repository.get_operation(operation_id)
            if operation is None:
                raise KeyError(operation_id)
            if operation.status in _TERMINAL_STATUSES:
                return operation
            if operation.action is DocumentLifecycleAction.WITHDRAW:
                return self._process_withdrawal(operation)
            return self._process_restore(operation)
        finally:
            self.repository.release_operation(operation_id, owner)

    def prepare_retry(self, operation_id: str) -> DocumentLifecycleOperationRecord:
        operation = self.repository.get_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation.status not in {
            DocumentLifecycleOperationStatus.COMPENSATION_REQUIRED,
            DocumentLifecycleOperationStatus.FAILED,
        }:
            return operation
        status = (
            DocumentLifecycleOperationStatus.VECTOR_CLEANUP
            if operation.action is DocumentLifecycleAction.WITHDRAW
            else DocumentLifecycleOperationStatus.RESTORE_VALIDATING
        )
        return self._save(
            operation,
            status=status,
            compensation_status=CompensationStatus.RUNNING,
        )

    def mark_dispatch_failed(self, operation_id: str) -> DocumentLifecycleOperationRecord:
        operation = self.repository.get_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        return self._save(
            operation,
            status=DocumentLifecycleOperationStatus.COMPENSATION_REQUIRED,
            compensation_status=CompensationStatus.PENDING,
            warning_code="TASK_QUEUE_UNAVAILABLE",
        )

    def _process_withdrawal(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        bundle = self.repository.get_bundle(operation.selector)
        if bundle is None:
            return self._fail_validation(operation, "REVISION_NOT_FOUND")
        if bundle.document.lifecycle is not DocumentLifecycle.WITHDRAWN:
            return self._fail_validation(operation, "REVISION_NOT_BLOCKED")
        chunk_ids = [chunk.chunk_id for chunk in bundle.chunks]
        try:
            self.vectors.delete_chunks(bundle.document.index_version, chunk_ids)
            self.vectors.verify_chunks_absent(bundle.document.index_version, chunk_ids)
        except Exception as exc:
            return self._require_compensation(operation, exc)
        completed = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.WITHDRAWN,
            after_lifecycle=DocumentLifecycle.WITHDRAWN,
            compensation_status=(
                CompensationStatus.COMPLETED
                if operation.compensation_status is not CompensationStatus.NOT_REQUIRED
                else CompensationStatus.NOT_REQUIRED
            ),
            completed=True,
        )
        self._audit(completed, "completed")
        return completed

    def _process_restore(
        self,
        operation: DocumentLifecycleOperationRecord,
    ) -> DocumentLifecycleOperationRecord:
        bundle = self.repository.get_bundle(operation.selector)
        if bundle is None:
            return self._fail_validation(operation, "REVISION_NOT_FOUND")
        try:
            self._validate_restore_bundle(bundle, operation)
        except LifecycleValidationError as exc:
            return self._fail_validation(operation, self._warning_code(exc))

        target_index_version = operation.target_index_version or self.settings.milvus_index_version
        operation = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.RESTORE_INDEXING,
        )
        chunks = [
            chunk.model_copy(
                update={
                    "index_version": target_index_version,
                    "lifecycle": DocumentLifecycle.STAGED,
                }
            )
            for chunk in bundle.chunks
        ]
        try:
            embeddings = self._encode(chunks)
            self.vectors.upsert_chunks(
                chunks,
                embeddings,
                lifecycle=DocumentLifecycle.STAGED,
            )
            self.repository.publish_revision(
                operation.selector,
                operation_id=operation.operation_id,
                target_index_version=target_index_version,
            )
            self.vectors.upsert_chunks(
                chunks,
                embeddings,
                lifecycle=DocumentLifecycle.PUBLISHED,
            )
            self.vectors.activate_alias(target_index_version)
        except Exception as exc:
            self._reblock_after_restore_failure(operation, target_index_version, chunks)
            return self._require_compensation(operation, exc)

        completed = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.RESTORED,
            after_lifecycle=DocumentLifecycle.PUBLISHED,
            compensation_status=(
                CompensationStatus.COMPLETED
                if operation.compensation_status is not CompensationStatus.NOT_REQUIRED
                else CompensationStatus.NOT_REQUIRED
            ),
            completed=True,
        )
        self._audit(completed, "completed")
        return completed

    def _validate_restore_bundle(
        self,
        bundle: RevisionBundle,
        operation: DocumentLifecycleOperationRecord,
    ) -> None:
        document = bundle.document
        if document.lifecycle is not DocumentLifecycle.WITHDRAWN:
            raise LifecycleValidationError("revision_not_withdrawn")
        if document.approval_status is not ApprovalStatus.APPROVED:
            raise LifecycleValidationError("revision_not_approved")
        now = datetime.now(UTC)
        effective_at = self._utc(document.effective_at)
        if effective_at > now:
            raise LifecycleValidationError("revision_not_yet_effective")
        if document.expires_at and self._utc(document.expires_at) <= now:
            raise LifecycleValidationError("revision_expired")
        if not bundle.chunks:
            raise LifecycleValidationError("revision_has_no_chunks")
        if operation.target_index_version != self.settings.milvus_index_version:
            raise LifecycleValidationError("target_index_version_inactive")
        manifest = (
            self.repository.get_source_manifest(
                document.source_id,
                document.source_manifest_version,
            )
            if document.source_id and document.source_manifest_version
            else None
        )
        try:
            validate_publication_governance(
                document,
                bundle.chunks,
                bundle.images,
                bundle.tables,
                manifest,
            )
        except PublicationGovernanceError as exc:
            raise LifecycleValidationError(exc.code) from exc

        references = [document.source_ref]
        if document.parsed_ref is not None:
            references.append(document.parsed_ref)
        references.extend(image.object_ref for image in bundle.images)
        references.extend(table.object_ref for table in bundle.tables)
        for reference in references:
            self._validate_object(reference)
        source_content = self.artifacts.load_object(document.source_ref)
        if hashlib.sha256(source_content).hexdigest() != document.source_hash:
            raise LifecycleValidationError("source_hash_mismatch")

    def _validate_object(self, object_ref: ObjectRef) -> None:
        try:
            content = self.artifacts.load_object(object_ref)
        except Exception as exc:
            raise LifecycleValidationError("retained_object_missing") from exc
        if hashlib.sha256(content).hexdigest() != object_ref.sha256:
            raise LifecycleValidationError("retained_object_hash_mismatch")

    def _reblock_after_restore_failure(
        self,
        operation: DocumentLifecycleOperationRecord,
        index_version: str,
        chunks: Sequence[Chunk],
    ) -> None:
        try:
            self.repository.block_revision(
                operation.selector,
                operation_id=operation.operation_id,
                actor_user_id=operation.actor_user_id,
                reason=operation.reason,
                expected_lifecycle=None,
            )
        except Exception:
            pass
        try:
            self.vectors.delete_chunks(index_version, [chunk.chunk_id for chunk in chunks])
            self.vectors.verify_chunks_absent(
                index_version,
                [chunk.chunk_id for chunk in chunks],
            )
        except Exception:
            pass

    def _authorized_bundle(
        self,
        selector: DocumentRevisionSelector,
        actor_scope: ActorScope,
    ) -> RevisionBundle:
        bundle = self.repository.get_bundle(selector)
        if bundle is None:
            raise KeyError(f"{selector.document_id}:{selector.revision}")
        if "admin" not in actor_scope.roles and (
            bundle.document.access_scope_key not in actor_scope.access_scope_keys
        ):
            raise PermissionError(f"{selector.document_id}:{selector.revision}")
        return bundle

    def _encode(self, chunks: Sequence[Chunk]) -> list[HybridEmbedding]:
        embeddings: list[HybridEmbedding] = []
        batch_size = self.settings.embedding_batch_size
        texts = [chunk.chunk_text for chunk in chunks]
        for offset in range(0, len(texts), batch_size):
            embeddings.extend(self.encoder.encode(texts[offset : offset + batch_size]))
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding encoder returned an unexpected row count.")
        return embeddings

    def _validate_idempotent_request(
        self,
        existing: DocumentLifecycleOperationRecord,
        *,
        action: DocumentLifecycleAction,
        selector: DocumentRevisionSelector,
        actor_user_id: str,
        reason: str,
        target_index_version: str | None,
    ) -> DocumentLifecycleOperationRecord:
        identity = (
            existing.action,
            existing.selector,
            existing.actor_user_id,
            existing.reason,
            existing.target_index_version,
        )
        expected = (action, selector, actor_user_id, reason, target_index_version)
        if identity != expected:
            from semikb.storage.knowledge_documents import (
                LifecycleOperationRequestConflictError,
            )

            raise LifecycleOperationRequestConflictError(existing.request_id)
        return existing

    def _save(
        self,
        operation: DocumentLifecycleOperationRecord,
        *,
        status: DocumentLifecycleOperationStatus,
        after_lifecycle: DocumentLifecycle | None = None,
        compensation_status: CompensationStatus | None = None,
        warning_code: str | None = None,
        completed: bool = False,
    ) -> DocumentLifecycleOperationRecord:
        warnings = list(operation.warning_codes)
        if warning_code and warning_code not in warnings:
            warnings.append(warning_code)
        now = datetime.now(UTC)
        update: dict[str, object] = {
            "status": status,
            "updated_at": now,
            "warning_codes": warnings,
        }
        if after_lifecycle is not None:
            update["after_lifecycle"] = after_lifecycle
        if compensation_status is not None:
            update["compensation_status"] = compensation_status
        if completed:
            update["completed_at"] = now
        return self.repository.save_operation(operation.model_copy(update=update))

    def _require_compensation(
        self,
        operation: DocumentLifecycleOperationRecord,
        exc: Exception,
    ) -> DocumentLifecycleOperationRecord:
        failed = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.COMPENSATION_REQUIRED,
            compensation_status=CompensationStatus.PENDING,
            warning_code=self._warning_code(exc),
        )
        self._audit(failed, "compensation_required")
        return failed

    def _fail_validation(
        self,
        operation: DocumentLifecycleOperationRecord,
        warning_code: str,
    ) -> DocumentLifecycleOperationRecord:
        failed = self._save(
            operation,
            status=DocumentLifecycleOperationStatus.FAILED,
            warning_code=warning_code,
            completed=True,
        )
        self._audit(failed, "failed")
        return failed

    def _audit(self, operation: DocumentLifecycleOperationRecord, phase: str) -> None:
        self.repository.append_audit(
            AuditEvent(
                event_type=f"knowledge_document_{operation.action.value}_{phase}",
                actor_user_id=operation.actor_user_id,
                details={
                    "resource_type": "knowledge_document_revision",
                    "resource_id": operation.selector.document_id,
                    "revision": operation.selector.revision,
                    "operation_id": operation.operation_id,
                    "request_id": operation.request_id,
                    "action": operation.action.value,
                    "status": operation.status.value,
                    "reason": operation.reason,
                    "affected": operation.affected.model_dump(mode="json"),
                    "compensation_status": operation.compensation_status.value,
                },
            )
        )

    @staticmethod
    def _warning_code(exc: Exception) -> str:
        raw = str(exc).strip()
        if raw and raw.replace("_", "").isalnum() and raw.lower() == raw:
            return raw.upper()
        return type(exc).__name__.upper()

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
