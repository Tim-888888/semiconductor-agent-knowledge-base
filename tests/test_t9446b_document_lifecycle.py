from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from semikb.api.main import app
from semikb.bootstrap import ApplicationContainer, get_container
from semikb.config import Settings, get_settings
from semikb.contracts.models import (
    ActorScope,
    CompensationStatus,
    DocumentLifecycle,
    DocumentLifecycleOperationRecord,
    DocumentLifecycleOperationStatus,
    DocumentRevisionSelector,
    RestoreDocumentRevisionRequest,
    WithdrawDocumentRevisionRequest,
)
from semikb.rag_ingestion.document_lifecycle import (
    LifecycleValidationError,
    NoopVectorProjectionRepository,
)
from semikb.storage.knowledge_documents import (
    LifecycleOperationRequestConflictError,
    _clean_operation,
)


class RecordingVectors(NoopVectorProjectionRepository):
    def __init__(self) -> None:
        self.fail_delete = False
        self.deleted: list[tuple[str, tuple[str, ...]]] = []
        self.upserted: list[tuple[str, tuple[str, ...]]] = []

    def delete_chunks(self, index_version, chunk_ids) -> None:
        if self.fail_delete:
            raise ConnectionError("milvus unavailable")
        self.deleted.append((index_version, tuple(chunk_ids)))

    def upsert_chunks(self, chunks, embeddings, *, lifecycle) -> None:
        super().upsert_chunks(chunks, embeddings, lifecycle=lifecycle)
        self.upserted.append((lifecycle.value, tuple(chunk.chunk_id for chunk in chunks)))


class BlockingRestoreVectors(RecordingVectors):
    def __init__(self) -> None:
        super().__init__()
        self.restore_started = threading.Event()
        self.allow_restore = threading.Event()

    def upsert_chunks(self, chunks, embeddings, *, lifecycle) -> None:
        if lifecycle is DocumentLifecycle.STAGED:
            self.restore_started.set()
            if not self.allow_restore.wait(timeout=5):
                raise TimeoutError("test did not release staged restore")
        super().upsert_chunks(chunks, embeddings, lifecycle=lifecycle)


@pytest.fixture
def lifecycle_container() -> ApplicationContainer:
    settings = Settings(
        _env_file=None,
        demo_mode=True,
        milvus_index_version="v4",
    )
    container = ApplicationContainer(settings)
    container.seed_demo_data()
    return container


def _admin() -> ActorScope:
    return ActorScope(
        user_id="knowledge_admin",
        roles=["knowledge_admin"],
        access_scope_keys=["demo_engineering"],
        fabs=["FAB-01"],
        products=["P-ALPHA"],
        tool_ids=["ETCH-03"],
    )


def _selector() -> DocumentRevisionSelector:
    return DocumentRevisionSelector(document_id="CASE-FA-2026-004", revision="R1")


def test_mongo_execution_claim_is_not_exposed_to_operation_contract(
    lifecycle_container: ApplicationContainer,
) -> None:
    operation = lifecycle_container.knowledge_documents.request_withdrawal(
        _selector(),
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-private-execution-claim",
            reason="验证 Worker 私有占用字段不会污染公共操作契约。",
        ),
        _admin(),
    )
    stored = operation.model_dump(mode="python") | {
        "_id": "mongo-object-id",
        "execution_id": "celery-task-id",
    }

    parsed = DocumentLifecycleOperationRecord.model_validate(_clean_operation(stored))

    assert parsed == operation


def test_withdrawal_blocks_retrieval_and_asset_then_restore_republishes(
    lifecycle_container: ApplicationContainer,
) -> None:
    service = lifecycle_container.knowledge_documents
    vectors = RecordingVectors()
    service.vectors = vectors
    actor = _admin()
    selector = _selector()

    assert lifecycle_container.retrieval.asset_access(
        "IMG-FA-ETCH-03-2026-004",
        actor,
    )["image_id"] == "IMG-FA-ETCH-03-2026-004"
    requested = service.request_withdrawal(
        selector,
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-case-004-r1",
            reason="演示语料误入库，需要受控下架。",
        ),
        actor,
    )
    assert requested.status is DocumentLifecycleOperationStatus.VECTOR_CLEANUP
    blocked = lifecycle_container.store.get_document(selector.document_id, selector.revision)
    assert blocked and blocked.lifecycle is DocumentLifecycle.WITHDRAWN

    completed = service.process(requested.operation_id)
    assert completed.status is DocumentLifecycleOperationStatus.WITHDRAWN
    assert completed.compensation_status is CompensationStatus.NOT_REQUIRED
    assert vectors.deleted and vectors.deleted[0][0] == "v4"
    evidence, _ = lifecycle_container.retrieval.search("ETCh-03 边缘环状缺陷", actor)
    assert all(chunk.document_id != selector.document_id for chunk in evidence)
    with pytest.raises(PermissionError):
        lifecycle_container.retrieval.asset_access("IMG-FA-ETCH-03-2026-004", actor)

    idempotent = service.request_withdrawal(
        selector,
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-case-004-r1",
            reason="演示语料误入库，需要受控下架。",
        ),
        actor,
    )
    assert idempotent.operation_id == completed.operation_id
    with pytest.raises(LifecycleOperationRequestConflictError):
        service.request_withdrawal(
            selector,
            WithdrawDocumentRevisionRequest(
                request_id="withdraw-case-004-r1",
                reason="同一请求编号不能改成另一个原因。",
            ),
            actor,
        )

    restore = service.request_restore(
        selector,
        RestoreDocumentRevisionRequest(
            request_id="restore-case-004-r1",
            reason="已核验来源与内容，批准恢复演示。",
            target_index_version="v4",
        ),
        actor,
    )
    restored = service.process(restore.operation_id)
    assert restored.status is DocumentLifecycleOperationStatus.RESTORED
    document = lifecycle_container.store.get_document(selector.document_id, selector.revision)
    assert document and document.lifecycle is DocumentLifecycle.PUBLISHED
    assert [stage for stage, _ in vectors.upserted] == ["staged", "published"]
    assert lifecycle_container.retrieval.asset_access(
        "IMG-FA-ETCH-03-2026-004",
        actor,
    )["image_id"] == "IMG-FA-ETCH-03-2026-004"


def test_vector_cleanup_failure_never_reexposes_withdrawn_content(
    lifecycle_container: ApplicationContainer,
) -> None:
    service = lifecycle_container.knowledge_documents
    vectors = RecordingVectors()
    vectors.fail_delete = True
    service.vectors = vectors
    actor = _admin()
    selector = _selector()
    operation = service.request_withdrawal(
        selector,
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-cleanup-failure",
            reason="验证向量清理失败时仍然不可检索。",
        ),
        actor,
    )

    pending = service.process(operation.operation_id)
    assert pending.status is DocumentLifecycleOperationStatus.COMPENSATION_REQUIRED
    assert pending.compensation_status is CompensationStatus.PENDING
    evidence, _ = lifecycle_container.retrieval.search("边缘环状缺陷", actor)
    assert all(chunk.document_id != selector.document_id for chunk in evidence)

    vectors.fail_delete = False
    service.prepare_retry(operation.operation_id)
    completed = service.process(operation.operation_id)
    assert completed.status is DocumentLifecycleOperationStatus.WITHDRAWN
    assert completed.compensation_status is CompensationStatus.COMPLETED


def test_restore_fails_closed_when_retained_source_hash_changes(
    lifecycle_container: ApplicationContainer,
) -> None:
    service = lifecycle_container.knowledge_documents
    actor = _admin()
    selector = _selector()
    withdrawal = service.request_withdrawal(
        selector,
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-before-hash-test",
            reason="为恢复完整性校验准备下架状态。",
        ),
        actor,
    )
    service.process(withdrawal.operation_id)
    document = lifecycle_container.store.get_document(selector.document_id, selector.revision)
    assert document is not None
    key = (document.source_ref.bucket, document.source_ref.object_key)
    lifecycle_container.store.objects[key] = b"tampered-source"

    restore = service.request_restore(
        selector,
        RestoreDocumentRevisionRequest(
            request_id="restore-with-bad-hash",
            reason="尝试恢复但原件完整性已经被破坏。",
        ),
        actor,
    )
    failed = service.process(restore.operation_id)
    assert failed.status is DocumentLifecycleOperationStatus.FAILED
    assert "RETAINED_OBJECT_HASH_MISMATCH" in failed.warning_codes
    assert document.lifecycle is DocumentLifecycle.WITHDRAWN


def test_duplicate_restore_delivery_cannot_publish_concurrently(
    lifecycle_container: ApplicationContainer,
) -> None:
    service = lifecycle_container.knowledge_documents
    vectors = BlockingRestoreVectors()
    service.vectors = vectors
    actor = _admin()
    selector = _selector()
    withdrawal = service.request_withdrawal(
        selector,
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-before-duplicate-restore",
            reason="为重复 Worker 投递回归测试准备下架状态。",
        ),
        actor,
    )
    service.process(withdrawal.operation_id, execution_id="withdraw-worker")
    restore = service.request_restore(
        selector,
        RestoreDocumentRevisionRequest(
            request_id="duplicate-restore-delivery",
            reason="验证两个 Worker 不会同时发布同一 revision。",
        ),
        actor,
    )
    completed: list[DocumentLifecycleOperationStatus] = []

    def run_restore() -> None:
        operation = service.process(restore.operation_id, execution_id="worker-a")
        completed.append(operation.status)

    thread = threading.Thread(target=run_restore)
    thread.start()
    assert vectors.restore_started.wait(timeout=5)
    duplicate = service.process(restore.operation_id, execution_id="worker-b")
    assert duplicate.status is DocumentLifecycleOperationStatus.RESTORE_INDEXING
    vectors.allow_restore.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert completed == [DocumentLifecycleOperationStatus.RESTORED]
    assert [stage for stage, _ in vectors.upserted] == ["staged", "published"]


def test_scope_and_index_version_are_checked_before_restore(
    lifecycle_container: ApplicationContainer,
) -> None:
    service = lifecycle_container.knowledge_documents
    outsider = ActorScope(
        user_id="other_admin",
        roles=["knowledge_admin"],
        access_scope_keys=["other_scope"],
    )
    with pytest.raises(PermissionError):
        service.request_withdrawal(
            _selector(),
            WithdrawDocumentRevisionRequest(
                request_id="withdraw-out-of-scope",
                reason="越权管理员不应该能够执行这个操作。",
            ),
            outsider,
        )

    actor = _admin()
    withdrawal = service.request_withdrawal(
        _selector(),
        WithdrawDocumentRevisionRequest(
            request_id="withdraw-before-index-check",
            reason="为目标索引版本校验准备下架状态。",
        ),
        actor,
    )
    service.process(withdrawal.operation_id)
    with pytest.raises(LifecycleValidationError, match="active index version"):
        service.request_restore(
            _selector(),
            RestoreDocumentRevisionRequest(
                request_id="restore-wrong-index",
                reason="不允许恢复到非活动索引版本。",
                target_index_version="v99",
            ),
            actor,
        )


def test_document_management_api_enforces_role_and_runs_demo_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    get_settings.cache_clear()
    get_container.cache_clear()
    client = TestClient(app)

    def token_for(user_id: str, roles: list[str]) -> dict[str, str]:
        response = client.post(
            "/api/v1/auth/demo-token",
            json={
                "user_id": user_id,
                "roles": roles,
                "access_scope_keys": ["demo_engineering"],
                "fabs": ["FAB-01"],
                "products": ["P-ALPHA"],
                "tool_ids": ["ETCH-03"],
            },
        )
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    try:
        engineer_headers = token_for("engineer", ["engineer"])
        assert client.get(
            "/api/v1/knowledge-documents",
            headers=engineer_headers,
        ).status_code == 403

        admin_headers = token_for("knowledge_admin", ["knowledge_admin"])
        listing = client.get("/api/v1/knowledge-documents", headers=admin_headers)
        assert listing.status_code == 200
        assert listing.json()["total"] >= 5

        withdrawal = client.post(
            "/api/v1/knowledge-documents/CASE-FA-2026-004/revisions/R1/withdraw",
            json={
                "request_id": "api-withdraw-case-004",
                "reason": "API 验收需要验证受控下架闭环。",
            },
            headers=admin_headers,
        )
        assert withdrawal.status_code == 202
        assert withdrawal.json()["status"] == "withdrawn"
        operation_id = withdrawal.json()["operation_id"]
        operation = client.get(
            f"/api/v1/knowledge-document-operations/{operation_id}",
            headers=admin_headers,
        )
        assert operation.status_code == 200

        revisions = client.get(
            "/api/v1/knowledge-documents/CASE-FA-2026-004/revisions",
            headers=admin_headers,
        )
        assert revisions.json()[0]["lifecycle"] == "withdrawn"

        restore = client.post(
            "/api/v1/knowledge-documents/CASE-FA-2026-004/revisions/R1/restore",
            json={
                "request_id": "api-restore-case-004",
                "reason": "API 验收已经完成，恢复受控文档。",
            },
            headers=admin_headers,
        )
        assert restore.status_code == 202
        assert restore.json()["status"] == "restored"
    finally:
        get_container.cache_clear()
        get_settings.cache_clear()


def test_openapi_and_typescript_publish_lifecycle_contracts() -> None:
    openapi = app.openapi()
    paths = openapi["paths"]
    assert "/api/v1/knowledge-documents" in paths
    assert "/api/v1/knowledge-documents/{document_id}/revisions" in paths
    withdraw = paths[
        "/api/v1/knowledge-documents/{document_id}/revisions/{revision}/withdraw"
    ]["post"]
    assert withdraw["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DocumentLifecycleOperationRecord"
    }
    typescript = Path("web/src/types.ts").read_text(encoding="utf-8")
    api_client = Path("web/src/api.ts").read_text(encoding="utf-8")
    assert "export type DocumentLifecycleOperationRecord" in typescript
    assert "withdrawKnowledgeDocumentRevision" in api_client
    assert "restoreKnowledgeDocumentRevision" in api_client
