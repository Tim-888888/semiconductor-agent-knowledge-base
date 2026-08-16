from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from semikb.api.main import _enqueue_ingestion, app
from semikb.bootstrap import get_container
from semikb.config import Settings, get_settings
from semikb.contracts.models import IngestionStatus
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import DeterministicHybridEncoder
from semikb.storage.memory import DemoStore
from semikb_ingest import IngestErrorCode


@pytest.fixture(autouse=True)
def isolate_demo_api_from_local_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    """API unit tests must not inherit production mode from the developer's .env."""

    monkeypatch.setenv("DEMO_MODE", "true")
    get_settings.cache_clear()
    get_container.cache_clear()
    yield
    get_container.cache_clear()
    get_settings.cache_clear()


def test_api_can_create_continuous_thread_and_expose_owned_trace() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "test_engineer",
            "roles": ["engineer"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    assert token_response.status_code == 200
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}

    thread_response = client.post("/api/v1/threads", json={"title": "API test"}, headers=headers)
    assert thread_response.status_code == 201
    thread_id = thread_response.json()["thread_id"]
    message_response = client.post(
        f"/api/v1/threads/{thread_id}/messages",
        json={"content": "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"},
        headers=headers,
    )
    assert message_response.status_code == 200
    trace_id = message_response.json()["trace_id"]

    trace_response = client.get(f"/api/v1/retrieval-traces/{trace_id}", headers=headers)
    assert trace_response.status_code == 200
    assert trace_response.json()["actor_user_id"] == "test_engineer"


def test_liveness_probe_does_not_initialize_application_container() -> None:
    def fail_if_called():
        raise AssertionError("liveness must not initialize external dependencies")

    app.dependency_overrides[get_container] = fail_if_called
    try:
        response = TestClient(app).get("/api/v1/live")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_production_demo_token_requires_configured_access_key() -> None:
    protected = Settings(
        _env_file=None,
        app_env="production",
        demo_access_key="deployment-access-code",
    )
    from semikb.api.main import get_app_settings

    app.dependency_overrides[get_app_settings] = lambda: protected
    client = TestClient(app)
    payload = {"user_id": "test_engineer", "roles": ["engineer"]}
    try:
        assert client.post("/api/v1/auth/demo-token", json=payload).status_code == 401
        assert client.post(
            "/api/v1/auth/demo-token",
            json=payload,
            headers={"X-Demo-Access-Key": "wrong"},
        ).status_code == 401
        accepted = client.post(
            "/api/v1/auth/demo-token",
            json=payload,
            headers={"X-Demo-Access-Key": "deployment-access-code"},
        )
    finally:
        app.dependency_overrides.clear()

    assert accepted.status_code == 200
    assert accepted.json()["token_type"] == "bearer"


def test_production_demo_token_fails_closed_without_access_key_configuration() -> None:
    unconfigured = Settings(_env_file=None, app_env="production", demo_access_key="")
    from semikb.api.main import get_app_settings

    app.dependency_overrides[get_app_settings] = lambda: unconfigured
    try:
        response = TestClient(app).post(
            "/api/v1/auth/demo-token",
            json={"user_id": "test_engineer", "roles": ["engineer"]},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


def test_api_accepts_markdown_upload_as_an_ingestion_job() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "knowledge_admin",
            "roles": ["knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
    metadata = {
        "document_id": "UPLOAD-TEST-01",
        "revision": "R1",
        "title": "Upload test",
        "document_type": "training_note",
        "tool_id": "ETCH-03",
    }
    response = client.post(
        "/api/v1/ingestion-jobs/upload",
        data={"metadata": json.dumps(metadata)},
        files={"file": ("upload.md", b"# Upload\n\nThis is a test note.", "text/markdown")},
        headers=headers,
    )
    assert response.status_code == 201
    assert response.json()["status"] == "staged"


def test_api_publishes_only_reviewed_corpus_outputs_and_exposes_reconciliation() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "knowledge_admin",
            "roles": ["knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
    standardization = client.post(
        "/api/v1/corpus-standardization-jobs/upload",
        data={
            "metadata": json.dumps(
                {
                    "corpus_id": "api-corpus-publication",
                    "snapshot_version": "v1",
                    "display_name": "API corpus publication",
                    "source_kind": "user_upload",
                    "source_uri": "https://example.com/api-corpus",
                    "source_license": "Declared demo terms",
                    "access_scope_key": "demo_engineering",
                    "corpus_kind": "document_collection",
                }
            )
        },
        files=[
            (
                "files",
                (
                    "generic-note.md",
                    b"# Generic note\n\nCheck chamber pressure and RF match before release.",
                    "text/markdown",
                ),
            )
        ],
        headers=headers,
    )
    assert standardization.status_code == 201
    job = standardization.json()
    assert job["status"] == "review_required"
    selected = [
        item["file_id"]
        for item in job["report"]["files"]
        if item["standardized_ref"]
    ]
    publication = client.post(
        "/api/v1/corpus-publication-batches",
        json={
            "request_id": "api-publication-review-1",
            "standardization_job_id": job["job_id"],
            "expected_snapshot_hash": job["snapshot_hash"],
            "selected_file_ids": selected,
            "acknowledged_warning_codes": job["report"]["warning_codes"],
            "source_type": "curated_corpus",
            "content_origin": "real",
            "source_url": "https://example.com/api-corpus",
            "license_name": "Declared demo terms",
            "license_status": "declared",
            "redistribution_policy": "restricted",
            "license_notes": "Interview demonstration only.",
            "access_scope_key": "demo_engineering",
            "review_note": "Reviewed the source, license, warnings, and selected artifacts.",
            "retrieval_policy": "standard",
        },
        headers=headers,
    )
    assert publication.status_code == 202
    batch = publication.json()
    assert batch["status"] == "completed"
    assert batch["items"][0]["reconciliation"]["passed"] is True


def test_upload_api_returns_stable_conflict_for_changed_idempotent_metadata() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "knowledge_admin",
            "roles": ["knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
    metadata = {
        "document_id": "UPLOAD-IDEMPOTENCY-CONFLICT",
        "revision": "R1",
        "title": "Original upload title",
        "document_type": "training_note",
    }
    file_payload = {
        "file": (
            "idempotency.md",
            b"# Idempotency\n\nThe source bytes remain identical.",
            "text/markdown",
        )
    }

    first = client.post(
        "/api/v1/ingestion-jobs/upload",
        data={"metadata": __import__("json").dumps(metadata)},
        files=file_payload,
        headers=headers,
    )
    changed = client.post(
        "/api/v1/ingestion-jobs/upload",
        data={
            "metadata": json.dumps({**metadata, "title": "Changed upload title"})
        },
        files=file_payload,
        headers=headers,
    )

    assert first.status_code == 201
    assert changed.status_code == 409
    assert changed.json()["detail"] == {
        "code": "INGESTION_IDEMPOTENCY_CONFLICT",
        "message": (
            "The same document revision and file were already submitted "
            "with different ingestion metadata."
        ),
    }


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "status_code", "error_code"),
    [
        (
            "unsupported.exe",
            b"MZ synthetic",
            "application/octet-stream",
            415,
            IngestErrorCode.UNSUPPORTED_FORMAT.value,
        ),
        (
            "mismatch.pdf",
            b"%PDF-1.7\nsynthetic",
            "text/plain",
            422,
            IngestErrorCode.FILE_TYPE_MISMATCH.value,
        ),
    ],
)
def test_upload_api_returns_stable_exact_format_errors(
    filename: str,
    content: bytes,
    content_type: str,
    status_code: int,
    error_code: str,
) -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "knowledge_admin",
            "roles": ["knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
    metadata = {
        "document_id": f"UPLOAD-ERROR-{status_code}",
        "revision": "R1",
        "title": "Upload error contract",
        "document_type": "training_note",
    }

    response = client.post(
        "/api/v1/ingestion-jobs/upload",
        data={"metadata": __import__("json").dumps(metadata)},
        files={"file": (filename, content, content_type)},
        headers=headers,
    )

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == error_code
    assert response.json()["detail"]["message"]


def test_queue_submission_failure_marks_job_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    service = IngestionService(
        DemoStore(),
        settings,
        encoder=DeterministicHybridEncoder(8),
    )
    job = service.submit_payload(
        {
            "document_id": "QUEUE-FAIL-01",
            "revision": "R1",
            "title": "Queue failure test",
            "document_type": "training_note",
            "content": "# Queue failure\n\nThis document must remain unpublished.",
        }
    )

    def fail_delay(_: str) -> None:
        raise ConnectionError("broker unavailable")

    from semikb.workers.tasks import process_ingestion_job

    monkeypatch.setattr(process_ingestion_job, "delay", fail_delay)
    with pytest.raises(HTTPException) as exc_info:
        _enqueue_ingestion(SimpleNamespace(ingestion=service), job.job_id)

    failed = service.get_job(job.job_id)
    assert exc_info.value.status_code == 503
    assert failed is not None
    assert failed.status is IngestionStatus.FAILED
    assert failed.error_code == "QUEUE_SUBMISSION_FAILED"


def test_api_serves_authorized_synthetic_wafer_png() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "test_engineer",
            "roles": ["engineer"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}

    response = client.get(
        "/api/v1/assets/IMG-FA-ETCH-03-2026-004/preview",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_api_manages_explicit_user_memory() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    token_response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "memory_api_user",
            "roles": ["engineer"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}

    created = client.post(
        "/api/v1/memories",
        json={"memory_type": "preference", "content": "先列证据，再给建议。"},
        headers=headers,
    )
    assert created.status_code == 201
    memory_id = created.json()["memory_id"]
    listed = client.get("/api/v1/memories", headers=headers)
    assert [item["memory_id"] for item in listed.json()] == [memory_id]
    deleted = client.delete(f"/api/v1/memories/{memory_id}", headers=headers)
    assert deleted.status_code == 204


def test_evaluation_api_requires_admin_and_exposes_reproducible_run() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    engineer_token = client.post(
        "/api/v1/auth/demo-token",
        json={"user_id": "eval_engineer", "roles": ["engineer"]},
    ).json()["access_token"]
    engineer_headers = {"Authorization": f"Bearer {engineer_token}"}

    assert client.get("/api/v1/evaluation-runs", headers=engineer_headers).status_code == 403
    assert (
        client.post(
            "/api/v1/evaluation-runs",
            json={"dataset_version": "demo-v1", "retrieval_profile": "dense"},
            headers=engineer_headers,
        ).status_code
        == 403
    )

    admin_token = client.post(
        "/api/v1/auth/demo-token",
        json={"user_id": "eval_admin", "roles": ["knowledge_admin"]},
    ).json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    created = client.post(
        "/api/v1/evaluation-runs",
        json={"dataset_version": "demo-v1", "retrieval_profile": "reranked"},
        headers=admin_headers,
    )

    assert created.status_code == 202
    assert created.json()["status"] == "completed"
    run_id = created.json()["evaluation_run_id"]
    assert client.get(f"/api/v1/evaluation-runs/{run_id}", headers=admin_headers).status_code == 200
    case_id = created.json()["case_results"][0]["case_id"]
    trace_response = client.get(
        f"/api/v1/evaluation-runs/{run_id}/cases/{case_id}/trace",
        headers=admin_headers,
    )
    assert trace_response.status_code == 200
    assert trace_response.json()["trace_id"] == created.json()["case_results"][0]["trace_id"]
    datasets = client.get("/api/v1/evaluation-datasets", headers=admin_headers)
    assert datasets.status_code == 200
    assert datasets.json()[0]["dataset_version"] == "demo-v1"

    holdout = client.post(
        "/api/v1/evaluation-datasets",
        headers=admin_headers,
        json={
            "dataset_version": "api-sealed-holdout-v1",
            "source_kind": "private-holdout",
            "description": "Sealed API redaction test",
            "purpose": "holdout",
            "source_snapshot_hash": "a" * 64,
            "leakage_status": "cleared",
            "seal": True,
            "cases": [
                {
                    "case_id": "sealed-case-1",
                    "question": "This question must remain sealed.",
                    "expected_chunk_ids": ["SOP-ETCH-03-R2-002"],
                    "actor_scope": {
                        "user_id": "holdout-evaluator",
                        "roles": ["engineer"],
                        "access_scope_keys": ["internal_controlled"],
                    },
                }
            ],
        },
    )
    assert holdout.status_code == 201
    assert holdout.json()["cases_redacted"] is True
    assert "cases" not in holdout.json()
    listed = client.get("/api/v1/evaluation-datasets", headers=admin_headers).json()
    sealed = next(item for item in listed if item["dataset_version"] == "api-sealed-holdout-v1")
    assert sealed["cases_redacted"] is True
    assert "cases" not in sealed


def test_corpus_standardization_api_requires_admin_and_returns_review_inventory() -> None:
    get_container.cache_clear()
    client = TestClient(app)
    engineer_token = client.post(
        "/api/v1/auth/demo-token",
        json={"user_id": "corpus_engineer", "roles": ["engineer"]},
    ).json()["access_token"]
    admin_token = client.post(
        "/api/v1/auth/demo-token",
        json={"user_id": "corpus_admin", "roles": ["knowledge_admin"]},
    ).json()["access_token"]
    metadata = {
        "corpus_id": "api-generic-corpus",
        "snapshot_version": "v1",
        "display_name": "API generic corpus",
        "source_kind": "user_upload",
        "source_license": "unknown",
        "corpus_kind": "auto",
    }
    files = [
        ("files", ("guide.md", b"# Guide\n\nGeneric process note.", "text/markdown")),
        ("files", ("signals.csv", b"signal,value\npressure,12\n", "text/csv")),
    ]
    denied = client.post(
        "/api/v1/corpus-standardization-jobs/upload",
        data={"metadata": json.dumps(metadata)},
        files=files,
        headers={"Authorization": f"Bearer {engineer_token}"},
    )
    assert denied.status_code == 403

    accepted = client.post(
        "/api/v1/corpus-standardization-jobs/upload",
        data={"metadata": json.dumps(metadata)},
        files=files,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert accepted.status_code == 201
    assert accepted.json()["status"] == "review_required"
    assert accepted.json()["files_count"] == 2
    assert {item["role"] for item in accepted.json()["report"]["files"]} == {
        "document",
        "table",
    }
    assert get_container().store.documents.get(("api-generic-corpus", "v1")) is None
