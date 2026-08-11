from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from semikb.api.main import _enqueue_ingestion, app
from semikb.bootstrap import get_container
from semikb.config import Settings
from semikb.contracts.models import IngestionStatus
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import DeterministicHybridEncoder
from semikb.storage.memory import DemoStore


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
        data={"metadata": __import__("json").dumps(metadata)},
        files={"file": ("upload.md", b"# Upload\n\nThis is a test note.", "text/markdown")},
        headers=headers,
    )
    assert response.status_code == 201
    assert response.json()["status"] == "published"


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
