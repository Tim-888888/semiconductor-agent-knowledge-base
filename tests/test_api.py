from __future__ import annotations

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
