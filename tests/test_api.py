from __future__ import annotations

from fastapi.testclient import TestClient

from semikb.api.main import app
from semikb.bootstrap import get_container


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
