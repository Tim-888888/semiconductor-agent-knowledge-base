"""Verify the T9-4.4.6b lifecycle loop against live project services."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from hashlib import sha256
from typing import Any

from semikb.bootstrap import get_container
from semikb.config import get_settings
from semikb.contracts.models import DocumentRevisionSelector

TERMINAL_STATUSES = {
    "withdrawn",
    "restored",
    "failed",
    "compensation_required",
}


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    token: str = "",
    access_key: str = "",
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if access_key:
        headers["X-Demo-Access-Key"] = access_key
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content = response.read()
            return response.status, json.loads(content) if content else None
    except urllib.error.HTTPError as error:
        content = error.read()
        try:
            detail = json.loads(content) if content else None
        except json.JSONDecodeError:
            detail = content.decode("utf-8", errors="replace")
        return error.code, detail


def _token(
    base_url: str,
    access_key: str,
    *,
    user_id: str = "t9446b_verifier",
    roles: list[str] | None = None,
) -> str:
    status, response = _request(
        base_url,
        "/api/v1/auth/demo-token",
        method="POST",
        access_key=access_key,
        body={
            "user_id": user_id,
            "roles": roles or ["knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    if status != 200:
        raise RuntimeError(f"token request failed with HTTP {status}: {response}")
    return str(response["access_token"])


def _poll_operation(
    base_url: str,
    token: str,
    operation_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status, operation = _request(
            base_url,
            f"/api/v1/knowledge-document-operations/{operation_id}",
            token=token,
        )
        if status != 200:
            raise RuntimeError(f"operation poll failed with HTTP {status}: {operation}")
        if operation["status"] in TERMINAL_STATUSES:
            return operation
        time.sleep(1.0)
    raise TimeoutError(f"operation {operation_id} did not finish in time")


def _search_document(base_url: str, token: str, document_id: str) -> bool:
    status, response = _request(
        base_url,
        "/api/v1/retrieval/search",
        method="POST",
        token=token,
        body={
            "query": "ETCH-03 Chamber B 清腔后首片边缘环状缺陷历史 Case",
            "top_k": 20,
        },
    )
    if status != 200:
        raise RuntimeError(f"retrieval request failed with HTTP {status}: {response}")
    return any(item.get("document_id") == document_id for item in response["evidence"])


def _start_operation(
    base_url: str,
    token: str,
    selector: DocumentRevisionSelector,
    action: str,
    request_id: str,
    reason: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {"request_id": request_id, "reason": reason}
    if action == "restore":
        body["target_index_version"] = get_settings().milvus_index_version
    status, operation = _request(
        base_url,
        (
            f"/api/v1/knowledge-documents/{selector.document_id}/revisions/"
            f"{selector.revision}/{action}"
        ),
        method="POST",
        token=token,
        body=body,
    )
    if status != 202:
        raise RuntimeError(f"{action} request failed with HTTP {status}: {operation}")
    return operation


def verify(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    settings = get_settings()
    container = get_container()
    selector = DocumentRevisionSelector(document_id="CASE-FA-2026-004", revision="R1")
    image_id = "IMG-FA-ETCH-03-2026-004"
    token = _token(base_url, settings.demo_access_key)
    admin_status, _ = _request(
        base_url,
        "/api/v1/knowledge-documents",
        token=token,
    )
    if admin_status != 200:
        raise RuntimeError(
            f"knowledge administrator access returned HTTP {admin_status}; expected 200"
        )
    engineer_token = _token(
        base_url,
        settings.demo_access_key,
        user_id="t9446b_engineer",
        roles=["engineer"],
    )
    engineer_status, _ = _request(
        base_url,
        "/api/v1/knowledge-documents",
        token=engineer_token,
    )
    if engineer_status != 403:
        raise RuntimeError(
            f"engineer document-management access returned HTTP {engineer_status}; expected 403"
        )
    bundle = container.knowledge_documents.repository.get_bundle(selector)
    if bundle is None:
        raise RuntimeError("live verification document is missing")
    if bundle.document.lifecycle.value != "published":
        raise RuntimeError("live verification document must start as published")
    chunk_ids = [chunk.chunk_id for chunk in bundle.chunks]
    source_before = container.knowledge_documents.artifacts.load_object(
        bundle.document.source_ref
    )
    source_hash_before = sha256(source_before).hexdigest()
    run_id = uuid.uuid4().hex
    withdraw_operation: dict[str, Any] | None = None
    restore_operation: dict[str, Any] | None = None
    restored = False
    try:
        if not _search_document(base_url, token, selector.document_id):
            raise RuntimeError("verification document is not retrievable before withdrawal")
        asset_before, _ = _request(
            base_url,
            f"/api/v1/assets/{image_id}/access",
            token=token,
        )
        if asset_before != 200:
            raise RuntimeError("verification image is not accessible before withdrawal")

        withdraw_request_id = f"t9446b-withdraw-{run_id}"
        withdraw_reason = "T9-4.4.6b ECS 跨存储闭环验收下架。"
        requested = _start_operation(
            base_url,
            token,
            selector,
            "withdraw",
            withdraw_request_id,
            withdraw_reason,
        )
        withdraw_operation = _poll_operation(
            base_url,
            token,
            requested["operation_id"],
            timeout_seconds=timeout_seconds,
        )
        if withdraw_operation["status"] != "withdrawn":
            raise RuntimeError(f"withdrawal did not complete: {withdraw_operation}")
        replay = _start_operation(
            base_url,
            token,
            selector,
            "withdraw",
            withdraw_request_id,
            withdraw_reason,
        )
        if replay["operation_id"] != withdraw_operation["operation_id"]:
            raise RuntimeError("idempotent withdrawal replay created a second operation")
        conflict_status, _ = _request(
            base_url,
            (
                f"/api/v1/knowledge-documents/{selector.document_id}/revisions/"
                f"{selector.revision}/withdraw"
            ),
            method="POST",
            token=token,
            body={
                "request_id": withdraw_request_id,
                "reason": "同一请求编号不能改变操作原因。",
            },
        )
        if conflict_status != 409:
            raise RuntimeError(
                f"conflicting idempotency replay returned HTTP {conflict_status}; expected 409"
            )

        withdrawn_bundle = container.knowledge_documents.repository.get_bundle(selector)
        if withdrawn_bundle is None or withdrawn_bundle.document.lifecycle.value != "withdrawn":
            raise RuntimeError("MongoDB did not retain the withdrawn lifecycle gate")
        container.knowledge_documents.vectors.verify_chunks_absent(
            settings.milvus_index_version,
            chunk_ids,
        )
        if _search_document(base_url, token, selector.document_id):
            raise RuntimeError("withdrawn document is still retrievable")
        asset_withdrawn, _ = _request(
            base_url,
            f"/api/v1/assets/{image_id}/access",
            token=token,
        )
        if asset_withdrawn != 403:
            raise RuntimeError(
                f"withdrawn image returned HTTP {asset_withdrawn}; expected 403"
            )
        retained_hash = sha256(
            container.knowledge_documents.artifacts.load_object(
                withdrawn_bundle.document.source_ref
            )
        ).hexdigest()
        if retained_hash != source_hash_before:
            raise RuntimeError("retained MinIO source hash changed during withdrawal")

        requested = _start_operation(
            base_url,
            token,
            selector,
            "restore",
            f"t9446b-restore-{run_id}",
            "T9-4.4.6b ECS 验收完成并校验恢复。",
        )
        restore_operation = _poll_operation(
            base_url,
            token,
            requested["operation_id"],
            timeout_seconds=timeout_seconds,
        )
        if restore_operation["status"] != "restored":
            raise RuntimeError(f"restore did not complete: {restore_operation}")
        restored = True

        restored_bundle = container.knowledge_documents.repository.get_bundle(selector)
        if restored_bundle is None or restored_bundle.document.lifecycle.value != "published":
            raise RuntimeError("MongoDB did not republish the restored revision")
        if not _search_document(base_url, token, selector.document_id):
            raise RuntimeError("restored document is not retrievable")
        asset_restored, _ = _request(
            base_url,
            f"/api/v1/assets/{image_id}/access",
            token=token,
        )
        if asset_restored != 200:
            raise RuntimeError("restored image is not accessible")
        source_after = container.knowledge_documents.artifacts.load_object(
            restored_bundle.document.source_ref
        )
        if sha256(source_after).hexdigest() != source_hash_before:
            raise RuntimeError("MinIO source hash changed after restoration")
        return {
            "status": "passed",
            "selector": selector.model_dump(mode="json"),
            "chunk_count": len(chunk_ids),
            "image_count": len(bundle.images),
            "table_count": len(bundle.tables),
            "source_hash_preserved": True,
            "rbac": {"knowledge_admin": admin_status, "engineer": engineer_status},
            "idempotency": {"same_request": "reused", "changed_reason": 409},
            "withdraw_operation_id": withdraw_operation["operation_id"],
            "withdraw_status": withdraw_operation["status"],
            "restore_operation_id": restore_operation["operation_id"],
            "restore_status": restore_operation["status"],
            "final_lifecycle": restored_bundle.document.lifecycle.value,
            "asset_http": {"before": 200, "withdrawn": 403, "restored": 200},
            "retrieval": {"before": True, "withdrawn": False, "restored": True},
        }
    finally:
        if not restored:
            current = container.knowledge_documents.repository.get_bundle(selector)
            if current is not None and current.document.lifecycle.value == "withdrawn":
                try:
                    requested = _start_operation(
                        base_url,
                        token,
                        selector,
                        "restore",
                        f"t9446b-safety-restore-{run_id}",
                        "T9-4.4.6b 验收异常后的安全恢复。",
                    )
                    _poll_operation(
                        base_url,
                        token,
                        requested["operation_id"],
                        timeout_seconds=timeout_seconds,
                    )
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.base_url, args.timeout_seconds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
