"""Verify the complete T9-4.3.4 business loop against production services."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from semikb.config import Settings
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.storage.clients import StorageClientFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _token(client: httpx.Client, user_id: str) -> str:
    access_key = os.environ.get("DEMO_ACCESS_KEY", "")
    headers = {"X-Demo-Access-Key": access_key} if access_key else {}
    response = client.post(
        "/api/v1/auth/demo-token",
        headers=headers,
        json={
            "user_id": user_id,
            "roles": ["engineer", "knowledge_admin"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _metadata(document_id: str, title: str, document_type: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "revision": "R1",
        "title": title,
        "document_type": document_type,
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic_acceptance",
        "source_uri": f"synthetic://t9-4.3.4/{document_id}",
        "source_license": "CC0-1.0",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "process_layer": "ETCH",
        "tool_id": "ETCH-03",
        "chamber": "B",
        "recipe_id": "ETCH-ALPHA",
        "recipe_version": "V2.3",
    }


def _minimal_pdf(lines: list[str]) -> bytes:
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    text_commands = ["BT", "/F1 11 Tf", "72 740 Td"]
    for index, line in enumerate(escaped):
        if index:
            text_commands.append("0 -18 Td")
        text_commands.append(f"({line}) Tj")
    text_commands.append("ET")
    stream = ("\n".join(text_commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def _upload(
    client: httpx.Client,
    headers: dict[str, str],
    *,
    filename: str,
    content: bytes,
    content_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/ingestion-jobs/upload",
        headers=headers,
        files={"file": (filename, content, content_type)},
        data={"metadata": json.dumps(metadata, ensure_ascii=False)},
    )
    response.raise_for_status()
    return response.json()


def _wait_job(
    client: httpx.Client,
    headers: dict[str, str],
    job_id: str,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/ingestion-jobs/{job_id}", headers=headers)
        response.raise_for_status()
        job = response.json()
        if job.get("status") in {"published", "failed"}:
            break
        time.sleep(2)
    if job.get("status") != "published":
        raise RuntimeError(
            f"ingestion job {job_id} did not publish: "
            f"{job.get('status')} {job.get('error_code')}"
        )
    return job


def _send(
    client: httpx.Client,
    headers: dict[str, str],
    thread_id: str,
    content: str,
) -> dict[str, Any]:
    response = client.post(
        f"/api/v1/threads/{thread_id}/messages",
        headers=headers,
        json={"content": content},
    )
    response.raise_for_status()
    return response.json()


def _wait_evaluation(
    client: httpx.Client,
    headers: dict[str, str],
    run_id: str,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    run: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/evaluation-runs/{run_id}", headers=headers)
        response.raise_for_status()
        run = response.json()
        if run.get("status") in {"completed", "failed"}:
            break
        time.sleep(2)
    if run.get("status") != "completed":
        raise RuntimeError(
            f"evaluation {run_id} did not complete: "
            f"{run.get('status')} {run.get('error_code')}"
        )
    return run


def _latest_assistant(result: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(result.get("thread", {}).get("messages", [])):
        if message.get("role") == "assistant":
            return message
    raise AssertionError("completed response has no assistant message")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    markdown_token = f"T9434MD{suffix}"
    pdf_token = f"T9434PDF{suffix}"
    markdown_id = f"T9434-MD-{suffix}"
    pdf_id = f"T9434-PDF-{suffix}"
    markdown = (
        f"# ETCH-03 Chamber B acceptance note\n\n"
        f"Verification token: {markdown_token}.\n\n"
        "## Controlled action\n\n"
        "When the first wafer after chamber clean shows an edge-ring defect, stop the lot, "
        "review chamber pressure and RF match, and complete a leak check before release.\n"
    ).encode()
    pdf = _minimal_pdf(
        [
            "ETCH-03 Chamber B controlled PDF acceptance note",
            f"Verification token {pdf_token}",
            "Check chamber pressure, RF match, and leak status before lot release.",
        ]
    )
    settings = Settings(demo_mode=False)
    factory = StorageClientFactory(settings)

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        token = _token(client, f"t9434_acceptance_{suffix}")
        headers = {"Authorization": f"Bearer {token}"}

        markdown_job = _upload(
            client,
            headers,
            filename=f"{markdown_id}.md",
            content=markdown,
            content_type="text/markdown",
            metadata=_metadata(markdown_id, "T9-4.3.4 Markdown acceptance", "sop"),
        )
        markdown_job = _wait_job(client, headers, markdown_job["job_id"], args.timeout)
        markdown_duplicate = _upload(
            client,
            headers,
            filename=f"{markdown_id}.md",
            content=markdown,
            content_type="text/markdown",
            metadata=_metadata(markdown_id, "T9-4.3.4 Markdown acceptance", "sop"),
        )
        if markdown_duplicate["job_id"] != markdown_job["job_id"]:
            raise AssertionError("identical Markdown upload created a duplicate job")

        pdf_job = _upload(
            client,
            headers,
            filename=f"{pdf_id}.pdf",
            content=pdf,
            content_type="application/pdf",
            metadata=_metadata(pdf_id, "T9-4.3.4 PDF acceptance", "training_note"),
        )
        pdf_job = _wait_job(client, headers, pdf_job["job_id"], args.timeout)

        search_reports = []
        for token_text, document_id in (
            (markdown_token, markdown_id),
            (pdf_token, pdf_id),
        ):
            response = client.post(
                "/api/v1/retrieval/search",
                headers=headers,
                json={"query": token_text, "top_k": 5},
            )
            response.raise_for_status()
            payload = response.json()
            if document_id not in {item["document_id"] for item in payload["evidence"]}:
                raise AssertionError(f"new document was not retrieved: {document_id}")
            search_reports.append(
                {
                    "document_id": document_id,
                    "trace_id": payload["trace"]["trace_id"],
                    "final_evidence_ids": payload["trace"]["final_evidence_ids"],
                }
            )

        thread_response = client.post(
            "/api/v1/threads",
            headers=headers,
            json={"title": "T9-4.3.4 complete business loop"},
        )
        thread_response.raise_for_status()
        thread_id = thread_response.json()["thread_id"]
        first_question = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"
        first = _send(client, headers, thread_id, first_question)
        history = _send(client, headers, thread_id, "我刚才问了什么？")
        if first_question not in history.get("response", ""):
            raise AssertionError("continuous conversation did not recover the prior question")
        mixed = _send(
            client,
            headers,
            thread_id,
            "查 P-ALPHA ETCH-03 最近24小时 FDC 报警，再对照 SOP 给排查建议",
        )
        if mixed.get("route_decision") != "rag_and_tool":
            raise AssertionError("mixed task did not use rag_and_tool")
        if len(mixed.get("task_results", [])) != len(mixed.get("task_items", [])):
            raise AssertionError("mixed task silently lost a task result")

        image_result = _send(
            client,
            headers,
            thread_id,
            "有没有 ETCH-03 Chamber B 的边缘环状缺陷晶圆图？",
        )
        if image_result.get("route_decision") != "internal_rag":
            raise AssertionError("text-to-image lookup did not use internal_rag")
        image_ids = image_result.get("image_asset_ids", [])
        if not image_ids:
            raise AssertionError("authorized image query returned no final image IDs")
        image_presentation = _latest_assistant(image_result).get("presentation", {})
        if image_presentation.get("image_asset_ids") != image_ids:
            raise AssertionError("message presentation did not preserve final image order")
        access = client.get(
            f"/api/v1/assets/{image_ids[0]}/access",
            headers=headers,
        )
        access.raise_for_status()
        access_payload = access.json()
        if "url" not in access_payload or "expires_at" not in access_payload:
            raise AssertionError("asset access did not return a short-lived access contract")

        no_image = _send(client, headers, thread_id, "你好")
        if no_image.get("image_asset_ids"):
            raise AssertionError("a no-image result retained the previous image list")
        if _latest_assistant(no_image).get("presentation", {}).get("image_asset_ids"):
            raise AssertionError("a no-image message retained stale presentation images")

        trace_id = mixed.get("trace_id") or first.get("trace_id")
        trace_response = client.get(
            f"/api/v1/retrieval-traces/{trace_id}",
            headers=headers,
        )
        trace_response.raise_for_status()
        trace = trace_response.json()
        if trace.get("trace_id") != trace_id or not trace.get("final_evidence_ids"):
            raise AssertionError("retrieval trace cannot reproduce the final evidence")

        evaluation_response = client.post(
            "/api/v1/evaluation-runs",
            headers=headers,
            json={"dataset_version": "demo-v2", "retrieval_profile": "full"},
        )
        if evaluation_response.status_code != 202:
            raise RuntimeError(f"evaluation was not queued: {evaluation_response.text}")
        evaluation = _wait_evaluation(
            client,
            headers,
            evaluation_response.json()["evaluation_run_id"],
            args.timeout,
        )
        first_case = evaluation["case_results"][0]
        case_trace = client.get(
            f"/api/v1/evaluation-runs/{evaluation['evaluation_run_id']}"
            f"/cases/{first_case['case_id']}/trace",
            headers=headers,
        )
        case_trace.raise_for_status()

    cross_store = _cross_store_report(
        factory,
        settings,
        (markdown_job, pdf_job),
        (markdown_id, pdf_id),
    )
    return {
        "verification": "T9-4.3.4-complete-business-loop",
        "documents": [
            _safe_job(markdown_job, duplicate_verified=True),
            _safe_job(pdf_job, duplicate_verified=False),
        ],
        "searches": search_reports,
        "conversation": {
            "thread_id": thread_id,
            "history_recovered": True,
            "mixed_route": mixed.get("route_decision"),
            "mixed_task_statuses": [item["status"] for item in mixed["task_results"]],
            "image_count": len(image_ids),
            "first_image_id": image_ids[0],
            "final_image_order_persisted": True,
            "no_image_clears_projection": True,
        },
        "trace": {
            "trace_id": trace_id,
            "final_evidence_count": len(trace["final_evidence_ids"]),
            "candidate_count": len(trace["candidates"]),
        },
        "evaluation": {
            "evaluation_run_id": evaluation["evaluation_run_id"],
            "status": evaluation["status"],
            "profile": evaluation["retrieval_profile"],
            "case_count": evaluation["case_count"],
            "case_trace_visible": True,
        },
        "cross_store": cross_store,
    }


def _cross_store_report(
    factory: StorageClientFactory,
    settings: Settings,
    jobs: tuple[dict[str, Any], dict[str, Any]],
    document_ids: tuple[str, str],
) -> dict[str, Any]:
    chunk_ids: list[str] = []
    with factory.mongodb() as client:
        database = client[settings.mongodb_database]
        documents = list(
            database.document_catalog.find(
                {"document_id": {"$in": list(document_ids)}, "revision": "R1"},
                {"_id": 0, "document_id": 1, "lifecycle": 1},
            )
        )
        chunks = list(
            database.chunk_catalog.find(
                {"document_id": {"$in": list(document_ids)}, "revision": "R1"},
                {"_id": 0, "chunk_id": 1, "document_id": 1},
            )
        )
        chunk_ids = [str(item["chunk_id"]) for item in chunks]
    if len(documents) != 2 or not chunk_ids:
        raise AssertionError("MongoDB does not contain both published acceptance documents")
    if any(item.get("lifecycle") != "published" for item in documents):
        raise AssertionError("MongoDB contains an unpublished acceptance document")

    minio = factory.create_minio()
    object_count = 0
    for job in jobs:
        for key in ("source_ref", "parsed_ref"):
            object_ref = job.get(key)
            if not object_ref:
                raise AssertionError(f"published job is missing {key}")
            minio.stat_object(object_ref["bucket"], object_ref["object_key"])
            object_count += 1

    active_collection = collection_name(settings.milvus_index_version)
    with factory.milvus() as client:
        rows = client.query(
            active_collection,
            ids=chunk_ids,
            output_fields=["chunk_id", "document_id", "lifecycle"],
            consistency_level="Strong",
        )
        alias = client.describe_alias("semikb_chunks_active")
    if len(rows) != len(chunk_ids):
        raise AssertionError("Milvus does not contain every acceptance chunk")
    if any(item.get("lifecycle") != "published" for item in rows):
        raise AssertionError("Milvus returned an unpublished acceptance chunk")
    if alias.get("collection_name") != active_collection:
        raise AssertionError("Milvus active alias does not target the configured collection")
    return {
        "mongodb_documents": len(documents),
        "mongodb_chunks": len(chunks),
        "minio_objects_verified": object_count,
        "milvus_rows": len(rows),
        "active_collection": active_collection,
        "alias_verified": True,
    }


def _safe_job(job: dict[str, Any], *, duplicate_verified: bool) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "document_id": job["document_id"],
        "file_type": job["file_type"],
        "status": job["status"],
        "attempt": job["attempt"],
        "chunks_count": job["chunks_count"],
        "images_count": job["images_count"],
        "event_stages": [event["stage"] for event in job.get("events", [])],
        "duplicate_verified": duplicate_verified,
    }


def main() -> None:
    args = parse_args()
    report = verify(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
