"""Verify T9-4.4.5 multi-format ingestion against production services."""

from __future__ import annotations

import argparse
import hashlib
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

GOLDEN_ROOT = Path("data/t9445_golden")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://api:8000")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _manifest() -> dict[str, Any]:
    manifest = json.loads((GOLDEN_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        content = (GOLDEN_ROOT / entry["filename"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise AssertionError(f"golden file hash changed: {entry['filename']}")
    return manifest


def _token(
    client: httpx.Client,
    *,
    user_id: str,
    scopes: list[str],
) -> str:
    access_key = os.environ.get("DEMO_ACCESS_KEY", "")
    headers = {"X-Demo-Access-Key": access_key} if access_key else {}
    response = client.post(
        "/api/v1/auth/demo-token",
        headers=headers,
        json={
            "user_id": user_id,
            "roles": ["engineer", "knowledge_admin"],
            "access_scope_keys": scopes,
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


def _metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": entry["document_id"],
        "revision": entry["revision"],
        "title": entry["title"],
        "document_type": entry["document_type"],
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": entry["source_kind"],
        "source_uri": entry["source_uri"],
        "source_license": entry["source_license"],
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "process_layer": "ETCH",
        "tool_id": "ETCH-03",
        "chamber": "B",
        "recipe_id": "ETCH-ALPHA",
        "recipe_version": "V2.3",
    }


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
            f"{job.get('status')} {job.get('error_code')} {job.get('safe_error_summary')}"
        )
    return job


def _search(
    client: httpx.Client,
    headers: dict[str, str],
    query: str,
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/retrieval/search",
        headers=headers,
        json={"query": query, "top_k": 8},
    )
    response.raise_for_status()
    return response.json()


def _verify_jobs(
    client: httpx.Client,
    headers: dict[str, str],
    manifest: dict[str, Any],
    timeout: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        content = (GOLDEN_ROOT / entry["filename"]).read_bytes()
        metadata = _metadata(entry)
        submitted = _upload(
            client,
            headers,
            filename=entry["filename"],
            content=content,
            content_type=entry["content_type"],
            metadata=metadata,
        )
        job = _wait_job(client, headers, submitted["job_id"], timeout)
        if job["parser_name"] != entry["parser_name"]:
            raise AssertionError(f"unexpected parser for {entry['category']}: {job['parser_name']}")
        if job["chunks_count"] < entry["min_chunks"]:
            raise AssertionError(f"too few chunks for {entry['category']}")
        if job["images_count"] < entry["min_images"]:
            raise AssertionError(f"too few images for {entry['category']}")
        if job["tables_count"] < entry["min_tables"]:
            raise AssertionError(f"too few tables for {entry['category']}")
        stages = [event["stage"] for event in job.get("events", [])]
        for expected_stage in (
            "validating",
            "parsing",
            "quality_check",
            "embedding",
            "staged",
            "published",
        ):
            if expected_stage not in stages:
                raise AssertionError(f"{entry['category']} missed stage {expected_stage}")
        duplicate = _upload(
            client,
            headers,
            filename=entry["filename"],
            content=content,
            content_type=entry["content_type"],
            metadata=metadata,
        )
        if duplicate["job_id"] != job["job_id"]:
            raise AssertionError(f"duplicate {entry['category']} upload created a new job")
        result = _search(client, headers, entry["retrieval_query"])
        evidence = result["evidence"]
        if entry["document_id"] not in {item["document_id"] for item in evidence}:
            raise AssertionError(f"retrieval missed {entry['category']} golden document")
        jobs.append(job)
        searches.append(
            {
                "category": entry["category"],
                "document_id": entry["document_id"],
                "trace_id": result["trace"]["trace_id"],
                "routes": result["trace"]["routes"],
                "final_evidence_ids": result["trace"]["final_evidence_ids"],
                "image_asset_ids": result["trace"]["image_asset_ids"],
            }
        )
    return jobs, searches


def _verify_acl(
    client: httpx.Client,
    unauthorized_headers: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    protected_ids = {entry["document_id"] for entry in manifest["entries"]}
    trace_ids: list[str] = []
    for entry in manifest["entries"]:
        result = _search(client, unauthorized_headers, entry["retrieval_query"])
        leaked = protected_ids.intersection(item["document_id"] for item in result["evidence"])
        if leaked:
            raise AssertionError(f"unauthorized retrieval leaked documents: {sorted(leaked)}")
        trace_ids.append(result["trace"]["trace_id"])
    return {"queries": len(trace_ids), "leaked_documents": 0, "trace_ids": trace_ids}


def _verify_version_filter(
    client: httpx.Client,
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    document_id = "T9445-VERSION-PROBE"
    common = {
        "document_id": document_id,
        "title": "T9-4.4.5 Version Filter Probe",
        "document_type": "sop",
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic_acceptance",
        "source_license": "CC0-1.0",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "process_layer": "ETCH",
        "tool_id": "ETCH-03",
        "chamber": "B",
    }
    first_content = b"# Obsolete gate\n\nT9445-VERSION-OLD-11 must not remain active.\n"
    first = _upload(
        client,
        headers,
        filename="version-r1.md",
        content=first_content,
        content_type="text/markdown",
        metadata={
            **common,
            "revision": "R1",
            "source_uri": "synthetic://t9-4.4.5/version-r1",
        },
    )
    _wait_job(client, headers, first["job_id"], timeout)
    second_content = b"# Current gate\n\nT9445-VERSION-CURRENT-22 is the approved release rule.\n"
    second = _upload(
        client,
        headers,
        filename="version-r2.md",
        content=second_content,
        content_type="text/markdown",
        metadata={
            **common,
            "revision": "R2",
            "supersedes_revision": "R1",
            "source_uri": "synthetic://t9-4.4.5/version-r2",
        },
    )
    second_job = _wait_job(client, headers, second["job_id"], timeout)
    old_result = _search(client, headers, "T9445-VERSION-OLD-11 obsolete gate")
    if any(
        item["document_id"] == document_id and item["revision"] == "R1"
        for item in old_result["evidence"]
    ):
        raise AssertionError("superseded R1 was returned by live retrieval")
    current_result = _search(client, headers, "T9445-VERSION-CURRENT-22 approved release rule")
    if not any(
        item["document_id"] == document_id and item["revision"] == "R2"
        for item in current_result["evidence"]
    ):
        raise AssertionError("published R2 was not returned by live retrieval")
    return {
        "document_id": document_id,
        "published_job_id": second_job["job_id"],
        "superseded_revision_hidden": True,
        "current_revision_retrieved": True,
        "trace_ids": [old_result["trace"]["trace_id"], current_result["trace"]["trace_id"]],
    }


def _verify_invalid_format(client: httpx.Client, headers: dict[str, str]) -> dict[str, Any]:
    metadata = {
        "document_id": "T9445-INVALID-FORMAT",
        "revision": "R1",
        "title": "Invalid format probe",
        "document_type": "test_probe",
        "source_kind": "synthetic_acceptance",
        "source_license": "CC0-1.0",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
    }
    response = client.post(
        "/api/v1/ingestion-jobs/upload",
        headers=headers,
        files={"file": ("mismatch.pdf", b"not a PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    if response.status_code != 422:
        raise AssertionError(f"format mismatch returned {response.status_code}: {response.text}")
    detail = response.json().get("detail", {})
    if detail.get("code") != "INGEST_FILE_TYPE_MISMATCH":
        raise AssertionError(f"unexpected mismatch error: {detail}")
    return {"status_code": response.status_code, "error_code": detail["code"]}


def _cross_store_report(
    settings: Settings,
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    factory = StorageClientFactory(settings)
    document_ids = [entry["document_id"] for entry in manifest["entries"]]
    entries_by_id = {entry["document_id"]: entry for entry in manifest["entries"]}
    with factory.mongodb() as client:
        database = client[settings.mongodb_database]
        documents = list(
            database.document_catalog.find(
                {"document_id": {"$in": document_ids}, "revision": "R1"},
                {"_id": 0},
            )
        )
        chunks = list(
            database.chunk_catalog.find(
                {"document_id": {"$in": document_ids}, "revision": "R1"},
                {"_id": 0},
            )
        )
        images = list(
            database.image_assets.find(
                {"document_id": {"$in": document_ids}, "revision": "R1"},
                {"_id": 0},
            )
        )
        tables = list(
            database.table_assets.find(
                {"document_id": {"$in": document_ids}, "revision": "R1"},
                {"_id": 0},
            )
        )
    if len(documents) != len(document_ids):
        raise AssertionError("MongoDB does not contain every golden document")
    if any(document.get("lifecycle") != "published" for document in documents):
        raise AssertionError("MongoDB contains an unpublished golden document")
    if any(document.get("parser_name") != entries_by_id[document["document_id"]]["parser_name"] for document in documents):
        raise AssertionError("MongoDB parser audit does not match the golden manifest")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        document_id: {"chunks": [], "images": [], "tables": []}
        for document_id in document_ids
    }
    for name, records in (("chunks", chunks), ("images", images), ("tables", tables)):
        for record in records:
            grouped[record["document_id"]][name].append(record)
    for document_id, records in grouped.items():
        entry = entries_by_id[document_id]
        if len(records["chunks"]) < entry["min_chunks"]:
            raise AssertionError(f"MongoDB chunk count is too low for {document_id}")
        if len(records["images"]) < entry["min_images"]:
            raise AssertionError(f"MongoDB image count is too low for {document_id}")
        if len(records["tables"]) < entry["min_tables"]:
            raise AssertionError(f"MongoDB table count is too low for {document_id}")
        corpus_parts: list[str] = []
        for chunk in records["chunks"]:
            corpus_parts.extend(
                (
                    chunk.get("chunk_text", ""),
                    " ".join(chunk.get("title_path", [])),
                    chunk.get("page_or_section", ""),
                )
            )
        for table in records["tables"]:
            corpus_parts.extend((table.get("title", ""), table.get("markdown", "")))
        for image in records["images"]:
            corpus_parts.extend(
                (
                    image.get("caption", ""),
                    image.get("ocr_text", ""),
                    image.get("detection_summary", ""),
                )
            )
        corpus = "\n".join(corpus_parts).lower()
        if not all(term.lower() in corpus for term in entry["required_terms"]):
            raise AssertionError(f"governed records lost required evidence for {document_id}")
        if any(term.lower() in corpus for term in entry.get("forbidden_terms", [])):
            raise AssertionError(f"governed records retained forbidden noise for {document_id}")

    minio = factory.create_minio()
    object_refs: list[dict[str, Any]] = []
    for job in jobs:
        object_refs.extend(ref for ref in (job.get("source_ref"), job.get("parsed_ref")) if ref)
    object_refs.extend(record["object_ref"] for record in images)
    object_refs.extend(record["object_ref"] for record in tables)
    for object_ref in object_refs:
        stat = minio.stat_object(object_ref["bucket"], object_ref["object_key"])
        if stat.size <= 0:
            raise AssertionError(f"empty MinIO object: {object_ref['object_key']}")

    chunk_ids = [record["chunk_id"] for record in chunks]
    active_collection = collection_name(settings.milvus_index_version)
    with factory.milvus() as client:
        rows = client.query(
            active_collection,
            ids=chunk_ids,
            output_fields=["chunk_id", "document_id", "revision", "lifecycle"],
            consistency_level="Strong",
        )
        alias = client.describe_alias("semikb_chunks_active")
    if len(rows) != len(chunk_ids):
        raise AssertionError("Milvus does not contain every golden chunk")
    if any(row.get("lifecycle") != "published" for row in rows):
        raise AssertionError("Milvus contains an unpublished golden chunk")
    if alias.get("collection_name") != active_collection:
        raise AssertionError("Milvus active alias does not target the configured collection")
    return {
        "mongodb": {
            "documents": len(documents),
            "chunks": len(chunks),
            "images": len(images),
            "tables": len(tables),
            "parser_audit_verified": True,
        },
        "minio": {"objects_verified": len(object_refs)},
        "milvus": {
            "rows": len(rows),
            "active_collection": active_collection,
            "alias_verified": True,
        },
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest()
    settings = Settings(demo_mode=False)
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    with httpx.Client(base_url=args.base_url, timeout=240) as client:
        authorized = {
            "Authorization": f"Bearer {_token(client, user_id=f't9445_admin_{suffix}', scopes=['demo_engineering'])}"
        }
        unauthorized = {
            "Authorization": f"Bearer {_token(client, user_id=f't9445_other_{suffix}', scopes=['restricted_other'])}"
        }
        jobs, searches = _verify_jobs(client, authorized, manifest, args.timeout)
        acl = _verify_acl(client, unauthorized, manifest)
        version_filter = _verify_version_filter(client, authorized, args.timeout)
        invalid_format = _verify_invalid_format(client, authorized)
    cross_store = _cross_store_report(settings, manifest, jobs)
    return {
        "verification": "T9-4.4.5-multi-format-live-loop",
        "golden_version": manifest["golden_version"],
        "executed_at": datetime.now(UTC).isoformat(),
        "formats": [
            {
                "category": entry["category"],
                "document_id": entry["document_id"],
                "filename": entry["filename"],
                "sha256": entry["sha256"],
                "job_id": job["job_id"],
                "parser_name": job["parser_name"],
                "provider_name": job.get("provider_name"),
                "provider_version": job.get("provider_version"),
                "chunks_count": job["chunks_count"],
                "images_count": job["images_count"],
                "tables_count": job["tables_count"],
                "duplicate_job_verified": True,
            }
            for entry, job in zip(manifest["entries"], jobs, strict=True)
        ],
        "searches": searches,
        "acl": acl,
        "version_filter": version_filter,
        "invalid_format": invalid_format,
        "cross_store": cross_store,
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
