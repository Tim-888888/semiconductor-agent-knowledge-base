"""Verify fresh Worker ingestion into Qwen native Dense+Sparse v4, then clean up."""

from __future__ import annotations

import json
import time
from uuid import uuid4

from semikb.config import Settings
from semikb.contracts.models import IngestionStatus, ObjectRef
from semikb.rag_ingestion.service import IngestionService
from semikb.storage.clients import StorageClientFactory
from semikb.storage.production_ingestion import ProductionIngestionStore
from semikb.workers.tasks import process_ingestion_job


def _payload(document_id: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "revision": "R1",
        "title": "Qwen native sparse ingestion acceptance SOP",
        "document_type": "sop",
        "content": (
            "# Chamber pressure recovery\n\n"
            "For ETCH-03 Chamber B, stop automatic recovery after a pressure alarm. "
            "Verify leak-check status, RF match stability, and two monitor wafers "
            "before restoring recipe ETCH-ALPHA V2.3."
        ),
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic",
        "source_uri": "synthetic://acceptance/qwen-native-sparse",
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


def _sparse_size(value: object) -> int:
    if isinstance(value, dict):
        return len(value)
    indices = getattr(value, "indices", None)
    if indices is not None:
        return len(indices)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def main() -> None:
    settings = Settings(demo_mode=False)
    if settings.milvus_index_version != "v4":
        raise RuntimeError("This acceptance requires MILVUS_INDEX_VERSION=v4.")
    if settings.embedding_output_type != "dense&sparse":
        raise RuntimeError("This acceptance requires EMBEDDING_OUTPUT_TYPE=dense&sparse.")

    factory = StorageClientFactory(settings)
    service = IngestionService(
        ProductionIngestionStore(settings, factory),
        settings,
    )
    document_id = f"V4-SPARSE-ACCEPT-{uuid4().hex[:10].upper()}"
    job_id: str | None = None
    chunk_ids: list[str] = []
    object_refs: list[ObjectRef] = []
    release_before: dict[str, object] | None = None
    cleanup = {"mongodb": 0, "milvus": 0, "minio": 0}

    with factory.mongodb() as client:
        release_before = client[settings.mongodb_database].index_releases.find_one(
            {"index_version": settings.milvus_index_version}
        )

    try:
        queued = service.submit_payload(
            _payload(document_id),
            created_by="qwen_native_sparse_acceptance",
        )
        job_id = queued.job_id
        process_ingestion_job.delay(job_id)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = service.get_job(job_id)
            if job and job.status in {IngestionStatus.PUBLISHED, IngestionStatus.FAILED}:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Ingestion job {job_id} did not finish in time.")
        if job is None or job.status is not IngestionStatus.PUBLISHED:
            summary = job.safe_error_summary if job else "job missing"
            raise RuntimeError(f"Fresh v4 ingestion failed: {summary}")
        embedding_event = next(
            (
                event
                for event in job.events
                if event.stage is IngestionStatus.EMBEDDING
            ),
            None,
        )
        expected_event_message = (
            "Generating Dense and Sparse representations with "
            f"{settings.embedding_model} / {settings.sparse_encoder_version}."
        )
        if embedding_event is None or embedding_event.message != expected_event_message:
            raise RuntimeError("Embedding task event does not describe the active encoder.")
        object_refs = [ref for ref in (job.source_ref, job.parsed_ref) if ref is not None]

        with factory.mongodb() as client:
            database = client[settings.mongodb_database]
            document = database.document_catalog.find_one(
                {"document_id": document_id, "revision": "R1"},
                {"_id": 0},
            )
            chunks = list(
                database.chunk_catalog.find(
                    {"document_id": document_id, "revision": "R1"},
                    {"_id": 0, "chunk_id": 1, "embedding_version": 1, "index_version": 1},
                )
            )
            release = database.index_releases.find_one(
                {"index_version": "v4"},
                {
                    "_id": 0,
                    "embedding_output_type": 1,
                    "sparse_encoder_version": 1,
                    "normalization": 1,
                },
            )
        if document is None or document.get("index_version") != "v4":
            raise RuntimeError("Fresh document did not publish into index v4.")
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if not chunk_ids or any(
            chunk.get("index_version") != "v4"
            or chunk.get("embedding_version") != settings.embedding_version
            for chunk in chunks
        ):
            raise RuntimeError("Fresh Chunk catalog has incorrect vector versions.")
        expected_release = {
            "embedding_output_type": "dense&sparse",
            "sparse_encoder_version": settings.sparse_encoder_version,
            "normalization": "dense_l2_sparse_provider_raw",
        }
        if release != expected_release:
            raise RuntimeError("v4 release metadata does not describe native Sparse output.")

        with factory.milvus() as client:
            rows = client.query(
                "semikb_chunks_v4",
                ids=chunk_ids,
                output_fields=["chunk_id", "dense_vector", "sparse_vector", "lifecycle"],
                consistency_level="Strong",
            )
        sparse_sizes = [_sparse_size(row.get("sparse_vector")) for row in rows]
        if len(rows) != len(chunk_ids) or any(size <= 0 for size in sparse_sizes):
            raise RuntimeError("Fresh v4 Milvus rows are missing native Sparse values.")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "job_id": job_id,
                    "document_id": document_id,
                    "chunk_ids": chunk_ids,
                    "sparse_nonzero_counts": sparse_sizes,
                    "embedding_version": settings.embedding_version,
                    "embedding_event": embedding_event.message,
                    "release_metadata": release,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if job_id:
            latest = service.get_job(job_id)
            for ref in (latest.source_ref, latest.parsed_ref) if latest else ():
                if ref is not None and ref not in object_refs:
                    object_refs.append(ref)
        minio = factory.create_minio()
        for ref in object_refs:
            minio.remove_object(ref.bucket, ref.object_key, version_id=ref.version_id)
            cleanup["minio"] += 1
        if chunk_ids:
            with factory.milvus() as client:
                result = client.delete("semikb_chunks_v4", ids=chunk_ids)
                cleanup["milvus"] = int(result.get("delete_count", len(chunk_ids)))
        with factory.mongodb() as client:
            database = client[settings.mongodb_database]
            selector = {"document_id": document_id, "revision": "R1"}
            for name in ("document_catalog", "chunk_catalog", "image_assets"):
                cleanup["mongodb"] += database[name].delete_many(selector).deleted_count
            if job_id:
                cleanup["mongodb"] += database.ingestion_job_events.delete_many(
                    {"job_id": job_id}
                ).deleted_count
                cleanup["mongodb"] += database.ingestion_jobs.delete_many(
                    {"job_id": job_id}
                ).deleted_count
                cleanup["mongodb"] += database.audit_events.delete_many(
                    {"job_id": job_id}
                ).deleted_count
            if release_before is not None:
                database.index_releases.replace_one(
                    {"index_version": settings.milvus_index_version},
                    release_before,
                    upsert=True,
                )
        print(json.dumps({"cleanup": cleanup}, ensure_ascii=False))


if __name__ == "__main__":
    main()
