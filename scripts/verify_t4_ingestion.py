"""Repeatable live T4 acceptance against project-owned external storage."""

from __future__ import annotations

import json
from collections.abc import Sequence

from semikb.config import get_settings
from semikb.contracts.models import IngestionStatus, ObjectRef
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import BgeM3Encoder, HybridEmbedding
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.storage.clients import StorageClientFactory
from semikb.storage.production_ingestion import ProductionIngestionStore


class InjectedFailureEncoder:
    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        raise RuntimeError("T4 acceptance injected an embedding failure")


def _main_payload() -> dict[str, object]:
    return {
        "document_id": "T4-CASE-ETCH-03",
        "revision": "R1",
        "title": "ETCH-03 Chamber B edge-ring controlled case",
        "document_type": "failure_analysis_case",
        "content": (
            "# 现象\n\nETCH-03 Chamber B 清腔后首片出现边缘环状缺陷。\n\n"
            "## 排查\n\n确认 chamber pressure 短时波动，并执行 leak check 与 RF match 校验。\n\n"
            "## 验证\n\n两片 monitor wafer 均未复现 edge-ring pattern 后才允许恢复生产。"
        ),
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic",
        "source_uri": "synthetic://t4/etch03-edge-ring-case",
        "source_license": "CC0-1.0",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "process_layer": "ETCH",
        "tool_id": "ETCH-03",
        "chamber": "B",
        "recipe_id": "ETCH-ALPHA",
        "recipe_version": "V2.3",
        "images": [
            {
                "image_id": "T4-IMG-ETCH-03-EDGE-RING",
                "image_type": "wafer_map",
                "caption": "ETCH-03 Chamber B edge-ring defect wafer map",
                "caption_source": "human",
                "caption_confidence": 1.0,
                "detection_summary": "Continuous edge ring pattern detected near wafer perimeter.",
                "source_page": "T4 synthetic case attachment 1",
                "source_path": "data/assets/wafer_maps/etch03_chamber_b_edge_ring.png",
            }
        ],
    }


def _retry_payload() -> dict[str, object]:
    return {
        "document_id": "T4-RETRY-SOP-ETCH-03",
        "revision": "R1",
        "title": "ETCH-03 pressure alarm retry SOP",
        "document_type": "sop",
        "content": (
            "# Pressure alarm\n\nStop automatic recovery, verify chamber pressure and leak status, "
            "then obtain approval before restoring the recipe."
        ),
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic",
        "source_uri": "synthetic://t4/retry-sop",
        "source_license": "CC0-1.0",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "tool_id": "ETCH-03",
        "chamber": "B",
    }


def main() -> None:
    settings = get_settings().model_copy(update={"demo_mode": False, "bge_use_fp16": False})
    factory = StorageClientFactory(settings)
    store = ProductionIngestionStore(settings, factory)
    encoder = BgeM3Encoder(settings)
    service = IngestionService(store, settings, encoder=encoder)

    main_job = service.ingest_payload(_main_payload(), created_by="t4_acceptance")
    if main_job.status is IngestionStatus.FAILED:
        main_job = service.retry(main_job.job_id)
    assert main_job.status is IngestionStatus.PUBLISHED

    repeated = service.ingest_payload(_main_payload(), created_by="t4_acceptance")
    assert repeated.job_id == main_job.job_id

    retry_submission = service.submit_payload(_retry_payload(), created_by="t4_acceptance")
    if retry_submission.status is IngestionStatus.QUEUED:
        failed = IngestionService(
            store,
            settings,
            encoder=InjectedFailureEncoder(),
        ).process(retry_submission.job_id)
        assert failed.status is IngestionStatus.FAILED
        retry_submission = service.retry(failed.job_id)
    assert retry_submission.status is IngestionStatus.PUBLISHED
    assert retry_submission.attempt >= 2

    with factory.mongodb() as client:
        database = client[settings.mongodb_database]
        document = database.document_catalog.find_one(
            {"document_id": "T4-CASE-ETCH-03", "revision": "R1"}
        )
        chunks = list(
            database.chunk_catalog.find(
                {"document_id": "T4-CASE-ETCH-03", "revision": "R1"},
                {"_id": 0, "chunk_id": 1, "lifecycle": 1},
            )
        )
        images = list(
            database.image_assets.find(
                {"document_id": "T4-CASE-ETCH-03", "revision": "R1"},
                {"_id": 0, "image_id": 1, "object_ref": 1},
            )
        )
        event_count = database.ingestion_job_events.count_documents(
            {"job_id": main_job.job_id}
        )
        retry_event_count = database.ingestion_job_events.count_documents(
            {"job_id": retry_submission.job_id}
        )
        duplicate_jobs = database.ingestion_jobs.count_documents(
            {"idempotency_key": main_job.idempotency_key}
        )
    assert document is not None and document["lifecycle"] == "published"
    assert len(chunks) == main_job.chunks_count
    assert len(images) == 1
    assert event_count == len(main_job.events)
    assert retry_event_count == len(retry_submission.events)
    assert any(event.stage is IngestionStatus.FAILED for event in retry_submission.events)
    assert {event.attempt for event in retry_submission.events} == {1, 2}
    assert duplicate_jobs == 1

    for object_ref in (main_job.source_ref, main_job.parsed_ref):
        assert object_ref is not None
        factory.create_minio().stat_object(object_ref.bucket, object_ref.object_key)
    for image in images:
        object_ref = ObjectRef.model_validate(image["object_ref"])
        factory.create_minio().stat_object(object_ref.bucket, object_ref.object_key)

    chunk_ids = [item["chunk_id"] for item in chunks]
    retry_chunk_ids: list[str]
    with factory.mongodb() as client:
        retry_chunk_ids = [
            item["chunk_id"]
            for item in client[settings.mongodb_database].chunk_catalog.find(
                {"document_id": "T4-RETRY-SOP-ETCH-03", "revision": "R1"},
                {"chunk_id": 1},
            )
        ]
    all_chunk_ids = chunk_ids + retry_chunk_ids
    active_collection = collection_name(settings.milvus_index_version)
    with factory.milvus() as client:
        vector_rows = client.query(
            active_collection,
            ids=all_chunk_ids,
            output_fields=["chunk_id", "document_id", "lifecycle"],
            consistency_level="Strong",
        )
        legacy_rows = client.query(
            "semikb_chunks_v1",
            ids=all_chunk_ids,
            output_fields=["chunk_id"],
            consistency_level="Strong",
        )
        alias = client.describe_alias("semikb_chunks_active")
    assert len(vector_rows) == len(all_chunk_ids)
    assert all(item["lifecycle"] == "published" for item in vector_rows)
    assert legacy_rows == []
    assert alias.get("collection_name") == active_collection

    print(
        json.dumps(
            {
                "status": "passed",
                "main_job_id": main_job.job_id,
                "retry_job_id": retry_submission.job_id,
                "retry_attempts": retry_submission.attempt,
                "chunks": len(chunks),
                "images": len(images),
                "events": event_count,
                "milvus_rows": len(vector_rows),
                "active_alias": alias.get("alias"),
                "active_collection": alias.get("collection_name"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
