"""Capture and compare credential-safe T9-4.7 cold-restore fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bson import BSON

from semikb.config import Settings
from semikb.contracts.models import EvaluationCase
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.rag_retrieval.production_service import (
    ProductionRetrievalService,
    RetrievalOptions,
)
from semikb.storage.clients import StorageClientFactory

MONGO_COLLECTIONS = (
    "document_catalog",
    "chunk_catalog",
    "image_assets",
    "table_assets",
    "source_manifests",
    "ingestion_jobs",
    "ingestion_job_events",
    "document_lifecycle_operations",
    "corpus_standardization_jobs",
    "corpus_publication_batches",
    "evaluation_datasets",
    "evaluation_runs",
    "retrieval_traces",
    "agent_threads",
    "agent_message_requests",
    "checkpoints",
    "checkpoint_writes",
)
MINIO_BUCKETS = ("semikb-raw", "semikb-derived")


def aggregate_digest(values: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--retrieval-smoke", action="store_true")
    return parser.parse_args()


def mongo_fingerprints(database: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in MONGO_COLLECTIONS:
        document_hashes = (
            hashlib.sha256(BSON.encode(document)).digest()
            for document in database[name].find({})
        )
        count = database[name].count_documents({})
        result[name] = {
            "count": count,
            "content_sha256": aggregate_digest(document_hashes),
        }
    return result


def minio_fingerprints(client: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for bucket in MINIO_BUCKETS:
        rows: list[bytes] = []
        samples: list[dict[str, Any]] = []
        total_bytes = 0
        for item in client.list_objects(bucket, recursive=True):
            response = client.get_object(bucket, item.object_name)
            try:
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
            finally:
                response.close()
                response.release_conn()
            sha256 = digest.hexdigest()
            total_bytes += size
            rows.append(f"{item.object_name}\0{size}\0{sha256}".encode())
            if len(samples) < 3:
                samples.append(
                    {"object_key": item.object_name, "bytes": size, "sha256": sha256}
                )
        result[bucket] = {
            "object_count": len(rows),
            "total_bytes": total_bytes,
            "content_manifest_sha256": aggregate_digest(rows),
            "samples": samples,
        }
    return result


def milvus_fingerprint(settings: Settings, factory: StorageClientFactory) -> dict[str, Any]:
    active_collection = collection_name(settings.milvus_index_version)
    with factory.milvus() as client:
        alias = client.describe_alias("semikb_chunks_active")
        count_rows = client.query(
            active_collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        rows = client.query(
            active_collection,
            filter="",
            output_fields=["chunk_id", "document_id", "revision", "index_version"],
            limit=16_384,
            consistency_level="Strong",
        )
    canonical_rows = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        for row in rows
    ]
    return {
        "active_collection": active_collection,
        "alias_collection": alias.get("collection_name"),
        "row_count": int(count_rows[0].get("count(*)", 0)) if count_rows else 0,
        "row_count_consistency": "Strong",
        "metadata_sha256": aggregate_digest(canonical_rows),
    }


def capture(settings: Settings) -> dict[str, Any]:
    factory = StorageClientFactory(settings)
    with factory.mongodb() as client:
        mongo = mongo_fingerprints(client[settings.mongodb_database])
    minio = minio_fingerprints(factory.create_minio())
    with factory.redis() as redis:
        redis = {
            "ping": bool(redis.ping()),
            "celery_queue_depth": int(redis.llen("celery")),
        }
    return {
        "schema": "semikb-t947-restore-fingerprint-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "mongodb": mongo,
        "milvus": milvus_fingerprint(settings, factory),
        "minio": minio,
        "redis": redis,
    }


def invariant_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "mongodb": snapshot.get("mongodb", {}),
        "milvus": snapshot.get("milvus", {}),
        "minio": snapshot.get("minio", {}),
        "redis": snapshot.get("redis", {}),
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    expected = invariant_payload(before)
    actual = invariant_payload(after)
    differences = {
        section: {"expected": expected[section], "actual": actual.get(section)}
        for section in expected
        if expected[section] != actual.get(section)
    }
    return {"matched": not differences, "differences": differences}


def retrieval_smoke(settings: Settings) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "data" / "golden_sets" / "t5_live_v1.json").read_text(encoding="utf-8")
    )
    case = EvaluationCase.model_validate(payload["cases"][0])
    service = ProductionRetrievalService(settings)
    _, trace = service.search(
        case.question,
        case.actor_scope,
        top_k=5,
        options=RetrievalOptions(dense=True, sparse=True, rerank=False, hyde=False),
    )
    expected = set(case.expected_chunk_ids)
    actual = trace.final_evidence_ids
    return {
        "case_id": case.case_id,
        "expected_chunk_ids": sorted(expected),
        "actual_chunk_ids": actual,
        "routes": trace.routes,
        "passed": bool(expected.intersection(actual)),
    }


def main() -> None:
    args = parse_args()
    settings = Settings(demo_mode=False)
    report = capture(settings)
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        report["comparison"] = compare_snapshots(baseline, report)
    if args.retrieval_smoke:
        report["retrieval_smoke"] = retrieval_smoke(settings)

    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe restore fingerprint to {output}")
    else:
        print(serialized, end="")

    if report.get("comparison", {}).get("matched") is False:
        raise SystemExit(1)
    if report.get("retrieval_smoke", {}).get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
