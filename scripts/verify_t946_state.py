"""Capture or compare a credential-safe T9-4.6 datastore state snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semikb.config import Settings
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.storage.clients import StorageClientFactory

MONGO_COLLECTIONS = (
    "document_catalog",
    "chunk_catalog",
    "image_assets",
    "table_assets",
    "ingestion_jobs",
    "ingestion_job_events",
    "evaluation_runs",
    "retrieval_traces",
    "agent_threads",
    "agent_message_requests",
    "checkpoints",
    "checkpoint_writes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def capture(settings: Settings) -> dict[str, Any]:
    factory = StorageClientFactory(settings)
    with factory.mongodb() as client:
        database = client[settings.mongodb_database]
        mongo = {name: database[name].count_documents({}) for name in MONGO_COLLECTIONS}

    active_collection = collection_name(settings.milvus_index_version)
    with factory.milvus() as client:
        alias = client.describe_alias("semikb_chunks_active")
        stats = client.get_collection_stats(active_collection)

    minio = factory.create_minio()
    minio_counts: dict[str, int] = {}
    for bucket in ("semikb-raw", "semikb-derived"):
        minio_counts[bucket] = sum(1 for _ in minio.list_objects(bucket, recursive=True))

    with factory.redis() as redis:
        redis_state = {
            "ping": bool(redis.ping()),
            "celery_queue_depth": int(redis.llen("celery")),
        }
    return {
        "schema": "semikb-t946-state-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "mongodb_counts": mongo,
        "milvus": {
            "active_collection": active_collection,
            "alias_collection": alias.get("collection_name"),
            "row_count": int(stats.get("row_count", 0)),
        },
        "minio_object_counts": minio_counts,
        "redis": redis_state,
    }


def invariant_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "mongodb_counts": snapshot.get("mongodb_counts", {}),
        "milvus": snapshot.get("milvus", {}),
        "minio_object_counts": snapshot.get("minio_object_counts", {}),
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_invariants = invariant_payload(before)
    after_invariants = invariant_payload(after)
    differences: dict[str, dict[str, Any]] = {}
    for section in before_invariants:
        if before_invariants[section] != after_invariants.get(section):
            differences[section] = {
                "before": before_invariants[section],
                "after": after_invariants.get(section),
            }
    return {"matched": not differences, "differences": differences}


def main() -> None:
    args = parse_args()
    settings = Settings(demo_mode=False)
    snapshot = capture(settings)
    report: dict[str, Any] = snapshot
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        comparison = compare_snapshots(before, snapshot)
        report = {**snapshot, "comparison": comparison}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe state snapshot to {output}")
    else:
        print(serialized, end="")
    if report.get("comparison", {}).get("matched") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
