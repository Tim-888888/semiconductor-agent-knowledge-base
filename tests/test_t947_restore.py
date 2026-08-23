from __future__ import annotations

from scripts.verify_t947_restore import aggregate_digest, compare_snapshots


def _snapshot(*, mongo_hash: str = "mongo", queue_depth: int = 0) -> dict[str, object]:
    return {
        "mongodb": {"document_catalog": {"count": 1, "content_sha256": mongo_hash}},
        "milvus": {
            "active_collection": "semikb_chunks_v4",
            "alias_collection": "semikb_chunks_v4",
            "row_count": 1,
            "row_count_consistency": "Strong",
            "metadata_sha256": "milvus",
        },
        "minio": {
            "semikb-raw": {
                "object_count": 1,
                "total_bytes": 3,
                "content_manifest_sha256": "raw",
                "samples": [],
            }
        },
        "redis": {"ping": True, "celery_queue_depth": queue_depth},
    }


def test_aggregate_digest_is_order_independent_and_length_delimited() -> None:
    assert aggregate_digest([b"a", b"bc"]) == aggregate_digest([b"bc", b"a"])
    assert aggregate_digest([b"a", b"bc"]) != aggregate_digest([b"ab", b"c"])


def test_restore_snapshot_comparison_accepts_exact_business_invariants() -> None:
    comparison = compare_snapshots(_snapshot(), _snapshot())

    assert comparison == {"matched": True, "differences": {}}


def test_restore_snapshot_comparison_reports_changed_collection_and_queue() -> None:
    comparison = compare_snapshots(
        _snapshot(),
        _snapshot(mongo_hash="changed", queue_depth=2),
    )

    assert comparison["matched"] is False
    assert set(comparison["differences"]) == {"mongodb", "redis"}
