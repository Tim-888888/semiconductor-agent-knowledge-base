"""Milvus schema contract kept independent from the pymilvus runtime adapter."""

from __future__ import annotations

from datetime import datetime

from semikb.contracts.models import Chunk

MILVUS_STRING_FIELDS = (
    "document_id",
    "revision",
    "chunk_type",
    "approval_status",
    "fab",
    "product",
    "process_layer",
    "tool_id",
    "chamber",
    "recipe_id",
    "recipe_version",
    "access_scope_key",
    "index_version",
)


def collection_name(index_version: str) -> str:
    return f"semikb_chunks_{index_version.replace('-', '_')}"


def schema_contract(embedding_dim: int, index_version: str) -> dict[str, object]:
    """Return the logical Milvus schema used by bootstrap and documentation tooling."""

    return {
        "collection": collection_name(index_version),
        "alias": "semikb_chunks_active",
        "primary_key": {"name": "chunk_id", "type": "VARCHAR", "max_length": 160},
        "vector_fields": [
            {"name": "dense_vector", "type": "FLOAT_VECTOR", "dimension": embedding_dim, "metric": "IP"},
            {"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR", "metric": "IP"},
        ],
        "filter_fields": [
            *MILVUS_STRING_FIELDS[:4],
            "effective_at",
            "expires_at",
            *MILVUS_STRING_FIELDS[4:],
        ],
    }


def _epoch_seconds(value: datetime) -> int:
    return int(value.timestamp())


def chunk_to_milvus_row(
    chunk: Chunk,
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    *,
    embedding_dim: int,
) -> dict[str, object]:
    """Normalize optional metadata into the non-null Milvus v1 storage contract.

    Empty optional strings use ``""``. A non-expiring record uses ``expires_at=0``;
    retrieval filters must treat zero as no expiry.
    """

    if len(dense_vector) != embedding_dim:
        raise ValueError(
            f"Dense vector dimension mismatch: expected {embedding_dim}, got {len(dense_vector)}"
        )
    if not sparse_vector:
        raise ValueError("Sparse vector must contain at least one non-zero value.")

    enum_fields = {
        "chunk_type": chunk.chunk_type.value,
        "approval_status": chunk.approval_status.value,
    }
    row: dict[str, object] = {
        "chunk_id": chunk.chunk_id,
        "dense_vector": [float(value) for value in dense_vector],
        "sparse_vector": {int(index): float(value) for index, value in sparse_vector.items()},
        "effective_at": _epoch_seconds(chunk.effective_at),
        "expires_at": _epoch_seconds(chunk.expires_at) if chunk.expires_at else 0,
    }
    for field in MILVUS_STRING_FIELDS:
        value = enum_fields.get(field, getattr(chunk, field, None))
        normalized = "" if value is None else str(value)
        if len(normalized) > 160:
            raise ValueError(f"Milvus field exceeds 160 characters: {field}")
        row[field] = normalized
    return row
