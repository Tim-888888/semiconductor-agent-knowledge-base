"""Milvus schema contract kept independent from the pymilvus runtime adapter."""

from __future__ import annotations


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
            "document_id",
            "revision",
            "chunk_type",
            "approval_status",
            "effective_at",
            "expires_at",
            "fab",
            "product",
            "process_layer",
            "tool_id",
            "chamber",
            "recipe_id",
            "recipe_version",
            "access_scope_key",
            "index_version",
        ],
    }
