"""Dense and sparse embedding providers shared by ingestion and retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from semikb.config import Settings


@dataclass(frozen=True, slots=True)
class HybridEmbedding:
    dense: list[float]
    sparse: dict[int, float]


class HybridEncoder(Protocol):
    model_name: str
    sparse_encoder_version: str

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]: ...


class EmbeddingProviderError(RuntimeError):
    """Safe provider error that excludes credentials and response bodies."""


class LexicalHashSparseEncoder:
    """Create reproducible sparse vectors without loading a neural model."""

    def __init__(self, version: str = "lexical-hash-v1") -> None:
        self.version = version

    def encode(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> dict[int, float]:
        terms = self._terms(text)
        counts = Counter(terms or ["__empty__"])
        weighted = {
            self._term_id(term): 1.0 + math.log(float(count))
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {key: value / norm for key, value in weighted.items()}

    @staticmethod
    def _terms(text: str) -> list[str]:
        normalized = text.lower()
        ascii_terms = re.findall(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*", normalized)
        han_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
        han_terms: list[str] = []
        for run in han_runs:
            han_terms.extend(run)
            han_terms.extend(run[index : index + 2] for index in range(len(run) - 1))
        return ascii_terms + han_terms

    @staticmethod
    def _term_id(term: str) -> int:
        return int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:4], "big")


class DeterministicHybridEncoder:
    """Lightweight encoder for tests and the synthetic demo only."""

    model_name = "deterministic-demo"
    sparse_encoder_version = "lexical-hash-demo-v1"

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._sparse = LexicalHashSparseEncoder(self.sparse_encoder_version)

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        sparse_vectors = self._sparse.encode(texts)
        return [
            HybridEmbedding(dense=self._dense(text), sparse=sparse)
            for text, sparse in zip(texts, sparse_vectors, strict=True)
        ]

    def _dense(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dense = [
            ((digest[index % len(digest)] / 255.0) * 2) - 1
            for index in range(self._dimension)
        ]
        return _normalize_dense(dense)


class QianwenHybridEncoder:
    """Use Qwen Dense+Sparse output, with lexical Sparse only for v3 rollback."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if settings.embedding_provider.strip().lower() != "qianwen":
            raise EmbeddingProviderError(
                f"Unsupported embedding provider: {settings.embedding_provider}"
            )
        if not settings.embedding_api_base_url:
            raise EmbeddingProviderError("EMBEDDING_API_BASE_URL is required.")
        if not settings.resolved_embedding_api_key:
            raise EmbeddingProviderError(
                "EMBEDDING_API_KEY or RERANK_API_KEY is required."
            )
        if not settings.embedding_model:
            raise EmbeddingProviderError("EMBEDDING_MODEL is required.")
        output_type = settings.embedding_output_type.strip().lower()
        if output_type not in {"dense", "dense&sparse"}:
            raise EmbeddingProviderError(
                "EMBEDDING_OUTPUT_TYPE must be dense or dense&sparse."
            )
        self._settings = settings
        self._transport = transport
        self._output_type = output_type
        self._lexical_sparse = (
            LexicalHashSparseEncoder(settings.sparse_encoder_version)
            if output_type == "dense"
            else None
        )
        self.model_name = settings.embedding_model
        self.sparse_encoder_version = settings.sparse_encoder_version

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        if not texts:
            return []
        items = self._request_embeddings(texts)
        dense_vectors = [_provider_dense(item) for item in items]
        if self._lexical_sparse is None:
            sparse_vectors = [_provider_sparse(item) for item in items]
        else:
            sparse_vectors = self._lexical_sparse.encode(texts)
        if any(len(vector) != self._settings.embedding_dim for vector in dense_vectors):
            raise EmbeddingProviderError(
                "Qianwen embedding returned an unexpected dimension."
            )
        return [
            HybridEmbedding(dense=dense, sparse=sparse)
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def _request_embeddings(self, texts: Sequence[str]) -> list[dict[str, object]]:
        parameters: dict[str, object] = {"dimension": self._settings.embedding_dim}
        if self._output_type == "dense&sparse":
            parameters["output_type"] = self._output_type
        payload = {
            "model": self.model_name,
            "input": {"texts": list(texts)},
            "parameters": parameters,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.resolved_embedding_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(
                timeout=self._settings.embedding_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.post(
                    _embedding_endpoint(self._settings.embedding_api_base_url),
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError("Qianwen embedding transport failure.") from exc
        if response.is_error:
            raise EmbeddingProviderError(
                f"Qianwen embedding returned HTTP {response.status_code}."
            )
        try:
            items = response.json()["output"]["embeddings"]
            ordered = sorted(items, key=lambda item: int(item.get("text_index", 0)))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                "Qianwen embedding returned an invalid response."
            ) from exc
        if len(ordered) != len(texts):
            raise EmbeddingProviderError("Qianwen embedding omitted one or more texts.")
        try:
            text_indices = [int(item["text_index"]) for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                "Qianwen embedding returned invalid text indices."
            ) from exc
        if text_indices != list(range(len(texts))):
            raise EmbeddingProviderError(
                "Qianwen embedding returned duplicate or incomplete text indices."
            )
        return ordered


def create_hybrid_encoder(settings: Settings) -> HybridEncoder:
    if settings.demo_mode:
        return DeterministicHybridEncoder(settings.embedding_dim)
    return QianwenHybridEncoder(settings)


def _embedding_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/text-embedding"):
        return normalized
    return f"{normalized}/api/v1/services/embeddings/text-embedding/text-embedding"


def _normalize_dense(values: Sequence[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding provider returned an empty or non-finite Dense vector.")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding provider returned a zero-norm Dense vector.")
    return [float(value) / norm for value in values]


def _provider_sparse(item: dict[str, object]) -> dict[int, float]:
    try:
        sparse_items = item["sparse_embedding"]
        if not isinstance(sparse_items, list) or not sparse_items:
            raise ValueError("empty sparse embedding")
        vector: dict[int, float] = {}
        for sparse_item in sparse_items:
            if not isinstance(sparse_item, dict):
                raise TypeError("invalid sparse item")
            index = int(sparse_item["index"])
            value = float(sparse_item["value"])
            if index < 0 or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid sparse index or value")
            if index in vector:
                raise ValueError("duplicate sparse index")
            vector[index] = value
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingProviderError(
            "Qianwen embedding returned an invalid Sparse vector."
        ) from exc
    return vector


def _provider_dense(item: dict[str, object]) -> list[float]:
    try:
        values = item["embedding"]
        if not isinstance(values, list):
            raise TypeError("invalid dense embedding")
        return _normalize_dense([float(value) for value in values])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingProviderError(
            "Qianwen embedding returned an invalid Dense vector."
        ) from exc
