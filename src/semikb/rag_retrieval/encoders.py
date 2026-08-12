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
    """Use Qwen for Dense vectors and a model-free encoder for Sparse vectors."""

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
        self._settings = settings
        self._transport = transport
        self._sparse = LexicalHashSparseEncoder(settings.sparse_encoder_version)
        self.model_name = settings.embedding_model
        self.sparse_encoder_version = settings.sparse_encoder_version

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        if not texts:
            return []
        dense_vectors = self._request_dense(texts)
        sparse_vectors = self._sparse.encode(texts)
        return [
            HybridEmbedding(dense=dense, sparse=sparse)
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def _request_dense(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": {"texts": list(texts)},
            "parameters": {"dimension": self._settings.embedding_dim},
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
            vectors = [
                _normalize_dense([float(value) for value in item["embedding"]])
                for item in ordered
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                "Qianwen embedding returned an invalid response."
            ) from exc
        if len(vectors) != len(texts):
            raise EmbeddingProviderError("Qianwen embedding omitted one or more texts.")
        if any(len(vector) != self._settings.embedding_dim for vector in vectors):
            raise EmbeddingProviderError("Qianwen embedding returned an unexpected dimension.")
        return vectors


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
