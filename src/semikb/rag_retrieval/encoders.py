"""Local BGE-M3 and reranker integration boundary.

The optional FlagEmbedding dependency is deliberately loaded only when a real local
model path is configured; the synthetic demo stays lightweight and deterministic.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from semikb.config import Settings


@dataclass(frozen=True, slots=True)
class HybridEmbedding:
    dense: list[float]
    sparse: dict[int, float]


class HybridEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]: ...


class DeterministicHybridEncoder:
    """Lightweight encoder for tests and the synthetic demo only."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> HybridEmbedding:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        dense = [((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in dense)) or 1.0
        normalized = [value / norm for value in dense]
        tokens = re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]", text.lower())
        sparse: dict[int, float] = {}
        for token in tokens or ["empty"]:
            token_id = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
            sparse[token_id] = sparse.get(token_id, 0.0) + 1.0
        sparse_norm = math.sqrt(sum(value * value for value in sparse.values())) or 1.0
        return HybridEmbedding(
            dense=normalized,
            sparse={key: value / sparse_norm for key, value in sparse.items()},
        )


class BgeM3Encoder:
    """Produces BGE-M3 dense and sparse representations from a local model path."""

    def __init__(self, settings: Settings) -> None:
        if not settings.bge_m3_model_path:
            raise RuntimeError("BGE_M3_MODEL_PATH is required outside DEMO_MODE.")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - optional runtime feature
            raise RuntimeError("Install the 'rag' dependency group to enable BGE-M3.") from exc
        self._model = BGEM3FlagModel(
            settings.bge_m3_model_path,
            use_fp16=settings.bge_use_fp16,
        )

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        result = self._model.encode(list(texts), return_dense=True, return_sparse=True, return_colbert_vecs=False)
        embeddings: list[HybridEmbedding] = []
        for dense, sparse in zip(
            result["dense_vecs"],
            result["lexical_weights"],
            strict=True,
        ):
            dense_values = [float(value) for value in dense]
            sparse_values = {int(key): float(value) for key, value in sparse.items()}
            if not dense_values or not all(math.isfinite(value) for value in dense_values):
                raise ValueError("BGE-M3 returned a non-finite dense vector.")
            if not sparse_values or not all(math.isfinite(value) for value in sparse_values.values()):
                raise ValueError("BGE-M3 returned an empty or non-finite sparse vector.")
            embeddings.append(HybridEmbedding(dense=dense_values, sparse=sparse_values))
        return embeddings


class BgeReranker:
    """Optional cross-encoder reranker used after hybrid recall."""

    def __init__(self, settings: Settings) -> None:
        if not settings.bge_reranker_model_path:
            raise RuntimeError("BGE_RERANKER_MODEL_PATH is required outside DEMO_MODE.")
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:  # pragma: no cover - optional runtime feature
            raise RuntimeError("Install the 'rag' dependency group to enable BGE reranking.") from exc
        self._model = FlagReranker(settings.bge_reranker_model_path, use_fp16=True)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(score) for score in self._model.compute_score([[query, passage] for passage in passages])]
