"""Local BGE-M3 and reranker integration boundary.

The optional FlagEmbedding dependency is deliberately loaded only when a real local
model path is configured; the synthetic demo stays lightweight and deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from semikb.config import Settings


@dataclass(frozen=True, slots=True)
class HybridEmbedding:
    dense: list[float]
    sparse: dict[int, float]


class BgeM3Encoder:
    """Produces BGE-M3 dense and sparse representations from a local model path."""

    def __init__(self, settings: Settings) -> None:
        if not settings.bge_m3_model_path:
            raise RuntimeError("BGE_M3_MODEL_PATH is required outside DEMO_MODE.")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - optional runtime feature
            raise RuntimeError("Install the 'rag' dependency group to enable BGE-M3.") from exc
        self._model = BGEM3FlagModel(settings.bge_m3_model_path, use_fp16=True)

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        result = self._model.encode(list(texts), return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return [
            HybridEmbedding(dense=list(dense), sparse={int(key): float(value) for key, value in sparse.items()})
            for dense, sparse in zip(result["dense_vecs"], result["lexical_weights"], strict=True)
        ]


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
