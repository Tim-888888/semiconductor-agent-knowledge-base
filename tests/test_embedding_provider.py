from __future__ import annotations

import math

import httpx
import pytest

from semikb.config import Settings
from semikb.rag_retrieval.encoders import (
    EmbeddingProviderError,
    LexicalHashSparseEncoder,
    QianwenHybridEncoder,
)


def settings(**updates: object) -> Settings:
    values = {
        "_env_file": None,
        "demo_mode": False,
        "embedding_api_base_url": "https://example.test/embedding/text-embedding",
        "embedding_api_key": "test-key",
        "embedding_model": "qwen3.7-text-embedding",
        "embedding_dim": 4,
    }
    values.update(updates)
    return Settings(**values)


def test_qianwen_encoder_returns_normalized_dense_and_model_free_sparse() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"text_index": 1, "embedding": [0.0, 3.0, 0.0, 4.0]},
                        {"text_index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                    ]
                }
            },
        )

    encoder = QianwenHybridEncoder(
        settings(),
        transport=httpx.MockTransport(handler),
    )
    embeddings = encoder.encode(["ETCH-03 清腔", "ETCH-03 清腔后验证"])

    assert captured["url"] == "https://example.test/embedding/text-embedding"
    assert captured["authorization"] == "Bearer test-key"
    assert '"dimension":4' in str(captured["payload"]).replace(" ", "")
    assert embeddings[0].dense == [1.0, 0.0, 0.0, 0.0]
    assert embeddings[1].dense == [0.0, 0.6, 0.0, 0.8]
    assert all(embedding.sparse for embedding in embeddings)
    assert all(
        math.isclose(sum(value * value for value in embedding.sparse.values()), 1.0)
        for embedding in embeddings
    )


def test_qianwen_encoder_can_reuse_reranker_key_without_exposing_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer shared-key"
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"text_index": 0, "embedding": [1, 0]}]}},
        )

    encoder = QianwenHybridEncoder(
        settings(embedding_api_key="", rerank_api_key="shared-key", embedding_dim=2),
        transport=httpx.MockTransport(handler),
    )

    assert len(encoder.encode(["pressure alarm"])[0].dense) == 2


def test_qianwen_encoder_rejects_wrong_dimension_and_safe_http_errors() -> None:
    wrong_dimension = QianwenHybridEncoder(
        settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"output": {"embeddings": [{"text_index": 0, "embedding": [1, 2]}]}},
            )
        ),
    )
    with pytest.raises(EmbeddingProviderError, match="unexpected dimension"):
        wrong_dimension.encode(["text"])

    provider_error = QianwenHybridEncoder(
        settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(429, json={"secret": "x"})),
    )
    with pytest.raises(EmbeddingProviderError, match="HTTP 429") as exc:
        provider_error.encode(["text"])
    assert "secret" not in str(exc.value)


def test_lexical_sparse_encoder_is_stable_and_preserves_semiconductor_terms() -> None:
    encoder = LexicalHashSparseEncoder()
    first, second = encoder.encode(
        [
            "ETCH-03 Chamber B 清腔后检查 pressure alarm",
            "ETCH-03 Chamber B 清腔后检查 pressure alarm",
        ]
    )

    assert first == second
    assert len(first) >= 8
    assert all(isinstance(key, int) and value > 0 for key, value in first.items())
