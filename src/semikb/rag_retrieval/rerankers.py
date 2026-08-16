"""Reranker providers used after hybrid recall."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from semikb.config import Settings
from semikb_provider_resilience import (
    ProviderAttemptAudit,
    ProviderRetriesExhausted,
    ProviderRetryPolicy,
    failure_from_response,
    invalid_response,
    run_with_retry,
)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class RerankerError(RuntimeError):
    """Safe provider error that never includes credentials or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        provider_attempts: tuple[ProviderAttemptAudit, ...] = (),
    ) -> None:
        super().__init__(message)
        self.provider_attempts = provider_attempts


class QianwenReranker:
    """DashScope-compatible Qwen3 reranker client."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.rerank_api_base_url:
            raise RerankerError("RERANK_API_BASE_URL is required for Qianwen reranking.")
        if not settings.rerank_api_key:
            raise RerankerError("RERANK_API_KEY is required for Qianwen reranking.")
        if not settings.rerank_model:
            raise RerankerError("RERANK_MODEL is required for Qianwen reranking.")
        self._settings = settings
        self._transport = transport
        self.model_name = settings.rerank_model
        self.last_attempts: tuple[ProviderAttemptAudit, ...] = ()
        self._retry_policy = ProviderRetryPolicy(
            max_attempts=settings.provider_max_attempts,
            backoff_base_seconds=settings.provider_backoff_base_seconds,
            backoff_max_seconds=settings.provider_backoff_max_seconds,
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            self.last_attempts = ()
            return []
        self.last_attempts = ()
        try:
            result = run_with_retry(
                "qianwen-reranker",
                "rerank",
                self._retry_policy,
                lambda: self._score_once(query, passages),
            )
        except ProviderRetriesExhausted as exc:
            self.last_attempts = exc.attempts
            raise RerankerError(
                "Qianwen reranker is unavailable after bounded retries.",
                provider_attempts=exc.attempts,
            ) from exc
        self.last_attempts = result.attempts
        return result.value

    def _score_once(self, query: str, passages: Sequence[str]) -> list[float]:
        base_url = self._settings.rerank_api_base_url.rstrip("/")
        endpoint = (
            base_url
            if base_url.endswith("/text-rerank")
            else f"{base_url}/api/v1/services/rerank/text-rerank/text-rerank"
        )
        payload = {
            "model": self.model_name,
            "input": {"query": query, "documents": list(passages)},
            "parameters": {"return_documents": False, "top_n": len(passages)},
        }
        headers = {
            "Authorization": f"Bearer {self._settings.rerank_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(
            timeout=self._settings.rerank_timeout_seconds,
            transport=self._transport,
        ) as client:
            response = client.post(endpoint, headers=headers, json=payload)
        if response.is_error:
            raise failure_from_response(response, "Qianwen reranker request failed.")
        try:
            results = response.json()["output"]["results"]
            scores_by_index = {
                int(item["index"]): float(item["relevance_score"])
                for item in results
            }
            scores = [scores_by_index[index] for index in range(len(passages))]
        except (KeyError, TypeError, ValueError) as exc:
            raise invalid_response("Qianwen reranker returned an invalid response.") from exc
        if len(scores) != len(passages):
            raise invalid_response("Qianwen reranker omitted one or more candidates.")
        return scores


def create_reranker(settings: Settings) -> Reranker:
    provider = settings.rerank_provider.strip().lower()
    if provider == "qianwen":
        return QianwenReranker(settings)
    raise RerankerError(f"Unsupported reranker provider: {settings.rerank_provider}")
