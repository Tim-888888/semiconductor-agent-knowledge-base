"""OpenAI-compatible Qwen vision client with strict, safe output validation."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import httpx

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.providers.types import VisionAnalysis
from semikb_provider_resilience import (
    ProviderAttemptAudit,
    ProviderRetriesExhausted,
    ProviderRetryPolicy,
    failure_from_response,
    invalid_response,
    run_with_retry,
)


@dataclass(frozen=True, slots=True)
class QwenVisionConfig:
    base_url: str
    api_key: str
    model: str = "qwen3.7-plus"
    timeout_seconds: float = 60
    max_attempts: int = 2
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 2.0


class QwenVisionClient:
    provider_name = "qwen-vl"

    def __init__(
        self,
        config: QwenVisionConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not config.base_url.strip() or not config.api_key.strip():
            raise IngestError(
                IngestErrorCode.PARSER_NOT_CONFIGURED,
                "The Qwen image understanding endpoint and API key are required.",
            )
        if not config.model.strip():
            raise IngestError(
                IngestErrorCode.PARSER_NOT_CONFIGURED,
                "A Qwen image understanding model name is required.",
            )
        self.config = config
        self._transport = transport
        self.last_attempts: tuple[ProviderAttemptAudit, ...] = ()
        self._retry_policy = ProviderRetryPolicy(
            max_attempts=config.max_attempts,
            backoff_base_seconds=config.backoff_base_seconds,
            backoff_max_seconds=config.backoff_max_seconds,
        )

    @property
    def provider_version(self) -> str:
        return self.config.model

    def analyze_image(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        correlation_id: str,
    ) -> VisionAnalysis:
        self.last_attempts = ()
        try:
            result = run_with_retry(
                self.provider_name,
                "image_understanding",
                self._retry_policy,
                lambda: self._analyze_once(
                    filename=filename,
                    content_type=content_type,
                    content=content,
                    correlation_id=correlation_id,
                ),
            )
        except ProviderRetriesExhausted as exc:
            self.last_attempts = exc.attempts
            code = (
                IngestErrorCode.PARSER_TIMEOUT
                if exc.failure.failure_kind.value == "timeout"
                else IngestErrorCode.PARSE_FAILED
                if exc.failure.failure_kind.value in {"invalid_response", "rejected"}
                else IngestErrorCode.PARSER_UNAVAILABLE
            )
            message = (
                "The image understanding provider returned an invalid structured result."
                if code is IngestErrorCode.PARSE_FAILED
                else "The image understanding provider timed out after bounded retries."
                if code is IngestErrorCode.PARSER_TIMEOUT
                else "The image understanding provider is unavailable after bounded retries."
            )
            raise IngestError(
                code,
                message,
                provider_attempts=exc.attempts,
            ) from exc
        self.last_attempts = result.attempts
        return result.value

    def _analyze_once(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        correlation_id: str,
    ) -> VisionAnalysis:
        encoded = base64.b64encode(content).decode("ascii")
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You analyze semiconductor manufacturing and inspection images. "
                        "Return only a JSON object with caption, ocr_text, detection_summary, "
                        "confidence, and detected_language. Describe only visible evidence; "
                        "never invent lot, tool, chamber, recipe, alarm, or measurement values."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{encoded}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Analyze uploaded image {filename!r}. OCR exact visible text, "
                                "summarize visual patterns or defects, and keep uncertainty explicit."
                            ),
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "X-SemiKB-Correlation-ID": correlation_id,
        }
        with httpx.Client(
            timeout=self.config.timeout_seconds,
            transport=self._transport,
        ) as client:
            response = client.post(self._endpoint(), headers=headers, json=payload)

        if response.status_code >= 400:
            raise failure_from_response(response, "The image understanding request failed.")

        try:
            body: dict[str, Any] = response.json()
            content_value = body["choices"][0]["message"]["content"]
            raw = self._parse_json_content(content_value)
            return VisionAnalysis.model_validate(raw)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise invalid_response(
                "The image understanding provider returned an invalid structured result."
            ) from exc

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    @staticmethod
    def _parse_json_content(content: object) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise TypeError("Expected text or object content.")
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.removeprefix("```json").removeprefix("```")
            stripped = stripped.removesuffix("```").strip()
        result = json.loads(stripped)
        if not isinstance(result, dict):
            raise TypeError("Expected a JSON object.")
        return result
