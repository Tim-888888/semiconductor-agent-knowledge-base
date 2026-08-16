"""Run credential-safe live smoke checks for T9-4.5 external Providers."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from PIL import Image, ImageDraw

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings, get_settings
from semikb.rag_retrieval.encoders import QianwenHybridEncoder
from semikb.rag_retrieval.rerankers import QianwenReranker
from semikb_ingest.providers.mineru_pdf import MinerUPdfClient, MinerUPdfConfig
from semikb_ingest.providers.qwen_vision import QwenVisionClient, QwenVisionConfig
from semikb_provider_resilience import ProviderAttemptAudit


def _attempts(items: tuple[ProviderAttemptAudit, ...]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def _skipped(provider: str, operation: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "operation": operation,
        "status": "skipped_not_configured",
        "attempts": [],
    }


def _failed(
    provider: str,
    operation: str,
    attempts: tuple[ProviderAttemptAudit, ...],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "operation": operation,
        "status": "failed",
        "failure_code": "provider_smoke_failed",
        "attempts": _attempts(attempts),
    }


def _run_sync(
    provider: str,
    operation: str,
    call: Callable[[], dict[str, Any]],
    attempts: Callable[[], tuple[ProviderAttemptAudit, ...]],
) -> dict[str, Any]:
    try:
        return call()
    except Exception:  # The report intentionally suppresses Provider response details.
        return _failed(provider, operation, attempts())


async def _check_llm(settings: Settings) -> dict[str, Any]:
    if not settings.closeai_api_key and not settings.qwen_api_key:
        return _skipped("llm", "chat_completion")
    gateway = OpenAICompatibleLLMGateway(settings)
    try:
        result = await gateway.complete(
            [
                {"role": "system", "content": "Return one JSON object only."},
                {"role": "user", "content": 'Return {"status":"ok"}.'},
            ],
            response_json=True,
            max_output_tokens=48,
            allow_fallback=True,
        )
        body = json.loads(result.content)
        if body.get("status") != "ok":
            raise ValueError("smoke contract mismatch")
        return {
            "provider": result.provider,
            "operation": "chat_completion",
            "status": "ok",
            "model": result.reported_model or result.requested_model,
            "fallback_used": result.fallback_used,
            "attempts": _attempts(result.provider_attempts),
        }
    except Exception:
        return _failed("llm", "chat_completion", gateway.last_attempts)


def _check_embedding(settings: Settings) -> dict[str, Any]:
    if not settings.resolved_embedding_api_key:
        return _skipped("qianwen-embedding", "dense_sparse_embedding")
    client = QianwenHybridEncoder(settings)

    def call() -> dict[str, Any]:
        item = client.encode(["半导体腔体清洁后的首片异常检查"])[0]
        return {
            "provider": "qianwen-embedding",
            "operation": "dense_sparse_embedding",
            "status": "ok",
            "model": client.model_name,
            "dense_dimension": len(item.dense),
            "sparse_terms": len(item.sparse),
            "attempts": _attempts(client.last_attempts),
        }

    return _run_sync(
        "qianwen-embedding",
        "dense_sparse_embedding",
        call,
        lambda: client.last_attempts,
    )


def _check_reranker(settings: Settings) -> dict[str, Any]:
    if not settings.rerank_api_key:
        return _skipped("qianwen-reranker", "rerank")
    client = QianwenReranker(settings)

    def call() -> dict[str, Any]:
        scores = client.score(
            "腔体清洁后首片异常先检查什么",
            ["检查 chamber pressure 和 RF match。", "检查办公网络连接。"],
        )
        return {
            "provider": "qianwen-reranker",
            "operation": "rerank",
            "status": "ok",
            "model": client.model_name,
            "score_count": len(scores),
            "attempts": _attempts(client.last_attempts),
        }

    return _run_sync(
        "qianwen-reranker",
        "rerank",
        call,
        lambda: client.last_attempts,
    )


def _sample_image() -> bytes:
    image = Image.new("RGB", (480, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 450, 190), outline="black", width=3)
    draw.text((55, 85), "ETCH-03  PRESSURE TREND", fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _sample_pdf() -> bytes:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 80), "Semiconductor chamber maintenance note", fill="black")
    draw.text((80, 145), "Check pressure and RF match after cleaning.", fill="black")
    output = io.BytesIO()
    image.save(output, format="PDF", resolution=120)
    return output.getvalue()


def _check_vision(settings: Settings) -> dict[str, Any]:
    if not settings.qwen_api_base_url or not settings.qwen_api_key:
        return _skipped("qwen-vl", "image_understanding")
    client = QwenVisionClient(
        QwenVisionConfig(
            base_url=settings.qwen_api_base_url,
            api_key=settings.qwen_api_key,
            model=settings.qwen_vision_model,
            timeout_seconds=settings.qwen_vision_timeout_seconds,
            max_attempts=settings.provider_max_attempts,
            backoff_base_seconds=settings.provider_backoff_base_seconds,
            backoff_max_seconds=settings.provider_backoff_max_seconds,
        )
    )

    def call() -> dict[str, Any]:
        result = client.analyze_image(
            filename="provider-smoke.png",
            content_type="image/png",
            content=_sample_image(),
            correlation_id="t945-provider-smoke",
        )
        return {
            "provider": "qwen-vl",
            "operation": "image_understanding",
            "status": "ok",
            "model": client.provider_version,
            "caption_chars": len(result.caption),
            "attempts": _attempts(client.last_attempts),
        }

    return _run_sync(
        "qwen-vl",
        "image_understanding",
        call,
        lambda: client.last_attempts,
    )


def _check_mineru(settings: Settings) -> dict[str, Any]:
    if not settings.mineru_api_base_url or not settings.mineru_api_key:
        return _skipped("mineru", "pdf_extraction")
    client = MinerUPdfClient(
        MinerUPdfConfig(
            base_url=settings.mineru_api_base_url,
            api_key=settings.mineru_api_key,
            model_version=settings.mineru_model_version,
            timeout_seconds=settings.mineru_timeout_seconds,
            poll_seconds=settings.mineru_poll_seconds,
            max_attempts=settings.provider_max_attempts,
            backoff_base_seconds=settings.provider_backoff_base_seconds,
            backoff_max_seconds=settings.provider_backoff_max_seconds,
        )
    )

    def call() -> dict[str, Any]:
        result = client.parse_pdf(
            filename="provider-smoke.pdf",
            content=_sample_pdf(),
            correlation_id="t945-provider-smoke-pdf",
        )
        return {
            "provider": "mineru",
            "operation": "pdf_extraction",
            "status": "ok",
            "model": client.provider_version,
            "markdown_chars": len(result.markdown),
            "pages": result.pages,
            "attempts": _attempts(client.last_attempts),
        }

    return _run_sync(
        "mineru",
        "pdf_extraction",
        call,
        lambda: client.last_attempts,
    )


async def _check_web(settings: Settings) -> dict[str, Any]:
    if not settings.aliyun_web_mcp_api_key:
        return _skipped("aliyun-web-mcp", "web_search")
    client = AliyunWebSearchGateway(settings)
    try:
        results = await client.search("最新公开半导体制造技术资料")
        return {
            "provider": "aliyun-web-mcp",
            "operation": "web_search",
            "status": "ok",
            "allowed_result_count": len(results),
            "attempts": _attempts(client.last_attempts),
        }
    except Exception:
        return _failed("aliyun-web-mcp", "web_search", client.last_attempts)


async def verify() -> dict[str, Any]:
    settings = get_settings()
    results = [
        await _check_llm(settings),
        _check_embedding(settings),
        _check_reranker(settings),
        _check_vision(settings),
        _check_mineru(settings),
        await _check_web(settings),
    ]
    return {
        "stage": "T9-4.5",
        "generated_at": datetime.now(UTC).isoformat(),
        "credential_safe": True,
        "results": results,
        "summary": {
            "ok": sum(item["status"] == "ok" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "skipped": sum(item["status"].startswith("skipped") for item in results),
        },
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))
