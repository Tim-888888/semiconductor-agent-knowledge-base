"""Probe configured LLM Providers for real OpenAI-compatible streaming deltas."""

from __future__ import annotations

import argparse
import asyncio
import json

from semikb.agent_runtime.llm_gateway import LLMProviderError, OpenAICompatibleLLMGateway
from semikb.config import get_settings

PROBE_MARKER = "SEMIKB_STREAM_OK"
PROBE_TEXT = (
    "SEMIKB_STREAM_OK alpha beta gamma delta epsilon zeta eta theta iota kappa "
    "lambda mu nu xi omicron pi rho sigma tau"
)


async def verify_provider(provider: str) -> dict[str, object]:
    gateway = OpenAICompatibleLLMGateway(get_settings())
    result = await gateway.probe_stream(
        provider,
        [
            {
                "role": "system",
                "content": "Return the requested text exactly. Do not add markdown or commentary.",
            },
            {"role": "user", "content": f"Return exactly: {PROBE_TEXT}"},
        ],
        max_output_tokens=128,
    )
    if PROBE_MARKER not in result.content:
        raise LLMProviderError(provider, "streamed content failed the probe marker")
    return {
        "status": "ok",
        "provider": result.provider,
        "requested_model": result.requested_model,
        "reported_model": result.reported_model,
        "event_count": result.event_count,
        "content_delta_count": result.content_delta_count,
        "reasoning_delta_count": result.reasoning_delta_count,
        "content_length": result.content_length,
        "content_sha256": result.content_sha256,
        "first_event_ms": result.first_event_ms,
        "first_content_delta_ms": result.first_content_delta_ms,
        "total_ms": result.total_ms,
        "finish_reason": result.finish_reason,
        "done_received": result.done_received,
        "termination": result.termination,
        "total_tokens": result.usage.get("total_tokens"),
    }


async def verify(providers: list[str]) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for provider in providers:
        try:
            results.append(await verify_provider(provider))
        except LLMProviderError as exc:
            results.append(
                {
                    "status": "error",
                    "provider": provider,
                    "error": str(exc),
                    "status_code": exc.status_code,
                }
            )
    status = "ok" if all(result["status"] == "ok" for result in results) else "error"
    return {"status": status, "providers": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("all", "closeai", "qwen"),
        default="all",
        help="Probe both configured Providers or one explicit Provider.",
    )
    args = parser.parse_args()
    providers = ["closeai", "qwen"] if args.provider == "all" else [args.provider]
    result = asyncio.run(verify(providers))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
