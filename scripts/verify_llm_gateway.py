"""Run a credential-safe primary LLM smoke test from the repository-root .env."""

from __future__ import annotations

import argparse
import asyncio
import json

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.config import get_settings


async def verify(*, allow_fallback: bool) -> dict[str, object]:
    gateway = OpenAICompatibleLLMGateway(get_settings())
    result = await gateway.complete(
        [
            {
                "role": "system",
                "content": "Return one JSON object only. Do not include markdown.",
            },
            {
                "role": "user",
                "content": 'Return {"status":"ok","purpose":"semikb-llm-smoke"}.',
            },
        ],
        response_json=True,
        max_output_tokens=80,
        allow_fallback=allow_fallback,
    )
    body = json.loads(result.content)
    if body.get("status") != "ok" or body.get("purpose") != "semikb-llm-smoke":
        raise RuntimeError("LLM returned valid JSON but failed the smoke-test contract")
    return {
        "status": "ok",
        "provider": result.provider,
        "requested_model": result.requested_model,
        "reported_model": result.reported_model,
        "fallback_used": result.fallback_used,
        "attempted_providers": result.attempted_providers,
        "total_tokens": result.usage.get("total_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow the configured fallback provider when the primary provider fails.",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(allow_fallback=args.allow_fallback)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
