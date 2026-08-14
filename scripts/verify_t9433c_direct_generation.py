"""Probe live controlled direct-reply streaming without retrieval or tool calls."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from semikb.agent_runtime.direct_reply import (
    DirectReplyGenerator,
    DirectReplyKind,
    DirectReplyRequest,
)
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def run_probe() -> dict:
    settings = Settings(demo_mode=False)
    generator = DirectReplyGenerator(settings, OpenAICompatibleLLMGateway(settings))
    context = {
        "summary": "用户正在了解 SEMIKB 的连续对话能力。",
        "recent_messages": [
            {
                "message_id": "msg_question",
                "role": "user",
                "content": "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
            },
            {
                "message_id": "msg_answer",
                "role": "assistant",
                "content": "先暂停后续 lot 放行，并复核 chamber pressure 与 RF match。",
            },
        ],
    }
    cases = [
        DirectReplyRequest(
            kind=DirectReplyKind.GENERAL_CHAT,
            user_request="你好，你能做什么？",
            conversation_context=context,
        ),
        DirectReplyRequest(
            kind=DirectReplyKind.HISTORY_RECALL,
            user_request="我刚才说什么？",
            conversation_context=context,
            context_message_ids=("msg_question",),
        ),
        DirectReplyRequest(
            kind=DirectReplyKind.HISTORY_TRANSFORM,
            user_request="把上一段回答说简单一点",
            conversation_context=context,
            context_message_ids=("msg_answer",),
            action="simplify",
        ),
        DirectReplyRequest(
            kind=DirectReplyKind.CLARIFICATION,
            user_request="帮我调查良率下降",
            conversation_context=context,
            missing_slots=("product", "time_range"),
            clarification_questions=("受影响的 Product 是什么？", "需要查询哪个时间范围？"),
        ),
        DirectReplyRequest(
            kind=DirectReplyKind.REFUSAL,
            user_request="替我完成一个与半导体无关的外部任务",
            conversation_context=context,
            reason_codes=("outside_semikb_capability",),
            alternative_codes=("capability_guidance",),
        ),
    ]
    results = []
    for case in cases:
        deltas: list[str] = []
        result = await generator.generate(
            case,
            lambda delta, provider, model: deltas.append(delta),
        )
        if "".join(deltas) != result.text:
            raise RuntimeError(f"stream/persistence mismatch for {case.kind.value}")
        if case.kind is DirectReplyKind.HISTORY_RECALL and "msg_question" in result.text:
            raise RuntimeError("history recall exposed an internal message ID")
        results.append(
            {
                "reply_kind": case.kind.value,
                "response": result.text,
                "delta_count": len(deltas),
                "audit": result.audit.model_dump(mode="json"),
            }
        )
    live_stream_count = sum(
        item["audit"]["generation_mode"] == "llm_stream" for item in results
    )
    if live_stream_count < 4:
        raise RuntimeError("fewer than four direct reply kinds passed live unit validation")
    return {
        "verification": "T9-4.3.3c-live-direct-generation",
        "live_stream_count": live_stream_count,
        "cases": results,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run_probe())
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
