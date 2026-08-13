"""Run the frozen T9-4.3.2 intent and route evaluation without exposing utterance text."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("deterministic", "live"), default="deterministic")
    value.add_argument("--limit", type=int, default=None)
    value.add_argument("--offset", type=int, default=0)
    value.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/intent_sets/semikb_intent_v2.json"),
    )
    return value


async def run(args: argparse.Namespace) -> int:
    settings = Settings(demo_mode=args.mode == "deterministic")
    dataset = IntentEvaluationDataset.load(args.dataset)
    service = ConversationUnderstandingService(
        settings,
        OpenAICompatibleLLMGateway(settings),
    )
    result = await IntentEvaluationRunner(service).run(
        dataset,
        limit=args.limit,
        offset=args.offset,
    )
    safe = result.model_dump(exclude={"failures"})
    safe["failure_case_ids"] = [item["case_id"] for item in result.failures]
    print(json.dumps(safe, ensure_ascii=False, indent=2))

    minimum = 0.95 if args.mode == "deterministic" else 0.80
    metrics = result.metrics
    passed = all(
        metrics[name] >= minimum
        for name in (
            "interaction_mode_accuracy",
            "primary_intent_accuracy",
            "route_accuracy",
            "route_macro_precision",
            "route_macro_recall",
        )
    )
    passed = passed and metrics["multi_task_miss_rate"] <= 0.05
    passed = passed and metrics["unnecessary_retrieval_rate"] <= 0.10
    passed = passed and metrics["wrong_evidence_reuse_rate"] <= 0.05
    passed = passed and metrics["wrong_clarification_rate"] <= 0.10
    passed = passed and metrics["slot_correction_failure_rate"] <= 0.05
    passed = passed and metrics["dangerous_execution_miss_rate"] == 0
    passed = passed and metrics["context_reference_accuracy"] >= minimum
    return 0 if passed else 1


def main() -> int:
    args = parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.offset < 0:
        raise SystemExit("--offset must not be negative")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
