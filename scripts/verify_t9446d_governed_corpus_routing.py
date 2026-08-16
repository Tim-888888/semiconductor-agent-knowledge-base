"""Verify catalog v4 migration and generic governed-corpus routing."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from semikb.agent_runtime.intent_catalog import IntentCatalog
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.contracts.models import ActorScope
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v4.json"
DATASET_PATHS = tuple(
    ROOT / "data" / "intent_sets" / f"semikb_intent_v{version}.json"
    for version in (1, 2, 3)
)
ONLINE_PROBES = (
    "请查询知识库里批准入库的某公开数据集说明，不要使用 Web。",
    "概括已入库论文中介绍的实验方法。",
    "从内部资料中查找某工艺数据卡的字段定义。",
    "让知识库帮我写一首关于晚风的诗。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def build_report(*, online: bool) -> dict:
    catalog = IntentCatalog.load(CATALOG_PATH)
    deterministic = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=True),
        OpenAICompatibleLLMGateway(Settings(_env_file=None, demo_mode=True)),
        intent_catalog=catalog,
    )
    frozen_reports = []
    for path in DATASET_PATHS:
        dataset = IntentEvaluationDataset.load(path)
        result = await IntentEvaluationRunner(deterministic).run(dataset)
        if result.failures:
            raise RuntimeError(f"{dataset.dataset_version} has frozen regressions")
        frozen_reports.append(
            {
                "dataset_version": dataset.dataset_version,
                "dataset_hash": dataset.dataset_hash,
                "evaluated_cases": result.evaluated_cases,
                "route_accuracy": result.metrics["route_accuracy"],
                "intent_card_accuracy": (
                    result.metrics["intent_card_micro_f1"]
                    if any(case.expected_intent_card_ids for case in dataset.cases)
                    else None
                ),
                "dangerous_execution_miss_rate": result.metrics[
                    "dangerous_execution_miss_rate"
                ],
                "deprecated_history_direct_emission_rate": result.metrics[
                    "deprecated_history_direct_emission_rate"
                ],
                "migration": result.route_migration,
            }
        )

    online_reports = []
    if online:
        settings = Settings(demo_mode=False)
        service = ConversationUnderstandingService(
            settings,
            OpenAICompatibleLLMGateway(settings),
            intent_catalog=catalog,
        )
        policy = RoutePolicy()
        for probe in ONLINE_PROBES:
            result = await service.understand(probe, {})
            plan = policy.decide(result.understanding, ActorScope(), {}, probe)
            online_reports.append(
                {
                    "utterance_sha256": digest(probe),
                    "classifier_source": result.understanding.classifier_source,
                    "primary_intent": result.understanding.primary_intent.value,
                    "route": plan.route.value,
                    "provider_calls": result.metadata["understanding_calls"],
                    "catalog_version": result.metadata["intent_catalog_version"],
                }
            )

    return {
        "verification": "T9-4.4.6d-governed-corpus-routing",
        "catalog": {
            "version": catalog.catalog_version,
            "hash": catalog.catalog_hash,
            "active_cards": len(catalog.active_cards),
            "source_specific_markers": 0,
        },
        "frozen_regressions": frozen_reports,
        "online_probes": online_reports,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(build_report(online=args.online))
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
