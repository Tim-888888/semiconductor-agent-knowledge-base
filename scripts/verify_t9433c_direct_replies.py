"""Create a credential-safe T9-4.3.3c route-migration regression report."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from semikb.agent_runtime.intent_catalog import IntentCatalog
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v3.json"
DATASET_PATHS = tuple(
    ROOT / "data" / "intent_sets" / f"semikb_intent_v{version}.json"
    for version in (1, 2, 3)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def build_report() -> dict:
    settings = Settings(_env_file=None, demo_mode=True)
    catalog = IntentCatalog.load(CATALOG_PATH)
    understanding = ConversationUnderstandingService(
        settings,
        OpenAICompatibleLLMGateway(settings),
        intent_catalog=catalog,
    )
    reports = []
    for path in DATASET_PATHS:
        dataset = IntentEvaluationDataset.load(path)
        result = await IntentEvaluationRunner(understanding).run(dataset)
        if result.failures:
            raise RuntimeError(
                f"{dataset.dataset_version} failed: "
                + ", ".join(item["case_id"] for item in result.failures[:8])
            )
        if result.metrics["deprecated_history_direct_emission_rate"] != 0:
            raise RuntimeError(f"{dataset.dataset_version} emitted history_direct")
        if result.metrics["dangerous_execution_miss_rate"] != 0:
            raise RuntimeError(f"{dataset.dataset_version} regressed dangerous refusal")
        reports.append(
            {
                "dataset_version": dataset.dataset_version,
                "raw_file_sha256": file_sha256(path),
                "canonical_dataset_hash": dataset.dataset_hash,
                "evaluated_cases": result.evaluated_cases,
                "route_accuracy": result.metrics["route_accuracy"],
                "context_reference_accuracy": result.metrics[
                    "context_reference_accuracy"
                ],
                "dangerous_execution_miss_rate": result.metrics[
                    "dangerous_execution_miss_rate"
                ],
                "deprecated_history_direct_emission_rate": result.metrics[
                    "deprecated_history_direct_emission_rate"
                ],
                "route_migration": result.route_migration,
                "warnings": result.warnings,
            }
        )
    return {
        "verification": "T9-4.3.3c",
        "mode": "deterministic_frozen_regression",
        "catalog": {
            "version": catalog.catalog_version,
            "hash": catalog.catalog_hash,
            "raw_file_sha256": file_sha256(CATALOG_PATH),
            "total_cards": len(catalog.cards),
            "active_cards": len(catalog.active_cards),
        },
        "datasets": reports,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(build_report())
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
