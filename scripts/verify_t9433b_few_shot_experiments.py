"""Run the four T9-4.3.3b shadow arms on one frozen intent dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentExampleBank
from semikb.agent_runtime.intent_experiments import (
    DynamicIntentExampleSelector,
    IntentExperimentProfile,
)
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner
from semikb.evaluation.intent_experiments import build_intent_experiment_comparison
from semikb.rag_retrieval.encoders import create_hybrid_encoder

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "intent_sets" / "semikb_intent_v3.json",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v1.json",
    )
    parser.add_argument(
        "--example-bank",
        type=Path,
        default=ROOT / "data" / "intent_examples" / "intent_example_bank_v1.json",
    )
    parser.add_argument("--dynamic-top-k", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    catalog = IntentCatalog.load(args.catalog.resolve())
    bank = IntentExampleBank.load(args.example_bank.resolve())
    dataset = IntentEvaluationDataset.load(args.dataset.resolve())
    bank.validate_against_catalog(catalog)
    bank.assert_no_evaluation_leakage(case.utterance for case in dataset.cases)

    settings = Settings(demo_mode=False)
    selector = DynamicIntentExampleSelector(
        bank,
        create_hybrid_encoder(settings),
        top_k=args.dynamic_top_k,
        batch_size=settings.embedding_batch_size,
    )
    profiles = (
        IntentExperimentProfile.current_prompt(),
        IntentExperimentProfile.production_baseline(),
        IntentExperimentProfile.fixed_few_shot(bank),
        IntentExperimentProfile.dynamic_few_shot(bank, selector),
    )
    results = {}
    for profile in profiles:
        print(f"starting {profile.arm.value}", flush=True)
        understanding = ConversationUnderstandingService(
            settings,
            OpenAICompatibleLLMGateway(settings),
            intent_catalog=catalog,
            experiment_profile=profile,
        )
        result = await IntentEvaluationRunner(understanding).run(
            dataset,
            offset=args.offset,
            limit=args.limit,
        )
        results[profile.arm] = result
        print(
            f"completed {profile.arm.value}: failures={len(result.failures)}, "
            f"calls={result.capacity['understanding_provider_calls']}",
            flush=True,
        )

    full_dataset_run = args.offset == 0 and args.limit is None
    comparison = build_intent_experiment_comparison(
        results,
        full_dataset_run=full_dataset_run,
    )
    return {
        "verification": "T9-4.3.3b",
        "experiment_mode": "offline_shadow_with_online_providers",
        "online_default_changed": False,
        "dataset": {
            "version": dataset.dataset_version,
            "hash": dataset.dataset_hash,
            "total_cases": len(dataset.cases),
            "evaluated_offset": args.offset,
            "evaluated_limit": args.limit,
        },
        "catalog": {
            "version": catalog.catalog_version,
            "hash": catalog.catalog_hash,
            "active_cards": len(catalog.active_cards),
        },
        "example_bank": {
            "version": bank.example_bank_version,
            "hash": bank.example_bank_hash,
            "examples": len(bank.examples),
            "evaluation_leakage": 0,
            "dynamic_top_k": args.dynamic_top_k,
        },
        "arms": {
            arm.value: result.model_dump(mode="json") for arm, result in results.items()
        },
        "comparison": comparison,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {args.output}", flush=True)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
