"""Produce a credential-safe T9-4.3.3a governance and intent report."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentExampleBank
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("deterministic", "online"),
        default="deterministic",
    )
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    catalog = IntentCatalog.load(args.catalog.resolve())
    example_bank = IntentExampleBank.load(args.example_bank.resolve())
    dataset = IntentEvaluationDataset.load(args.dataset.resolve())
    example_bank.validate_against_catalog(catalog)
    example_bank.assert_no_evaluation_leakage(case.utterance for case in dataset.cases)

    settings = Settings(demo_mode=args.mode == "deterministic")
    understanding = ConversationUnderstandingService(
        settings,
        OpenAICompatibleLLMGateway(settings),
        intent_catalog=catalog,
    )
    result = await IntentEvaluationRunner(understanding).run(
        dataset,
        offset=args.offset,
        limit=args.limit,
    )
    if args.mode == "online" and result.capacity["llm_evaluated_cases"]:
        if result.capacity["all_active_cards_injected_rate"] != 1.0:
            raise RuntimeError("not every online LLM case received the complete active catalog")

    return {
        "verification": "T9-4.3.3a",
        "mode": args.mode,
        "catalog": {
            "version": catalog.catalog_version,
            "hash": catalog.catalog_hash,
            "total_cards": len(catalog.cards),
            "active_cards": len(catalog.active_cards),
        },
        "example_bank": {
            "version": example_bank.example_bank_version,
            "hash": example_bank.example_bank_hash,
            "examples": len(example_bank.examples),
            "evaluation_leakage": 0,
        },
        "evaluation": result.model_dump(mode="json"),
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {args.output}")
    else:
        print(serialized, end="")
    failures = payload["evaluation"]["failures"]
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
