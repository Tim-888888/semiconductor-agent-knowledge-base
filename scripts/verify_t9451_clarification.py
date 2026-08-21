"""Run the frozen T9-4.5.1 multi-turn clarification acceptance set."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from semikb.agent_runtime.service import ConversationService
from semikb.config import Settings
from semikb.contracts.models import ActorScope
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.memory import DemoStore

INTERNAL_FIELDS = ("product", "time_range", "tool_or_chamber", "request_goal", "intent_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation_specs/t9451_clarification_scenarios_v1.json"),
    )
    return parser.parse_args()


def request_audit(store: DemoStore, result: dict[str, Any]):
    thread = result["thread"]
    request_id = thread["messages"][-1]["request_id"]
    return store.get_message_request(
        thread["thread_id"],
        thread["actor_scope"]["user_id"],
        request_id,
    )


async def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    counters = {
        "turns": 0,
        "relation_errors": 0,
        "route_errors": 0,
        "new_task_absorptions": 0,
        "cancel_failures": 0,
        "cross_frame_pollutions": 0,
        "duplicate_clarifications": 0,
        "clarification_comparisons": 0,
        "no_progress_turns": 0,
        "internal_field_leaks": 0,
        "premature_downstream_calls": 0,
        "business_clarification_rounds": 0,
        "closed_scenarios": 0,
    }

    for scenario in dataset["scenarios"]:
        store = DemoStore()
        ingestion = IngestionService(store)
        ingestion.seed_demo_corpus(root / "data" / "fixtures" / "demo_corpus.json")
        service = ConversationService(
            store,
            RetrievalService(store),
            Settings(_env_file=None, demo_mode=True),
        )
        thread = service.create_thread(scenario["case_id"], ActorScope())
        first = await service.send_message(thread.thread_id, scenario["initial_request"])
        if not first["clarification_required"]:
            failures.append(f"{scenario['case_id']}: initial request did not clarify")
            continue
        previous_signature: str | None = None
        final_result = first

        for turn in scenario["turns"]:
            counters["turns"] += 1
            traces_before = len(store.traces)
            result = await service.send_message(thread.thread_id, turn["input"])
            final_result = result
            record = request_audit(store, result)
            audit = record.clarification_transition_audit if record else None
            actual_relation = audit.relation.value if audit else None
            if actual_relation != turn["expected_relation"]:
                counters["relation_errors"] += 1
                failures.append(
                    f"{scenario['case_id']}: relation {actual_relation} != {turn['expected_relation']}"
                )
            if result["route_decision"].value != turn["expected_route"]:
                counters["route_errors"] += 1
                failures.append(
                    f"{scenario['case_id']}: route {result['route_decision'].value} != {turn['expected_route']}"
                )
            if result["clarification_required"] is not turn["clarification_required"]:
                failures.append(f"{scenario['case_id']}: clarification state mismatch")
            if turn.get("expected_status") and result.get("status") != turn["expected_status"]:
                failures.append(f"{scenario['case_id']}: terminal status mismatch")
            if not turn["downstream_allowed"] and len(store.traces) != traces_before:
                counters["premature_downstream_calls"] += 1
                failures.append(f"{scenario['case_id']}: downstream trace created while blocked")
            if turn["expected_relation"] == "replace_with_new_request" and result[
                "clarification_required"
            ]:
                counters["new_task_absorptions"] += 1
            if turn["expected_relation"] == "cancel_current" and result[
                "clarification_required"
            ]:
                counters["cancel_failures"] += 1
            if turn["expected_relation"] == "replace_with_new_request" and (
                result["thread"]["clarification_round"] != 0
                or result["thread"]["pending_fields"]
            ):
                counters["cross_frame_pollutions"] += 1
            if audit and "clarification_no_progress" in audit.warning_codes:
                counters["no_progress_turns"] += 1
            user_text = result["response"] + " " + " ".join(
                item["message"] for item in result["task_results"]
            )
            if any(field in user_text for field in INTERNAL_FIELDS):
                counters["internal_field_leaks"] += 1

            if result["clarification_required"]:
                counters["business_clarification_rounds"] += 1
                snapshot = service.graph.compiled.get_state(
                    {"configurable": {"thread_id": thread.thread_id}}
                )
                frame = snapshot.values.get("clarification_frame", {})
                signature = frame.get("signature") if isinstance(frame, dict) else None
                if (
                    previous_signature
                    and signature
                    and audit
                    and audit.relation.value == "continue_current"
                ):
                    counters["clarification_comparisons"] += 1
                    if signature == previous_signature:
                        counters["duplicate_clarifications"] += 1
                previous_signature = signature

        if not final_result["clarification_required"]:
            counters["closed_scenarios"] += 1

    scenario_count = len(dataset["scenarios"])
    replacement_count = sum(
        turn["expected_relation"] == "replace_with_new_request"
        for scenario in dataset["scenarios"]
        for turn in scenario["turns"]
    )
    cancel_count = sum(
        turn["expected_relation"] == "cancel_current"
        for scenario in dataset["scenarios"]
        for turn in scenario["turns"]
    )
    metrics = {
        "scenario_completion_rate": counters["closed_scenarios"] / max(scenario_count, 1),
        "relation_accuracy": 1 - counters["relation_errors"] / max(counters["turns"], 1),
        "route_accuracy": 1 - counters["route_errors"] / max(counters["turns"], 1),
        "new_task_absorption_rate": counters["new_task_absorptions"] / max(replacement_count, 1),
        "cancel_failure_rate": counters["cancel_failures"] / max(cancel_count, 1),
        "cross_frame_pollution_rate": counters["cross_frame_pollutions"]
        / max(replacement_count, 1),
        "duplicate_clarification_rate": counters["duplicate_clarifications"]
        / max(counters["clarification_comparisons"], 1),
        "no_progress_turn_rate": counters["no_progress_turns"] / max(counters["turns"], 1),
        "internal_field_leakage_rate": counters["internal_field_leaks"]
        / max(counters["turns"], 1),
        "premature_downstream_call_rate": counters["premature_downstream_calls"]
        / max(counters["turns"], 1),
        "average_clarification_rounds": counters["business_clarification_rounds"]
        / max(scenario_count, 1),
    }
    return {
        "dataset_version": dataset["dataset_version"],
        "scenario_count": scenario_count,
        "turn_count": counters["turns"],
        "metrics": metrics,
        "failures": failures,
        "passed": not failures,
    }


async def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("production_usage") != "forbidden":
        raise ValueError("clarification evaluation data must be isolated from production routing")
    report = await evaluate(dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
