"""Build the reviewed semikb-intent-v3 regression set without changing v1/v2."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentTaskSignature

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "data" / "intent_sets" / "semikb_intent_v2.json"
V3_PATH = ROOT / "data" / "intent_sets" / "semikb_intent_v3.json"
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v1.json"


def task(
    primary_intent: str,
    target_type: str,
    action: str,
    *,
    execution_policy: str = "execute",
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "primary_intent": primary_intent,
        "target_type": target_type,
        "action": action,
        "execution_policy": execution_policy,
        "depends_on": depends_on or [],
    }


def numbered_tasks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"task_id": f"task_{index}", **item} for index, item in enumerate(items, start=1)]


def expected_tasks(case: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = case["case_id"]
    group = case_id.split("-")[0]
    index = int(case_id.rsplit("-", 1)[1]) if case_id.rsplit("-", 1)[1].isdigit() else 0
    text = case["utterance"].lower()

    if group == "history":
        items = [task("conversation", "previous_user_message", "recall")]
    elif group == "content":
        action = "summarize" if "总结" in text else "translate" if "翻译" in text else "simplify"
        items = [task("content_task", "previous_answer", action)]
    elif group == "conversation":
        target = "previous_answer" if index >= 5 else "general"
        items = [task("conversation", target, "explain")]
    elif group == "knowledge":
        if index in {3, 4}:
            items = [task("knowledge_query", "recipe", "explain")]
        elif index == 6:
            items = [task("knowledge_query", "sop", "compare")]
        elif index == 8:
            items = [task("knowledge_query", "alarm", "explain")]
        elif index == 11:
            items = [
                task("knowledge_query", "general", "lookup"),
                task("knowledge_query", "sop", "compare"),
            ]
        elif index >= 9:
            items = [task("knowledge_query", "general", "lookup")]
        else:
            items = [task("knowledge_query", "sop", "lookup")]
    elif group in {"data", "mixed"}:
        if "spc" in text or "趋势" in text:
            target = "spc"
        elif "lot" in text or "良率" in text or "wafer" in text:
            target = "lot"
        elif "fdc" in text or "报警" in text or "alarm" in text:
            target = "fdc"
        else:
            target = "general"
        items = [task("data_query", target, "lookup")]
    elif group == "investigation":
        items = []
        if "fdc" in text or "报警" in text or "alarm" in text:
            items.append(task("data_query", "fdc", "lookup"))
        elif "良率" in text or "wafer" in text or "晶圆图" in text:
            items.append(task("data_query", "lot", "lookup"))
        if "sop" in text:
            items.append(task("knowledge_query", "sop", "lookup"))
        dependencies = [f"task_{position}" for position in range(1, len(items) + 1)]
        items.append(
            task(
                "investigation",
                "case",
                "diagnose",
                depends_on=dependencies[-2:],
            )
        )
    elif group == "clarify":
        if index in {1, 8}:
            items = [
                task("data_query", "lot", "lookup"),
                task("investigation", "case", "diagnose", depends_on=["task_1"]),
            ]
        elif index in {5, 6, 7}:
            items = [task("data_query", "fdc", "lookup")]
        else:
            items = [task("investigation", "case", "diagnose")]
    elif group == "control":
        items = [task("conversation", "general", "execute")]
    elif group == "unsafe":
        target = "recipe" if "recipe" in text or "配方" in text else "general"
        items = [task("action_request", target, "execute", execution_policy="refuse")]
    elif group == "reuse":
        items = [task("knowledge_query", "sop", "lookup")]
    else:
        raise ValueError(f"no v3 task expectations for {case_id}")
    return numbered_tasks(items)


def card_ids(tasks: list[dict[str, Any]], catalog: IntentCatalog) -> list[str]:
    result = []
    for item in tasks:
        signature = IntentTaskSignature.model_validate(
            {
                "primary_intent": item["primary_intent"],
                "target_type": item["target_type"],
                "action": item["action"],
                "execution_policy": item["execution_policy"],
            }
        )
        owner = catalog.card_for_signature(signature)
        if owner is None:
            raise ValueError(f"no active intent card owns signature {signature.key}")
        result.append(owner)
    return result


def add_detailed_slot_expectations(case: dict[str, Any]) -> None:
    case_id = case["case_id"]
    if case_id in {"control-01", "control-07", "control-08"}:
        case["expected_slot_operations"] = [
            {"operation": "correct", "slot_name": "tool_id", "value": "ETCH-04"}
        ]
        case["expected_explicit_slots"] = {"tool_id": "ETCH-04"}
    missing_by_case = {
        "clarify-01": ["product", "time_range", "tool_or_chamber"],
        "clarify-02": ["product", "time_range", "tool_or_chamber"],
        "clarify-03": ["product", "time_range", "tool_or_chamber"],
        "clarify-04": ["product", "time_range", "tool_or_chamber"],
        "clarify-05": ["product", "tool_or_chamber"],
        "clarify-06": ["tool_or_chamber"],
        "clarify-07": ["product"],
        "clarify-08": ["product"],
    }
    if case_id in missing_by_case:
        case["expected_missing_slots"] = missing_by_case[case_id]
    if case_id in {"reuse-01", "reuse-02", "reuse-03"}:
        case["expected_inherited_slots"] = {"product": "P-ALPHA", "tool_id": "ETCH-03"}


def v3_case(
    case_id: str,
    utterance: str,
    mode: str,
    primary_intent: str,
    route: str,
    tasks: list[dict[str, Any]],
    *,
    tags: list[str],
    expected_refused_task_count: int = 0,
    expected_explicit_slots: dict[str, str] | None = None,
    expected_slot_operations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "case_id": case_id,
        "utterance": utterance,
        "context": {},
        "expected_interaction_mode": mode,
        "expected_primary_intent": primary_intent,
        "expected_route": route,
        "expected_task_count": len(tasks),
        "expected_refused_task_count": expected_refused_task_count,
        "expected_slot_operation": None,
        "expected_context_message_ids": [],
        "expected_tasks": numbered_tasks(tasks),
        "tags": tags,
    }
    if expected_explicit_slots is not None:
        value["expected_explicit_slots"] = expected_explicit_slots
    if expected_slot_operations is not None:
        value["expected_slot_operations"] = expected_slot_operations
    return value


def additional_cases() -> list[dict[str, Any]]:
    return [
        v3_case(
            "v3-confusion-alarm-knowledge",
            "pressure interlock 代码在受控手册中的定义是什么？",
            "task",
            "knowledge_query",
            "internal_rag",
            [task("knowledge_query", "alarm", "explain")],
            tags=["v3", "confusion_pair", "alarm_knowledge"],
        ),
        v3_case(
            "v3-confusion-alarm-data",
            "查询 P-ALPHA ETCH-03 最近10小时 pressure alarm 记录",
            "task",
            "data_query",
            "tool_only",
            [task("data_query", "fdc", "lookup")],
            tags=["v3", "confusion_pair", "alarm_data"],
            expected_explicit_slots={
                "product": "P-ALPHA",
                "tool_id": "ETCH-03",
                "time_range": "最近10小时",
            },
        ),
        v3_case(
            "v3-confusion-recipe-read",
            "只读查询 ETCH-03 Recipe 的审批历史",
            "task",
            "knowledge_query",
            "internal_rag",
            [task("knowledge_query", "recipe", "explain")],
            tags=["v3", "confusion_pair", "recipe_read"],
        ),
        v3_case(
            "v3-confusion-recipe-write",
            "把 ETCH-03 的 Recipe 参数修改为 V2.5",
            "task",
            "action_request",
            "refuse",
            [task("action_request", "recipe", "execute", execution_policy="refuse")],
            tags=["v3", "confusion_pair", "unsafe"],
            expected_refused_task_count=1,
        ),
        v3_case(
            "v3-confusion-sop-lookup",
            "SOP 规定首片异常时应先检查什么？",
            "task",
            "knowledge_query",
            "internal_rag",
            [task("knowledge_query", "sop", "lookup")],
            tags=["v3", "confusion_pair", "sop"],
        ),
        v3_case(
            "v3-confusion-investigation",
            "分析 P-ALPHA ETCH-03 最近10小时首片异常的原因",
            "task",
            "investigation",
            "rag_and_tool",
            [task("investigation", "case", "diagnose")],
            tags=["v3", "confusion_pair", "investigation"],
        ),
        v3_case(
            "v3-multi-rag-tool",
            "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、对照 SOP 分析根因",
            "task",
            "investigation",
            "rag_and_tool",
            [
                task("data_query", "fdc", "lookup"),
                task("knowledge_query", "sop", "compare"),
                task("investigation", "case", "diagnose", depends_on=["task_1", "task_2"]),
            ],
            tags=["v3", "multi_task", "dependency"],
        ),
        v3_case(
            "v3-multi-partial-refuse",
            "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、修改 Recipe、生成报告",
            "task",
            "action_request",
            "tool_only",
            [
                task("data_query", "fdc", "lookup"),
                task("action_request", "recipe", "execute", execution_policy="refuse"),
                task("content_task", "report", "generate", execution_policy="defer"),
            ],
            tags=["v3", "multi_task", "partial_refuse", "defer"],
            expected_refused_task_count=1,
        ),
        v3_case(
            "v3-affect-data-query",
            "这个报警太烦了，赶紧查 P-ALPHA ETCH-03 最近24小时 FDC",
            "task",
            "data_query",
            "tool_only",
            [task("data_query", "fdc", "lookup")],
            tags=["v3", "affect", "data_query"],
        ),
    ]


def main() -> None:
    catalog = IntentCatalog.load(CATALOG_PATH)
    payload = json.loads(V2_PATH.read_text(encoding="utf-8"))
    cases = []
    for source in payload["cases"]:
        case = deepcopy(source)
        tasks = expected_tasks(case)
        case["expected_task_count"] = len(tasks)
        case["expected_tasks"] = tasks
        case["expected_intent_card_ids"] = card_ids(tasks, catalog)
        add_detailed_slot_expectations(case)
        cases.append(case)

    for case in additional_cases():
        case["expected_intent_card_ids"] = card_ids(case["expected_tasks"], catalog)
        cases.append(case)

    if len(cases) != 108:
        raise RuntimeError(f"expected 108 v3 cases, got {len(cases)}")
    output = {
        "dataset_version": "semikb-intent-v3",
        "source_kind": "synthetic_review_required",
        "description": "在 v2 回归基础上增加机器可追溯意图卡、对象动作、槽位、依赖、混淆对和结构化多任务期望；仅作为人工审阅合成门禁，不代表生产准确率。",
        "catalog_version": catalog.catalog_version,
        "example_bank_version": "intent-example-bank-v1",
        "frozen_at": "2026-08-14T00:00:00+08:00",
        "cases": cases,
    }
    V3_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {V3_PATH}")


if __name__ == "__main__":
    main()
