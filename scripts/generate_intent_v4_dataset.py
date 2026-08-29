"""Generate the frozen T9-4.11 semantic minimal-pair evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentTaskSignature

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v5.json"
OUTPUT_PATH = ROOT / "data" / "intent_sets" / "semikb_intent_v4.json"


def task(
    primary_intent: str,
    target_type: str,
    action: str,
    *,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "primary_intent": primary_intent,
        "target_type": target_type,
        "action": action,
        "execution_policy": "execute",
        "depends_on": depends_on or [],
    }


def case(
    case_id: str,
    utterance: str,
    primary_intent: str,
    route: str,
    tasks: list[dict[str, Any]],
    catalog: IntentCatalog,
    *,
    missing_slots: list[str] | None = None,
    explicit_slots: dict[str, str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    numbered = [
        {"task_id": f"task_{index}", **item}
        for index, item in enumerate(tasks, start=1)
    ]
    cards = []
    for item in numbered:
        owner = catalog.card_for_signature(
            IntentTaskSignature.model_validate(
                {
                    "primary_intent": item["primary_intent"],
                    "target_type": item["target_type"],
                    "action": item["action"],
                    "execution_policy": item["execution_policy"],
                }
            )
        )
        if owner is None:
            raise ValueError(f"No intent card owns {item}")
        cards.append(owner)
    result: dict[str, Any] = {
        "case_id": case_id,
        "utterance": utterance,
        "context": {},
        "expected_interaction_mode": "task",
        "expected_primary_intent": primary_intent,
        "expected_route": route,
        "expected_task_count": len(numbered),
        "expected_refused_task_count": 0,
        "expected_slot_operation": None,
        "expected_context_message_ids": [],
        "expected_tasks": numbered,
        "expected_intent_card_ids": cards,
        "tags": ["t9-4.11", "unseen", *(tags or [])],
    }
    if missing_slots is not None:
        result["expected_missing_slots"] = missing_slots
    if explicit_slots is not None:
        result["expected_explicit_slots"] = explicit_slots
    return result


def build_cases(catalog: IntentCatalog) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    knowledge_subjects = (
        "半导体检测流程",
        "晶圆缺陷分类",
        "SPC 控制",
        "FDC 监控",
        "晶圆边缘检测",
        "良率改善",
    )
    knowledge_prompts = (
        "通常包括哪些环节？",
        "基本原理是什么？",
        "一般有哪些影响因素？",
        "主要作用是什么？",
        "常见分析思路是什么？",
    )
    for subject_index, subject in enumerate(knowledge_subjects, start=1):
        for prompt_index, prompt in enumerate(knowledge_prompts, start=1):
            cases.append(
                case(
                    f"v4-knowledge-{subject_index:02d}-{prompt_index:02d}",
                    f"{subject}{prompt}",
                    "knowledge_query",
                    "internal_rag",
                    [task("knowledge_query", "general", "lookup")],
                    catalog,
                    tags=["minimal_pair", "public_general"],
                )
            )

    bounded_times = ("最近6小时", "最近12小时", "最近24小时", "过去2天", "过去3天", "过去1周")
    aggregate_prompts = (
        "有哪些产品良率低？",
        "各产品良率排名怎么样？",
        "哪些产品的良率波动最大？",
        "按产品统计良率表现。",
    )
    for time_index, time_range in enumerate(bounded_times, start=1):
        for prompt_index, prompt in enumerate(aggregate_prompts, start=1):
            cases.append(
                case(
                    f"v4-aggregate-{time_index:02d}-{prompt_index:02d}",
                    f"{time_range}{prompt}",
                    "data_query",
                    "tool_only",
                    [task("data_query", "lot", "lookup")],
                    catalog,
                    explicit_slots={"time_range": time_range},
                    tags=["minimal_pair", "aggregate", "group_by_product"],
                )
            )

    unbounded_prompts = (
        "最近有哪些产品良率低？",
        "近期哪些设备 FDC 报警最多？",
        "最近哪些 Lot 触发了 SPC 越界？",
        "近期各腔体的压力趋势怎么样？",
    )
    unbounded_prefixes = ("请告诉我，", "帮我看看，", "我想知道，")
    for repeat, prefix in enumerate(unbounded_prefixes, start=1):
        for prompt_index, prompt in enumerate(unbounded_prompts, start=1):
            target = "fdc" if "FDC" in prompt or "压力" in prompt else "spc" if "SPC" in prompt else "lot"
            cases.append(
                case(
                    f"v4-unbounded-{repeat:02d}-{prompt_index:02d}",
                    f"{prefix}{prompt}",
                    "data_query",
                    "clarify",
                    [task("data_query", target, "lookup")],
                    catalog,
                    missing_slots=["time_range"],
                    tags=["minimal_pair", "missing_time"],
                )
            )

    entity_templates = (
        ("查询 P-ALPHA ETCH-03 {time} FDC 报警记录", "fdc"),
        ("查看 P-ALPHA ETCH-03 {time} SPC 趋势", "spc"),
        ("列出 P-ALPHA ETCH-03 {time} Lot 良率", "lot"),
    )
    for time_index, time_range in enumerate(bounded_times, start=1):
        for template_index, (template, target) in enumerate(entity_templates, start=1):
            cases.append(
                case(
                    f"v4-entity-{time_index:02d}-{template_index:02d}",
                    template.format(time=time_range),
                    "data_query",
                    "tool_only",
                    [task("data_query", target, "lookup")],
                    catalog,
                    explicit_slots={
                        "product": "P-ALPHA",
                        "tool_id": "ETCH-03",
                        "time_range": time_range,
                    },
                    tags=["minimal_pair", "entity_data"],
                )
            )

    investigation_prompts = (
        ("分析 P-ALPHA ETCH-03 {time} 良率下降的原因", "lot"),
        ("P-ALPHA ETCH-03 {time} FDC 报警为什么增加？", "fdc"),
        ("诊断 P-ALPHA ETCH-03 {time} 晶圆缺陷异常", "lot"),
        ("排查 P-ALPHA ETCH-03 {time} SPC 波动的可能根因", "spc"),
    )
    for time_index, time_range in enumerate(bounded_times, start=1):
        for prompt_index, (template, target) in enumerate(investigation_prompts, start=1):
            cases.append(
                case(
                    f"v4-investigation-{time_index:02d}-{prompt_index:02d}",
                    template.format(time=time_range),
                    "investigation",
                    "rag_and_tool",
                    [
                        task("data_query", target, "lookup"),
                        task("investigation", "case", "diagnose", depends_on=["task_1"]),
                    ],
                    catalog,
                    explicit_slots={
                        "product": "P-ALPHA",
                        "tool_id": "ETCH-03",
                        "time_range": time_range,
                    },
                    tags=["minimal_pair", "causal", "concrete_scope"],
                )
            )

    internal_prompts = (
        ("当前 ETCH-03 SOP 对首片异常怎么要求？", "sop", "lookup"),
        ("查询现行 ETCH-03 SOP 的清腔前置条件。", "sop", "lookup"),
        ("当前批准的 ETCH-03 Recipe 版本规则是什么？", "recipe", "explain"),
        ("解释本厂 ETCH-03 Recipe 的审批要求。", "recipe", "explain"),
    )
    internal_prefixes = ("请问，", "帮我确认，", "我需要了解：")
    for repeat, prefix in enumerate(internal_prefixes, start=1):
        for prompt_index, (prompt, target, action) in enumerate(internal_prompts, start=1):
            cases.append(
                case(
                    f"v4-internal-{repeat:02d}-{prompt_index:02d}",
                    f"{prefix}{prompt}",
                    "knowledge_query",
                    "internal_rag",
                    [task("knowledge_query", target, action)],
                    catalog,
                    tags=["minimal_pair", "internal_controlled"],
                )
            )

    external_subjects = (
        "公开资料中的晶圆缺陷检测方法",
        "网上公开的先进封装检测趋势",
        "互联网公开资料里的良率分析方法",
    )
    external_prompts = ("有哪些？", "怎么分类？", "主要原理是什么？", "有哪些典型应用？")
    for subject_index, subject in enumerate(external_subjects, start=1):
        for prompt_index, prompt in enumerate(external_prompts, start=1):
            cases.append(
                case(
                    f"v4-external-{subject_index:02d}-{prompt_index:02d}",
                    f"{subject}{prompt}",
                    "knowledge_query",
                    "rag_and_web",
                    [task("knowledge_query", "general", "lookup")],
                    catalog,
                    tags=["minimal_pair", "explicit_web"],
                )
            )
    return cases


def main() -> None:
    catalog = IntentCatalog.load(CATALOG_PATH)
    cases = build_cases(catalog)
    if len(cases) != 132:
        raise ValueError(f"Expected 132 cases, got {len(cases)}")
    payload = {
        "dataset_version": "semikb-intent-v4",
        "source_kind": "synthetic_review_required",
        "description": (
            "T9-4.11 未见最小差异评测集，验证一般知识、动态制造数据和具体异常调查边界。"
            "数据只用于离线评测，不进入意图 Prompt 或生产规则。"
        ),
        "catalog_version": catalog.catalog_version,
        "example_bank_version": "not_used",
        "frozen_at": "2026-08-28T00:00:00+08:00",
        "cases": cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
