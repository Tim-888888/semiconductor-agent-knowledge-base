"""Server-side semantic grounding for structured intent output.

The language model proposes a plan; this module checks whether the plan is
supported by the turn as a whole. Individual words are weak signals only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from semikb.contracts.models import (
    ExpectedOutput,
    GroupingDimension,
    IntentTarget,
    KnowledgeScope,
    PrimaryIntent,
    SemanticFrame,
    SemanticTemporalScope,
    TaskShape,
)

_RELATIVE_TIME = re.compile(
    r"(?:最近|过去|近)\s*\d*\s*(?:小时|天|周|月)?|"
    r"(?:今天|昨日|昨天|本周|上周|本月|当前|现在|实时|近期)",
    re.IGNORECASE,
)
_EXPLICIT_TIME = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d+\s*(?:分钟|小时|天|周|月)",
    re.IGNORECASE,
)
_CONCRETE_ENTITY = re.compile(
    r"\b(?:P-[A-Z0-9][A-Z0-9-]*|(?:ETCH|CVD|PVD|CMP|PHOTO|LITHO|IMP|DIFF)-\d+[A-Z]?|"
    r"LOT[-_ ]?[A-Z0-9-]+|CASE[-_ ]?[A-Z0-9-]+|V\d+(?:\.\d+)+)\b|"
    r"Chamber\s*[A-Z0-9]+",
    re.IGNORECASE,
)

_CONCEPT_TERMS = (
    "一般",
    "通常",
    "常见",
    "原理",
    "流程",
    "定义",
    "含义",
    "作用",
    "包括",
    "包含",
    "规范",
    "规则",
    "是什么",
)
_DATA_OBJECT_TERMS = (
    "良率",
    "yield",
    "fdc",
    "spc",
    "lot",
    "wafer",
    "晶圆",
    "报警",
    "alarm",
    "参数",
    "记录",
    "明细",
    "制造数据",
)
_DATA_ACTION_TERMS = ("查询", "查一下", "查看", "拉取", "统计", "列出", "显示", "筛选")
_AGGREGATE_TERMS = (
    "哪些",
    "包括什么",
    "包含什么",
    "由什么组成",
    "排名",
    "排行",
    "最低",
    "最高",
    "各个",
    "各产品",
    "各设备",
    "按产品",
    "按设备",
)
_TREND_TERMS = ("趋势", "变化", "走势", "波动")
_EVENT_TERMS = ("记录", "明细", "事件", "报警", "触发", "越界")
_CAUSAL_TERMS = ("为什么", "原因", "根因", "诊断", "排查", "关联", "导致")
_INTERNAL_TERMS = (
    "现行",
    "生效",
    "当前版本",
    "批准",
    "本厂",
    "我们的",
    "内部",
    "当前配置",
)


@dataclass(frozen=True, slots=True)
class GroundedSemantics:
    primary_intent: PrimaryIntent
    semantic_frame: SemanticFrame
    task_shape: TaskShape
    group_by: tuple[GroupingDimension, ...]
    strong_override: bool
    reason_codes: tuple[str, ...]


def derive_grounded_semantics(
    request: str,
    proposed_intent: PrimaryIntent,
    proposed_frame: SemanticFrame,
    targets: list[IntentTarget],
) -> GroundedSemantics:
    """Combine independent signals and veto only high-confidence contradictions."""

    lowered = request.casefold()
    has_relative_time = bool(_RELATIVE_TIME.search(request))
    has_explicit_time = bool(_EXPLICIT_TIME.search(request))
    has_time = has_relative_time or has_explicit_time
    has_entity = bool(_CONCRETE_ENTITY.search(request))
    has_concept = any(term in lowered for term in _CONCEPT_TERMS)
    has_data_object = any(term in lowered for term in _DATA_OBJECT_TERMS)
    has_data_action = any(term in lowered for term in _DATA_ACTION_TERMS)
    has_aggregate = any(term in lowered for term in _AGGREGATE_TERMS)
    has_trend = any(term in lowered for term in _TREND_TERMS)
    has_event = any(term in lowered for term in _EVENT_TERMS)
    has_causal = any(term in lowered for term in _CAUSAL_TERMS)

    group_by = _grouping_dimensions(lowered, has_aggregate)
    dynamic_score = 0
    if has_time:
        dynamic_score += 2
    if has_data_object:
        dynamic_score += 1
    if has_data_action:
        dynamic_score += 2
    if has_aggregate and has_data_object:
        dynamic_score += 1
    if has_trend or has_event:
        dynamic_score += 1

    concept_score = 0
    if has_concept:
        concept_score += 2
    if not has_time and not has_entity:
        concept_score += 1
    if any(term in lowered for term in ("一般", "通常", "常见", "原理", "流程")):
        concept_score += 1

    investigation_score = 0
    if has_causal:
        investigation_score += 3
    if has_time or has_entity:
        investigation_score += 2
    if has_data_object:
        investigation_score += 1

    resolved = proposed_intent
    reasons: list[str] = []
    strong_override = False
    protected_intent = proposed_intent in {
        PrimaryIntent.ACTION_REQUEST,
        PrimaryIntent.CONTENT_TASK,
        PrimaryIntent.CONVERSATION,
    }
    if protected_intent:
        reasons.append("preserved_non_retrieval_intent")
    elif investigation_score >= 5:
        resolved = PrimaryIntent.INVESTIGATION
        strong_override = resolved is not proposed_intent
        reasons.append("grounded_dynamic_causal_request")
    elif has_data_object and dynamic_score >= 4:
        resolved = PrimaryIntent.DATA_QUERY
        strong_override = resolved is not proposed_intent
        reasons.append("grounded_dynamic_data_request")
    elif concept_score >= 3 and dynamic_score < 4:
        resolved = PrimaryIntent.KNOWLEDGE_QUERY
        strong_override = resolved is not proposed_intent
        reasons.append("grounded_general_knowledge_request")

    temporal_scope = proposed_frame.temporal_scope
    if has_explicit_time:
        temporal_scope = SemanticTemporalScope.EXPLICIT
    elif has_relative_time:
        temporal_scope = SemanticTemporalScope.RELATIVE
    elif resolved is PrimaryIntent.KNOWLEDGE_QUERY and has_concept:
        temporal_scope = SemanticTemporalScope.TIMELESS

    expected_output = _expected_output(
        resolved,
        has_aggregate=has_aggregate,
        has_trend=has_trend,
        has_event=has_event,
        has_causal=has_causal,
        fallback=proposed_frame.expected_output,
    )
    knowledge_scope = _knowledge_scope(
        lowered,
        resolved,
        targets,
        proposed_frame.knowledge_scope,
    )
    task_shape = _task_shape(
        resolved,
        has_aggregate=has_aggregate,
        has_trend=has_trend,
        has_event=has_event,
    )
    return GroundedSemantics(
        primary_intent=resolved,
        semantic_frame=SemanticFrame(
            temporal_scope=temporal_scope,
            expected_output=expected_output,
            knowledge_scope=knowledge_scope,
        ),
        task_shape=task_shape,
        group_by=group_by,
        strong_override=strong_override,
        reason_codes=tuple(reasons),
    )


def _grouping_dimensions(
    request: str,
    has_aggregate: bool,
) -> tuple[GroupingDimension, ...]:
    if not has_aggregate:
        return ()
    dimensions: list[GroupingDimension] = []
    mappings = (
        (GroupingDimension.PRODUCT, ("产品", "product")),
        (GroupingDimension.TOOL, ("设备", "机台", "tool")),
        (GroupingDimension.CHAMBER, ("腔体", "chamber")),
        (GroupingDimension.LOT, ("lot", "批次")),
        (GroupingDimension.WAFER, ("wafer", "晶圆")),
        (GroupingDimension.ALARM, ("报警", "alarm")),
    )
    for dimension, terms in mappings:
        if any(term in request for term in terms):
            dimensions.append(dimension)
    return tuple(dimensions[:3])


def _expected_output(
    intent: PrimaryIntent,
    *,
    has_aggregate: bool,
    has_trend: bool,
    has_event: bool,
    has_causal: bool,
    fallback: ExpectedOutput,
) -> ExpectedOutput:
    if intent is PrimaryIntent.KNOWLEDGE_QUERY:
        return ExpectedOutput.ENUMERATION if has_aggregate else ExpectedOutput.EXPLANATION
    if intent is PrimaryIntent.INVESTIGATION or has_causal and intent is not PrimaryIntent.KNOWLEDGE_QUERY:
        return ExpectedOutput.DIAGNOSIS
    if has_trend:
        return ExpectedOutput.TREND
    if has_aggregate:
        return ExpectedOutput.RANKING
    if has_event and intent is PrimaryIntent.DATA_QUERY:
        return ExpectedOutput.RECORDS
    return fallback


def _knowledge_scope(
    request: str,
    intent: PrimaryIntent,
    targets: list[IntentTarget],
    fallback: KnowledgeScope,
) -> KnowledgeScope:
    if intent is PrimaryIntent.INVESTIGATION:
        return KnowledgeScope.MIXED
    if intent is not PrimaryIntent.KNOWLEDGE_QUERY:
        return KnowledgeScope.NOT_APPLICABLE
    internal_target = any(target in {IntentTarget.SOP, IntentTarget.RECIPE} for target in targets)
    if internal_target or any(term in request for term in _INTERNAL_TERMS):
        return KnowledgeScope.INTERNAL_CONTROLLED
    if fallback is KnowledgeScope.MIXED:
        return fallback
    return KnowledgeScope.PUBLIC_GENERAL


def _task_shape(
    intent: PrimaryIntent,
    *,
    has_aggregate: bool,
    has_trend: bool,
    has_event: bool,
) -> TaskShape:
    if intent is PrimaryIntent.KNOWLEDGE_QUERY:
        return TaskShape.CONCEPT_EXPLANATION
    if intent is PrimaryIntent.INVESTIGATION:
        return TaskShape.CAUSAL_INVESTIGATION
    if intent is PrimaryIntent.DATA_QUERY:
        if has_aggregate:
            return TaskShape.AGGREGATE_RANKING
        if has_trend:
            return TaskShape.TREND_ANALYSIS
        if has_event:
            return TaskShape.EVENT_LIST
        return TaskShape.ENTITY_LOOKUP
    if intent in {PrimaryIntent.CONVERSATION, PrimaryIntent.CONTENT_TASK}:
        return TaskShape.DIRECT
    return TaskShape.CONTROL
