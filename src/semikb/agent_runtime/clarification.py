"""Versioned, task-local clarification state and deterministic transition policy."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from semikb.contracts.models import (
    AgentRoute,
    CancelScope,
    ClarificationFrame,
    ClarificationFrameStatus,
    ClarificationItemType,
    ClarificationKind,
    ClarificationPendingItem,
    ClarificationResolvedItem,
    ClarificationTransitionAudit,
    ClarificationTurnRelation,
    ConversationUnderstanding,
    InteractionMode,
    PrimaryIntent,
    RoutePlan,
    new_id,
)

INTENT_TARGET_KEY = "intent_target"
INTENT_TARGET_VALUES = ["knowledge_query", "data_query", "history_task"]
EXPLICIT_SWITCH_PATTERN = re.compile(
    r"(?:换(?:个|一个)?问题|改问|另一个问题|不问这个了|先不说这个|重新问|我想问别的)",
    re.IGNORECASE,
)
EXPLICIT_CONTINUE_PATTERN = re.compile(
    r"(?:继续(?:刚才|当前|这个|原来)?|补充(?:一下)?|关于刚才)",
    re.IGNORECASE,
)
UNKNOWN_ANSWER_PATTERN = re.compile(
    r"^(?:暂时)?(?:不知道|不清楚|没有更多信息|无法确认|不确定|还是没有更多信息)[。！! ]*$"
)
STANDALONE_TASK_PATTERN = re.compile(
    r"(?:[？?]|为什么|怎么|如何|是什么|查(?:询)?|查看|检索|分析|比较|对比|帮我|请问|"
    r"sop|recipe|fdc|spc|良率|报警|缺陷|wafer|lot)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ClarificationRelationDecision:
    relation: ClarificationTurnRelation
    classifier_source: str
    warning_codes: tuple[str, ...] = ()


def build_clarification_frame(
    *,
    request: str,
    understanding: ConversationUnderstanding,
    route_plan: RoutePlan,
    missing_keys: list[str],
    prompts: dict[str, str],
    existing: ClarificationFrame | None = None,
) -> ClarificationFrame:
    keys = list(dict.fromkeys(missing_keys))[:3]
    kind = ClarificationKind.SLOT_COLLECTION
    if keys == ["history_reference"]:
        kind = ClarificationKind.HISTORY_REFERENCE
    elif not keys:
        kind = ClarificationKind.INTENT_DISAMBIGUATION
        keys = [INTENT_TARGET_KEY]

    pending_items = []
    for key in keys:
        is_choice = key == INTENT_TARGET_KEY
        pending_items.append(
            ClarificationPendingItem(
                key=key,
                item_type=(
                    ClarificationItemType.CHOICE if is_choice else ClarificationItemType.SLOT
                ),
                prompt=(
                    "请明确希望查询受控知识、查看制造数据，还是处理上一轮内容。"
                    if is_choice
                    else prompts[key]
                ),
                allowed_values=INTENT_TARGET_VALUES if is_choice else [],
            )
        )

    task_ids = [item.task_id for item in understanding.task_items]
    no_progress_count = existing.no_progress_count if existing else 0
    signature = clarification_signature(kind, keys, task_ids, no_progress_count)
    base_understanding = (
        existing.base_understanding
        if existing and existing.base_understanding
        else understanding.model_dump(mode="json")
    )
    base_route_plan = (
        existing.base_route_plan
        if existing and existing.base_route_plan
        else route_plan.model_dump(mode="json")
    )
    return ClarificationFrame(
        frame_id=existing.frame_id if existing else new_id("clarify"),
        kind=kind,
        original_request=existing.original_request if existing else request,
        candidate_route=(
            existing.candidate_route
            if existing and existing.candidate_route is not AgentRoute.CLARIFY
            else _candidate_route(route_plan, understanding)
        ),
        task_ids=task_ids or (existing.task_ids if existing else []),
        pending_items=pending_items,
        resolved_items=list(existing.resolved_items) if existing else [],
        round=existing.round if existing else 0,
        no_progress_count=no_progress_count,
        signature=signature,
        status=ClarificationFrameStatus.WAITING,
        last_transition=existing.last_transition if existing else None,
        last_source_message_id=existing.last_source_message_id if existing else None,
        superseded_by_request_id=existing.superseded_by_request_id if existing else None,
        base_understanding=base_understanding,
        base_route_plan=base_route_plan,
    )


def adapt_legacy_frame(
    *,
    state: dict[str, Any],
    prompts: dict[str, str],
) -> ClarificationFrame:
    payload = state.get("clarification_frame")
    if isinstance(payload, dict) and payload:
        return ClarificationFrame.model_validate(payload)
    understanding = ConversationUnderstanding.model_validate(state.get("understanding", {}))
    route_plan = RoutePlan.model_validate(state.get("route_plan", {}))
    return build_clarification_frame(
        request=str(state.get("request") or understanding.standalone_query),
        understanding=understanding,
        route_plan=route_plan,
        missing_keys=[str(item) for item in state.get("missing_required_fields", [])],
        prompts=prompts,
    )


def decide_turn_relation(
    *,
    frame: ClarificationFrame,
    message: str,
    understanding: ConversationUnderstanding,
) -> ClarificationRelationDecision:
    if frame.status in {
        ClarificationFrameStatus.CANCELLED,
        ClarificationFrameStatus.RESOLVED,
        ClarificationFrameStatus.SUPERSEDED,
    }:
        return ClarificationRelationDecision(
            ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST,
            "server_policy",
            ("terminal_frame_cannot_resume",),
        )
    if understanding.cancel_scope in {CancelScope.CLARIFICATION, CancelScope.CURRENT_TASK}:
        return ClarificationRelationDecision(
            ClarificationTurnRelation.CANCEL_CURRENT,
            understanding.classifier_source,
        )
    if EXPLICIT_SWITCH_PATTERN.search(message):
        relation = (
            ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST
            if _looks_like_standalone_task(message, understanding)
            else ClarificationTurnRelation.AMBIGUOUS
        )
        return ClarificationRelationDecision(relation, "l0")

    if EXPLICIT_CONTINUE_PATTERN.search(message):
        warning_codes = (
            ()
            if resolved_pending_keys(frame, understanding)
            else ("explicit_resume_without_answer",)
        )
        return ClarificationRelationDecision(
            ClarificationTurnRelation.CONTINUE_CURRENT,
            "l0",
            warning_codes,
        )
    if UNKNOWN_ANSWER_PATTERN.fullmatch(message.strip()):
        return ClarificationRelationDecision(
            ClarificationTurnRelation.CONTINUE_CURRENT,
            understanding.classifier_source,
        )
    if _is_distinct_new_task(frame, message, understanding):
        return ClarificationRelationDecision(
            ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST,
            understanding.classifier_source,
        )
    resolved = resolved_pending_keys(frame, understanding)
    if resolved:
        return ClarificationRelationDecision(
            ClarificationTurnRelation.CONTINUE_CURRENT,
            understanding.classifier_source,
        )

    proposed = understanding.clarification_relation
    if proposed is ClarificationTurnRelation.CANCEL_CURRENT:
        return ClarificationRelationDecision(
            ClarificationTurnRelation.AMBIGUOUS,
            "server_policy",
            ("ungrounded_cancel_relation",),
        )
    if proposed is ClarificationTurnRelation.SIDE_CONVERSATION and _is_direct_conversation(
        understanding
    ):
        return ClarificationRelationDecision(proposed, understanding.classifier_source)
    if proposed is ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST and _looks_like_standalone_task(
        message, understanding
    ):
        return ClarificationRelationDecision(proposed, understanding.classifier_source)
    if proposed is ClarificationTurnRelation.CONTINUE_CURRENT:
        return ClarificationRelationDecision(proposed, understanding.classifier_source)
    if _is_direct_conversation(understanding):
        return ClarificationRelationDecision(
            ClarificationTurnRelation.SIDE_CONVERSATION,
            understanding.classifier_source,
        )
    if _looks_like_standalone_task(message, understanding):
        return ClarificationRelationDecision(
            ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST,
            understanding.classifier_source,
        )
    return ClarificationRelationDecision(
        ClarificationTurnRelation.CONTINUE_CURRENT,
        "server_policy",
        ("relation_defaulted_to_current_frame",),
    )


def merge_continuation_understanding(
    frame: ClarificationFrame,
    response: str,
    current: ConversationUnderstanding,
) -> ConversationUnderstanding:
    base = ConversationUnderstanding.model_validate(frame.base_understanding)
    explicit = {**base.explicit_slots, **current.explicit_slots}
    inherited = {**base.inherited_slots, **current.inherited_slots}
    operations = list(base.slot_operations)
    for operation in current.slot_operations:
        if operation not in operations:
            operations.append(operation)

    use_current_task = frame.kind is ClarificationKind.INTENT_DISAMBIGUATION and bool(
        current.task_items
    )
    return base.model_copy(
        update={
            "classifier_source": current.classifier_source,
            "interaction_mode": InteractionMode.CLARIFICATION_ANSWER,
            "primary_intent": current.primary_intent if use_current_task else base.primary_intent,
            "task_items": current.task_items if use_current_task else base.task_items,
            "slot_operations": operations[:12],
            "explicit_slots": explicit,
            "inherited_slots": inherited,
            "missing_slots": [],
            "context_message_ids": (
                current.context_message_ids or base.context_message_ids
            ),
            "standalone_query": f"{frame.original_request}\n补充条件：{response}"[:8000],
            "cancel_scope": None,
            "clarification_relation": ClarificationTurnRelation.CONTINUE_CURRENT,
            "suggested_route": (
                current.suggested_route if use_current_task else base.suggested_route
            ),
            "confidence": max(base.confidence, current.confidence),
        }
    )


def update_frame_after_answer(
    *,
    frame: ClarificationFrame,
    understanding: ConversationUnderstanding,
    pending_after: list[str],
    source_message_id: str | None,
    classifier_source: str,
) -> tuple[ClarificationFrame, ClarificationTransitionAudit]:
    pending_before = frame.pending_keys
    resolved = [key for key in pending_before if key not in pending_after]
    resolved_items = list(frame.resolved_items)
    values = {**understanding.inherited_slots, **understanding.explicit_slots}
    for key in resolved:
        value = _resolved_value(key, values, understanding)
        resolved_items = [item for item in resolved_items if item.key != key]
        resolved_items.append(
            ClarificationResolvedItem(
                key=key,
                value=value,
                source_message_id=source_message_id,
            )
        )
    made_progress = bool(resolved)
    no_progress_count = 0 if made_progress else min(frame.no_progress_count + 1, 3)
    pending_items = [item for item in frame.pending_items if item.key in pending_after]
    next_status = (
        ClarificationFrameStatus.RESOLVED
        if not pending_after
        else ClarificationFrameStatus.WAITING
    )
    updated = frame.model_copy(
        update={
            "pending_items": pending_items,
            "resolved_items": resolved_items,
            "no_progress_count": no_progress_count,
            "signature": clarification_signature(
                frame.kind,
                pending_after,
                frame.task_ids,
                no_progress_count,
            ),
            "status": next_status,
            "last_transition": ClarificationTurnRelation.CONTINUE_CURRENT,
            "last_source_message_id": source_message_id,
            "base_understanding": understanding.model_dump(mode="json"),
        }
    )
    return updated, ClarificationTransitionAudit(
        frame_id=frame.frame_id,
        relation=ClarificationTurnRelation.CONTINUE_CURRENT,
        classifier_source=classifier_source,
        previous_status=frame.status,
        next_status=next_status,
        pending_before=pending_before,
        resolved_by_answer=resolved,
        pending_after=pending_after,
        made_progress=made_progress,
        warning_codes=[] if made_progress else ["clarification_no_progress"],
    )


def terminal_transition(
    *,
    frame: ClarificationFrame,
    relation: ClarificationTurnRelation,
    classifier_source: str,
    source_message_id: str | None,
    request_id: str | None = None,
    warning_codes: tuple[str, ...] = (),
) -> tuple[ClarificationFrame, ClarificationTransitionAudit]:
    status = {
        ClarificationTurnRelation.CANCEL_CURRENT: ClarificationFrameStatus.CANCELLED,
        ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST: ClarificationFrameStatus.SUPERSEDED,
        ClarificationTurnRelation.SIDE_CONVERSATION: ClarificationFrameStatus.PAUSED,
        ClarificationTurnRelation.AMBIGUOUS: ClarificationFrameStatus.PAUSED,
    }.get(relation, frame.status)
    pending_after = frame.pending_keys if status is ClarificationFrameStatus.PAUSED else []
    updated = frame.model_copy(
        update={
            "status": status,
            "last_transition": relation,
            "last_source_message_id": source_message_id,
            "superseded_by_request_id": (
                request_id
                if relation is ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST
                else frame.superseded_by_request_id
            ),
        }
    )
    return updated, ClarificationTransitionAudit(
        frame_id=frame.frame_id,
        relation=relation,
        classifier_source=classifier_source,
        previous_status=frame.status,
        next_status=status,
        pending_before=frame.pending_keys,
        resolved_by_answer=[],
        pending_after=pending_after,
        made_progress=relation in {
            ClarificationTurnRelation.CANCEL_CURRENT,
            ClarificationTurnRelation.REPLACE_WITH_NEW_REQUEST,
        },
        warning_codes=list(warning_codes),
    )


def clarification_signature(
    kind: ClarificationKind,
    pending_keys: list[str],
    task_ids: list[str],
    no_progress_count: int,
) -> str:
    payload = json.dumps(
        {
            "kind": kind.value,
            "pending": sorted(pending_keys),
            "tasks": sorted(task_ids),
            "no_progress": no_progress_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolved_pending_keys(
    frame: ClarificationFrame,
    understanding: ConversationUnderstanding,
) -> list[str]:
    values = {**understanding.inherited_slots, **understanding.explicit_slots}
    resolved = []
    for key in frame.pending_keys:
        if key == "tool_or_chamber" and (values.get("tool_id") or values.get("chamber")):
            resolved.append(key)
        elif key == "history_reference" and understanding.context_message_ids:
            resolved.append(key)
        elif key == INTENT_TARGET_KEY and (
            understanding.primary_intent
            in {
                PrimaryIntent.KNOWLEDGE_QUERY,
                PrimaryIntent.DATA_QUERY,
                PrimaryIntent.CONTENT_TASK,
            }
            or understanding.suggested_route
            in {
                AgentRoute.INTERNAL_RAG,
                AgentRoute.TOOL_ONLY,
                AgentRoute.RAG_AND_TOOL,
                AgentRoute.RAG_AND_WEB,
                AgentRoute.CHAT_DIRECT,
            }
        ):
            resolved.append(key)
        elif values.get(key):
            resolved.append(key)
    return resolved


def _candidate_route(
    route_plan: RoutePlan,
    understanding: ConversationUnderstanding,
) -> AgentRoute:
    routes = [item.route for item in route_plan.task_decisions if item.route is not None]
    if routes:
        return routes[0]
    if understanding.suggested_route is not AgentRoute.CLARIFY:
        return understanding.suggested_route
    return AgentRoute.INTERNAL_RAG


def _is_direct_conversation(understanding: ConversationUnderstanding) -> bool:
    return (
        understanding.primary_intent is PrimaryIntent.CONVERSATION
        and understanding.suggested_route in {AgentRoute.CHAT_DIRECT, AgentRoute.HISTORY_DIRECT}
    )


def _looks_like_standalone_task(
    message: str,
    understanding: ConversationUnderstanding,
) -> bool:
    if not STANDALONE_TASK_PATTERN.search(message):
        return False
    return bool(understanding.task_items) and not (
        understanding.primary_intent is PrimaryIntent.CONVERSATION
        and understanding.suggested_route in {AgentRoute.CHAT_DIRECT, AgentRoute.HISTORY_DIRECT}
    )


def _is_distinct_new_task(
    frame: ClarificationFrame,
    message: str,
    understanding: ConversationUnderstanding,
) -> bool:
    if not _looks_like_standalone_task(message, understanding):
        return False
    base = ConversationUnderstanding.model_validate(frame.base_understanding)
    if understanding.primary_intent is not base.primary_intent:
        return True
    current_signatures = {(item.target_type, item.action) for item in understanding.task_items}
    base_signatures = {(item.target_type, item.action) for item in base.task_items}
    return bool(current_signatures and base_signatures and current_signatures.isdisjoint(base_signatures))


def _resolved_value(
    key: str,
    values: dict[str, str],
    understanding: ConversationUnderstanding,
) -> str:
    if key == "tool_or_chamber":
        return values.get("tool_id") or values.get("chamber") or "provided"
    if key == "history_reference":
        return understanding.context_message_ids[0] if understanding.context_message_ids else "provided"
    if key == INTENT_TARGET_KEY:
        return understanding.primary_intent.value
    return values.get(key, "provided")
