"""Controlled conversation understanding with an L0 fast path and one structured LLM call."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from semikb.agent_runtime.intent_catalog import IntentCatalog, load_intent_catalog
from semikb.agent_runtime.intent_experiments import (
    IntentExperimentProfile,
    IntentFewShotSelection,
)
from semikb.agent_runtime.llm_gateway import LLMProviderError, OpenAICompatibleLLMGateway
from semikb.config import Settings
from semikb.contracts.models import (
    AffectSignals,
    AgentRoute,
    CancelScope,
    ConversationUnderstanding,
    IntentTarget,
    IntentTaskAction,
    IntentTaskItem,
    InteractionMode,
    PrimaryIntent,
    SlotOperation,
    SlotOperationKind,
    TaskExecutionDecision,
)

SLOT_NAMES = {
    "product",
    "process_layer",
    "tool_id",
    "chamber",
    "recipe_id",
    "recipe_version",
    "time_range",
    "lot_id",
    "case_id",
}

ENTITY_PATTERNS = {
    "product": re.compile(r"\bP-[A-Z0-9][A-Z0-9-]*\b", re.IGNORECASE),
    "tool_id": re.compile(
        r"\b(?:ETCH|CVD|PVD|CMP|PHOTO|LITHO|IMP|DIFF)-\d+[A-Z]?\b",
        re.IGNORECASE,
    ),
    "recipe_version": re.compile(r"\bV\d+(?:\.\d+)+\b", re.IGNORECASE),
    "lot_id": re.compile(r"\bLOT[-_ ]?[A-Z0-9-]+\b", re.IGNORECASE),
    "case_id": re.compile(r"\bCASE[-_ ][A-Z0-9-]+\b", re.IGNORECASE),
}
CHAMBER_PATTERN = re.compile(r"(?:CHAMBER|腔体)\s*[-:]?\s*([A-Z0-9]+)", re.IGNORECASE)
TIME_PATTERN = re.compile(
    r"(?:最近|过去)?\s*\d+\s*(?:小时|天|周)|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*(?:至|到|~)\s*\d{4}[-/]\d{1,2}[-/]\d{1,2})?",
    re.IGNORECASE,
)
CORRECTION_PATTERN = re.compile(
    r"不是\s*([A-Z]+-\d+[A-Z]?)\s*[，,、]?\s*(?:是|改成|换成)\s*([A-Z]+-\d+[A-Z]?)",
    re.IGNORECASE,
)
HISTORY_RECALL_PATTERN = re.compile(
    r"^\s*(?:请)?(?:告诉我|帮我回忆一下|复述一下)?\s*(?:我)?"
    r"(?:刚才|刚刚|刚刚在|刚在|方才|上一轮|上一个|前面|之前)"
    r".{0,10}(?:问|说|输入|提到).{0,8}(?:什么|啥|哪一句|哪句话|哪个问题)"
    r"(?:问题|内容|话)?"
    r"(?:了|呢|吗)?[？?。！!\s]*$"
)
PREVIOUS_ANSWER_PATTERN = re.compile(
    r"(?:把|将)?(?:刚才|上一轮|上面|之前).{0,5}(?:回答|答案).{0,8}(?:简单|简化|总结|翻译|改写)"
)
PREVIOUS_ANSWER_PREFIX_ACTION_PATTERN = re.compile(
    r"(?:总结|翻译|简化|改写).{0,8}(?:刚才|上一轮|上面|之前).{0,5}(?:回答|答案)"
)
CANCEL_PATTERN = re.compile(r"^(?:别查了|取消(?:这次|本次)?(?:查询|任务)?|放弃本轮追问|停止生成)[。！! ]*$")
GREETING_PATTERN = re.compile(r"^(?:你好|您好|嗨|hello|hi|谢谢|感谢|明白了|收到)[。！!,.， ]*$", re.IGNORECASE)
FEEDBACK_PATTERN = re.compile(r"(?:回答|答复).*(?:太复杂|太长|不清楚|太慢|看不懂)|(?:太复杂|太长|看不懂)了")
OOD_PATTERN = re.compile(r"(?:写(?:一首)?诗|订机票|股票|荐股|彩票|医学处方|法律诉讼|玩游戏|讲个笑话)")
REFERENCE_TERMS = ("它", "这个", "该", "刚才", "上一轮", "继续", "基于之前", "上面")
LATEST_TERMS = ("最新", "当前版本", "现行", "实时", "现在", "today", "latest")
EXTERNAL_TERMS = ("外部", "互联网", "网上", "web", "公开资料", "行业最新", "官方公告")
MUTATION_TERMS = ("修改", "下发", "写入", "删除", "停机", "执行变更", "调整recipe", "调整 recipe")
ALARM_TERMS = ("报警", "告警", "alarm", "interlock")


class _RawAffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: str = Field(pattern="^(neutral|positive|negative)$")
    urgency: str = Field(pattern="^(normal|urgent)$")
    complaint_signal: bool


class _RawTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^task_[1-3]$")
    primary_intent: PrimaryIntent
    target_type: IntentTarget
    action: IntentTaskAction
    depends_on: list[str] = Field(max_length=2)
    execution_policy: TaskExecutionDecision


class _RawSlotValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_name: str
    value: str


class _RawSlotOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: SlotOperationKind
    slot_name: str
    value: str | None


class _RawUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interaction_mode: InteractionMode
    primary_intent: PrimaryIntent
    task_items: list[_RawTask] = Field(max_length=3)
    affect: _RawAffect
    slot_operations: list[_RawSlotOperation] = Field(max_length=12)
    explicit_slots: list[_RawSlotValue] = Field(max_length=12)
    inherited_slot_names: list[str] = Field(max_length=12)
    missing_slots: list[str] = Field(max_length=3)
    context_message_ids: list[str] = Field(max_length=8)
    standalone_query: str = Field(max_length=8000)
    cancel_scope: CancelScope | None
    suggested_route: AgentRoute
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    understanding: ConversationUnderstanding
    metadata: dict[str, Any]


class ConversationUnderstandingService:
    """Interpret a turn without allowing model output to execute tools directly."""

    def __init__(
        self,
        settings: Settings,
        llm: OpenAICompatibleLLMGateway,
        *,
        intent_catalog: IntentCatalog | None = None,
        experiment_profile: IntentExperimentProfile | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.intent_catalog = intent_catalog or load_intent_catalog(settings.intent_catalog_path)
        self.experiment_profile = (
            experiment_profile or IntentExperimentProfile.production_baseline()
        )
        if self.experiment_profile.example_bank is not None:
            self.experiment_profile.example_bank.validate_against_catalog(self.intent_catalog)

    async def understand(
        self,
        request: str,
        context: dict[str, Any],
        *,
        clarification_pending: bool = False,
    ) -> UnderstandingResult:
        started = time.perf_counter()
        empty_selection = IntentFewShotSelection()
        base_metadata = {
            **self._intent_audit_metadata(prompt_tokens=0, cards_in_prompt=0),
            **self.experiment_profile.audit_metadata(empty_selection),
        }
        l0 = self._l0(request, context)
        if l0 is not None:
            return UnderstandingResult(
                l0,
                self._finish_metadata(
                    {
                        **base_metadata,
                        "understanding_source": "l0",
                        "understanding_calls": 0,
                    },
                    started,
                ),
            )

        if self.settings.demo_mode:
            fallback = self._heuristic(request, context, clarification_pending=clarification_pending)
            return UnderstandingResult(
                fallback,
                self._finish_metadata(
                    {
                        **base_metadata,
                        "understanding_source": "deterministic_fallback",
                        "understanding_calls": 0,
                    },
                    started,
                ),
            )

        selection = await self.experiment_profile.select_examples(request)
        messages = self._prompt(request, context, clarification_pending, selection)
        estimated_tokens = self._estimate_prompt_tokens(messages)
        cards_in_prompt = (
            len(self.intent_catalog.active_cards)
            if self.experiment_profile.include_catalog
            else 0
        )
        metadata: dict[str, Any] = {
            **self._intent_audit_metadata(
                prompt_tokens=estimated_tokens,
                cards_in_prompt=cards_in_prompt,
            ),
            **self.experiment_profile.audit_metadata(selection),
            "understanding_calls": 0,
            "intent_prompt_tokens_source": "deterministic_estimate",
            "intent_completion_tokens": 0,
            "intent_completion_tokens_source": "not_available",
            "intent_total_tokens": estimated_tokens,
            "intent_usage_source": "deterministic_estimate",
        }
        raw: _RawUnderstanding | None = None
        invalid_content = ""
        observed_prompt_tokens = 0
        observed_completion_tokens = 0
        usage_sources: set[str] = set()
        try:
            completion = await self.llm.complete(
                messages,
                response_schema=_RawUnderstanding.model_json_schema(),
                schema_name="semikb_conversation_understanding_v1",
                temperature=0,
                max_output_tokens=1000,
            )
            metadata["understanding_calls"] = 1
            input_tokens = self._input_tokens(completion.usage)
            output_tokens = self._output_tokens(completion.usage)
            observed_prompt_tokens += (
                input_tokens if input_tokens is not None else estimated_tokens
            )
            observed_completion_tokens += (
                output_tokens
                if output_tokens is not None
                else self._estimate_text_tokens(completion.content)
            )
            usage_sources.update(
                {
                    "provider_usage" if input_tokens is not None else "deterministic_estimate",
                    "provider_usage" if output_tokens is not None else "deterministic_estimate",
                }
            )
            self._apply_usage_metadata(
                metadata,
                prompt_tokens=observed_prompt_tokens,
                completion_tokens=observed_completion_tokens,
                usage_sources=usage_sources,
                input_from_provider=input_tokens is not None,
                output_from_provider=output_tokens is not None,
            )
            invalid_content = completion.content
            raw = _RawUnderstanding.model_validate_json(completion.content)
            metadata.update(
                {
                    "understanding_source": "llm",
                    "understanding_provider": completion.provider,
                    "understanding_model": completion.reported_model,
                    "understanding_fallback_used": completion.fallback_used,
                    "understanding_repaired": False,
                }
            )
        except (ValidationError, json.JSONDecodeError, ValueError):
            try:
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": invalid_content[:8000]},
                    {
                        "role": "user",
                        "content": "The previous object violated the schema. Return one corrected object only.",
                    },
                ]
                repair = await self.llm.complete(
                    repair_messages,
                    response_schema=_RawUnderstanding.model_json_schema(),
                    schema_name="semikb_conversation_understanding_v1",
                    temperature=0,
                    max_output_tokens=1000,
                )
                metadata["understanding_calls"] = 2
                repair_input_tokens = self._input_tokens(repair.usage)
                repair_output_tokens = self._output_tokens(repair.usage)
                observed_prompt_tokens += (
                    repair_input_tokens
                    if repair_input_tokens is not None
                    else self._estimate_prompt_tokens(repair_messages)
                )
                observed_completion_tokens += (
                    repair_output_tokens
                    if repair_output_tokens is not None
                    else self._estimate_text_tokens(repair.content)
                )
                usage_sources.update(
                    {
                        "provider_usage"
                        if repair_input_tokens is not None
                        else "deterministic_estimate",
                        "provider_usage"
                        if repair_output_tokens is not None
                        else "deterministic_estimate",
                    }
                )
                self._apply_usage_metadata(
                    metadata,
                    prompt_tokens=observed_prompt_tokens,
                    completion_tokens=observed_completion_tokens,
                    usage_sources=usage_sources,
                    input_from_provider=(
                        input_tokens is not None and repair_input_tokens is not None
                    ),
                    output_from_provider=(
                        output_tokens is not None and repair_output_tokens is not None
                    ),
                )
                raw = _RawUnderstanding.model_validate_json(repair.content)
                metadata.update(
                    {
                        "understanding_source": "llm",
                        "understanding_provider": repair.provider,
                        "understanding_model": repair.reported_model,
                        "understanding_fallback_used": repair.fallback_used,
                        "understanding_repaired": True,
                    }
                )
            except (ValidationError, json.JSONDecodeError, ValueError, LLMProviderError):
                raw = None
        except LLMProviderError:
            raw = None

        if raw is None:
            fallback = self._heuristic(request, context, clarification_pending=clarification_pending)
            metadata.update(
                {
                    "understanding_source": "deterministic_fallback",
                    "understanding_warning": "structured_understanding_unavailable",
                }
            )
            return UnderstandingResult(fallback, self._finish_metadata(metadata, started))
        return UnderstandingResult(
            self._normalize_raw(
                raw,
                request,
                context,
                clarification_pending=clarification_pending,
            ),
            self._finish_metadata(metadata, started),
        )

    def _prompt(
        self,
        request: str,
        context: dict[str, Any],
        clarification_pending: bool,
        selection: IntentFewShotSelection,
    ) -> list[dict[str, Any]]:
        if not self.experiment_profile.include_catalog:
            return [
                {
                    "role": "system",
                    "content": (
                        "Classify one turn for a semiconductor knowledge Agent. Return only the strict schema object. "
                        "Preserve up to three explicit tasks. Separate emotion from task intent. Never invent Product, "
                        "Tool, Chamber, Recipe, Lot, Case, or time values. Use inherited_slot_names only for valid "
                        "context values needed by a pronoun or omitted reference. Recipe mutation, equipment control, "
                        "data deletion, and out-of-scope actions must use execution_policy=refuse. The suggested route "
                        "is advisory; server policy makes the final decision. Do not include reasoning or prose."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_request": request,
                            "clarification_pending": clarification_pending,
                            "conversation_context": ConversationUnderstandingService._safe_context(
                                context
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]

        system_content = (
            "Classify one turn for a semiconductor knowledge Agent. Return only the strict schema object. "
            "Preserve up to three explicit tasks. Separate emotion from task intent. Never invent Product, "
            "Tool, Chamber, Recipe, Lot, Case, or time values. Use inherited_slot_names only for valid "
            "context values needed by a pronoun or omitted reference. Recipe mutation, equipment control, "
            "data deletion, and out-of-scope actions must use execution_policy=refuse. The suggested route "
            "is advisory; server policy makes the final decision. For each task, compare against every active "
            "intent card supplied by the server and use a legal task signature from the closest matching card. "
            "The complete active catalog is authoritative; do not invent another intent. Do not include "
            "reasoning, card IDs, or prose in the response."
        )
        payload: dict[str, Any] = {
            "current_request": request,
            "clarification_pending": clarification_pending,
            "conversation_context": ConversationUnderstandingService._safe_context(context),
            "intent_catalog": self.intent_catalog.prompt_payload(),
        }
        if selection.examples:
            system_content += (
                " The supplied intent examples are classification precedents only. Use them to distinguish "
                "intent signatures, but extract slots, dependencies, context references, and entity values "
                "independently from the current request. Never copy an example entity into the current turn."
            )
            payload["intent_examples"] = selection.prompt_payload()
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def _intent_audit_metadata(
        self,
        *,
        prompt_tokens: int,
        cards_in_prompt: int,
    ) -> dict[str, Any]:
        return {
            "intent_catalog_version": self.intent_catalog.catalog_version,
            "intent_catalog_hash": self.intent_catalog.catalog_hash,
            "active_intent_card_count": len(self.intent_catalog.active_cards),
            "intent_card_selection": (
                "all_active" if self.experiment_profile.include_catalog else "none"
            ),
            "intent_cards_in_prompt": cards_in_prompt,
            "intent_prompt_tokens": prompt_tokens,
            "intent_catalog_capacity_warnings": self.intent_catalog.capacity_warnings(
                prompt_tokens=prompt_tokens
            ),
        }

    def _finish_metadata(
        self,
        metadata: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        prompt_tokens = int(metadata.get("intent_prompt_tokens", 0) or 0)
        metadata["intent_catalog_capacity_warnings"] = self.intent_catalog.capacity_warnings(
            prompt_tokens=prompt_tokens
        )
        metadata["understanding_latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            3,
        )
        return metadata

    @staticmethod
    def _input_tokens(usage: dict[str, Any]) -> int | None:
        for key in ("prompt_tokens", "input_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        return None

    @staticmethod
    def _output_tokens(usage: dict[str, Any]) -> int | None:
        for key in ("completion_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        return None

    @staticmethod
    def _apply_usage_metadata(
        metadata: dict[str, Any],
        *,
        prompt_tokens: int,
        completion_tokens: int,
        usage_sources: set[str],
        input_from_provider: bool,
        output_from_provider: bool,
    ) -> None:
        metadata["intent_prompt_tokens"] = prompt_tokens
        metadata["intent_completion_tokens"] = completion_tokens
        metadata["intent_total_tokens"] = prompt_tokens + completion_tokens
        metadata["intent_prompt_tokens_source"] = (
            "provider_usage" if input_from_provider else "deterministic_estimate"
        )
        metadata["intent_completion_tokens_source"] = (
            "provider_usage" if output_from_provider else "deterministic_estimate"
        )
        metadata["intent_usage_source"] = (
            "provider_usage" if usage_sources == {"provider_usage"} else "mixed_estimate"
        )

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return ConversationUnderstandingService._estimate_text_tokens(serialized)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        pieces = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
        return len(pieces)

    @staticmethod
    def _safe_context(context: dict[str, Any]) -> dict[str, Any]:
        recent = []
        for item in context.get("recent_messages", [])[-24:]:
            if isinstance(item, dict):
                recent.append(
                    {
                        "message_id": str(item.get("message_id", "")),
                        "role": str(item.get("role", "")),
                        "content": str(item.get("content", ""))[:600],
                    }
                )
        valid_slots = []
        active = context.get("active_context", {})
        slots = active.get("slots", {}) if isinstance(active, dict) else {}
        if isinstance(slots, dict):
            for name, item in slots.items():
                if isinstance(item, dict) and item.get("valid") is True:
                    valid_slots.append(
                        {
                            "slot_name": str(name),
                            "value": str(item.get("value", "")),
                            "source_message_id": str(item.get("source_message_id", "")),
                        }
                    )
        return {"recent_messages": recent, "valid_slots": valid_slots}

    def _normalize_raw(
        self,
        raw: _RawUnderstanding,
        request: str,
        context: dict[str, Any],
        *,
        clarification_pending: bool,
    ) -> ConversationUnderstanding:
        deterministic_slots = self.extract_explicit_slots(request)
        explicit = dict(deterministic_slots)
        for item in raw.explicit_slots:
            if item.slot_name in SLOT_NAMES and self._slot_value_is_grounded(
                item.slot_name,
                item.value,
                request,
            ):
                explicit[item.slot_name] = item.value.strip()

        valid_context = self._valid_context_slots(context)
        inherited = {
            name: valid_context[name]["value"]
            for name in raw.inherited_slot_names
            if name in valid_context
        }
        operations = self._normalize_operations(raw.slot_operations, explicit, valid_context)
        valid_message_ids = self._valid_context_message_ids(context)
        context_ids = [item for item in raw.context_message_ids if item in valid_message_ids]
        primary = self._normalize_primary_intent(raw.primary_intent, request)
        tasks = self._normalize_tasks(raw.task_items)
        history_recall_task = next(
            (
                item
                for item in tasks
                if item.target_type is IntentTarget.PREVIOUS_USER_MESSAGE
                and item.action is IntentTaskAction.RECALL
            ),
            None,
        )
        history_recall_only = history_recall_task is not None and len(tasks) == 1
        if history_recall_only:
            primary = PrimaryIntent.CONVERSATION
            tasks = [history_recall_task.model_copy(update={"primary_intent": primary})]
            previous_user = self._last_substantive_user_message_id(context)
            context_ids = [previous_user] if previous_user else []
        has_protected_task = any(term in request.lower() for term in MUTATION_TERMS) or any(
            term in request.lower() for term in ("生成报告", "写报告", "整理报告")
        )
        if not history_recall_only and (
            not tasks or primary is not raw.primary_intent or has_protected_task
        ):
            tasks = self._infer_tasks(request, primary)
        query = self._standalone_query(raw.standalone_query or request, request, inherited)
        return ConversationUnderstanding(
            classifier_source="llm",
            interaction_mode=self._normalize_interaction_mode(
                InteractionMode.CONVERSATION
                if history_recall_only
                else raw.interaction_mode,
                primary,
                request,
                clarification_pending,
            ),
            primary_intent=primary,
            task_items=tasks,
            affect=AffectSignals.model_validate(raw.affect.model_dump()),
            slot_operations=operations,
            explicit_slots=explicit,
            inherited_slots=inherited,
            missing_slots=[item for item in raw.missing_slots if item in SLOT_NAMES][:3],
            context_message_ids=context_ids,
            standalone_query=query,
            cancel_scope=raw.cancel_scope,
            suggested_route=(
                AgentRoute.HISTORY_DIRECT
                if history_recall_only
                else raw.suggested_route
            ),
            confidence=raw.confidence,
        )

    @staticmethod
    def _normalize_primary_intent(
        proposed: PrimaryIntent,
        request: str,
    ) -> PrimaryIntent:
        lowered = request.lower()
        if ConversationUnderstandingService._is_history_recall_request(request):
            return PrimaryIntent.CONVERSATION
        if any(term in lowered for term in MUTATION_TERMS):
            return PrimaryIntent.ACTION_REQUEST
        if any(term in lowered for term in ("简化", "总结", "翻译", "改写")) and any(
            term in lowered for term in ("回答", "答案", "上面", "刚才", "上一轮")
        ):
            return PrimaryIntent.CONTENT_TASK
        if any(term in lowered for term in ALARM_TERMS) and any(
            term in lowered for term in ("含义", "解释", "定义", "代表什么", "手册")
        ):
            return PrimaryIntent.KNOWLEDGE_QUERY
        if any(term in lowered for term in EXTERNAL_TERMS):
            return PrimaryIntent.KNOWLEDGE_QUERY
        if any(
            term in lowered
            for term in ("根因", "原因", "为什么", "排查", "诊断", "影响分析", "分析原因", "良率下降")
        ):
            return PrimaryIntent.INVESTIGATION
        if "sop" in lowered and any(
            term in lowered for term in ("查", "要求", "规定", "怎么做", "流程", "比较", "对照")
        ):
            return PrimaryIntent.KNOWLEDGE_QUERY
        if any(term in lowered for term in ("查", "查询", "看")) and any(
            term in lowered
            for term in ("fdc", "spc", "报警", "alarm", "lot", "wafer", "良率", "制造数据", "趋势")
        ):
            return PrimaryIntent.DATA_QUERY
        return proposed

    @staticmethod
    def _normalize_interaction_mode(
        proposed: InteractionMode,
        primary: PrimaryIntent,
        request: str,
        clarification_pending: bool,
    ) -> InteractionMode:
        if clarification_pending:
            return InteractionMode.CLARIFICATION_ANSWER
        if proposed is InteractionMode.CLARIFICATION_ANSWER:
            proposed = InteractionMode.TASK
        has_greeting = bool(
            re.search(r"(?:^|[，,。])\s*(?:你好|您好|hi|hello)", request, re.IGNORECASE)
        )
        has_business = any(
            term in request.lower()
            for term in (
                "sop",
                "recipe",
                "fdc",
                "spc",
                "报警",
                "alarm",
                "良率",
                "缺陷",
                "lot",
                "wafer",
                "查",
                "看",
                "分析",
            )
        )
        if has_greeting and has_business:
            return InteractionMode.MIXED
        if primary not in {PrimaryIntent.CONVERSATION, PrimaryIntent.CONTENT_TASK} and proposed in {
            InteractionMode.CONVERSATION,
            InteractionMode.FEEDBACK,
        }:
            return InteractionMode.TASK
        return proposed

    def _l0(
        self,
        request: str,
        context: dict[str, Any],
    ) -> ConversationUnderstanding | None:
        content = request.strip()
        normalized = re.sub(r"\s+", "", content).lower()
        previous_user = self._last_substantive_user_message_id(context)
        previous_answer = self._last_message_id(context, "assistant")

        if HISTORY_RECALL_PATTERN.search(content):
            return self._direct_understanding(
                InteractionMode.CONVERSATION,
                PrimaryIntent.CONVERSATION,
                IntentTarget.PREVIOUS_USER_MESSAGE,
                IntentTaskAction.RECALL,
                AgentRoute.HISTORY_DIRECT,
                context_message_ids=[previous_user] if previous_user else [],
            )
        if PREVIOUS_ANSWER_PATTERN.search(content) or PREVIOUS_ANSWER_PREFIX_ACTION_PATTERN.search(content):
            action = IntentTaskAction.SIMPLIFY
            if "总结" in content:
                action = IntentTaskAction.SUMMARIZE
            elif "翻译" in content:
                action = IntentTaskAction.TRANSLATE
            return self._direct_understanding(
                InteractionMode.FEEDBACK if FEEDBACK_PATTERN.search(content) else InteractionMode.CONVERSATION,
                PrimaryIntent.CONTENT_TASK,
                IntentTarget.PREVIOUS_ANSWER,
                action,
                AgentRoute.HISTORY_DIRECT,
                context_message_ids=[previous_answer] if previous_answer else [],
            )
        if FEEDBACK_PATTERN.search(content):
            return self._direct_understanding(
                InteractionMode.FEEDBACK,
                PrimaryIntent.CONVERSATION,
                IntentTarget.PREVIOUS_ANSWER,
                IntentTaskAction.EXPLAIN,
                AgentRoute.CHAT_DIRECT,
                context_message_ids=[previous_answer] if previous_answer else [],
                affect=AffectSignals(sentiment="negative", complaint_signal=True),
            )
        if CANCEL_PATTERN.fullmatch(content):
            scope = (
                CancelScope.CURRENT_GENERATION
                if "停止生成" in content
                else CancelScope.CLARIFICATION
                if "追问" in content
                else CancelScope.CURRENT_TASK
            )
            value = self._direct_understanding(
                InteractionMode.CONTROL,
                PrimaryIntent.CONVERSATION,
                IntentTarget.GENERAL,
                IntentTaskAction.EXECUTE,
                AgentRoute.CHAT_DIRECT,
            )
            return value.model_copy(update={"cancel_scope": scope})
        correction = CORRECTION_PATTERN.search(content)
        if correction:
            old, new = correction.groups()
            slot_name = "tool_id"
            explicit = {slot_name: new.upper()}
            operation = SlotOperation(
                operation=SlotOperationKind.CORRECT,
                slot_name=slot_name,
                value=new.upper(),
            )
            value = self._direct_understanding(
                InteractionMode.CONTROL,
                PrimaryIntent.CONVERSATION,
                IntentTarget.GENERAL,
                IntentTaskAction.EXECUTE,
                AgentRoute.CHAT_DIRECT,
            )
            return value.model_copy(
                update={
                    "explicit_slots": explicit,
                    "slot_operations": [operation],
                    "standalone_query": f"correct {old.upper()} to {new.upper()}",
                }
            )
        if GREETING_PATTERN.fullmatch(content) or normalized in {"/help", "/new"}:
            return self._direct_understanding(
                InteractionMode.CONVERSATION,
                PrimaryIntent.CONVERSATION,
                IntentTarget.GENERAL,
                IntentTaskAction.EXPLAIN,
                AgentRoute.CHAT_DIRECT,
            )
        if OOD_PATTERN.search(content):
            value = self._direct_understanding(
                InteractionMode.TASK,
                PrimaryIntent.ACTION_REQUEST,
                IntentTarget.GENERAL,
                IntentTaskAction.EXECUTE,
                AgentRoute.REFUSE,
            )
            return value.model_copy(
                update={
                    "task_items": [
                        value.task_items[0].model_copy(
                            update={"execution_policy": TaskExecutionDecision.REFUSE}
                        )
                    ]
                }
            )
        return None

    def _heuristic(
        self,
        request: str,
        context: dict[str, Any],
        *,
        clarification_pending: bool,
    ) -> ConversationUnderstanding:
        lowered = request.lower()
        explicit = self.extract_explicit_slots(request)
        valid_context = self._valid_context_slots(context)
        inherited = {}
        if any(term in lowered for term in REFERENCE_TERMS):
            inherited = {name: item["value"] for name, item in valid_context.items()}

        has_greeting = bool(re.search(r"(?:^|[，,。])\s*(?:你好|您好|hi|hello)", request, re.IGNORECASE))
        has_business = any(
            term in lowered
            for term in (
                "sop",
                "recipe",
                "fdc",
                "spc",
                "报警",
                "alarm",
                "良率",
                "缺陷",
                "lot",
                "wafer",
                "查",
                "看",
                "分析",
            )
        )
        if clarification_pending:
            mode = InteractionMode.CLARIFICATION_ANSWER
        elif has_greeting and has_business:
            mode = InteractionMode.MIXED
        elif FEEDBACK_PATTERN.search(request):
            mode = InteractionMode.FEEDBACK
        elif any(term in lowered for term in ("取消", "别查", "停止", "不是", "改成", "换成")):
            mode = InteractionMode.CONTROL
        else:
            mode = InteractionMode.TASK

        if any(term in lowered for term in MUTATION_TERMS):
            primary = PrimaryIntent.ACTION_REQUEST
        elif any(term in lowered for term in ("简化", "总结", "翻译", "改写", "报告")):
            primary = PrimaryIntent.CONTENT_TASK
        elif any(term in lowered for term in EXTERNAL_TERMS):
            primary = PrimaryIntent.KNOWLEDGE_QUERY
        elif "sop" in lowered and any(
            term in lowered for term in ("要求", "规定", "怎么做", "如何处理", "流程")
        ):
            primary = PrimaryIntent.KNOWLEDGE_QUERY
        elif any(
            term in lowered
            for term in ("根因", "原因", "为什么", "排查", "诊断", "影响分析", "分析原因", "良率下降")
        ):
            primary = PrimaryIntent.INVESTIGATION
        elif any(term in lowered for term in ("含义", "解释")):
            primary = PrimaryIntent.KNOWLEDGE_QUERY
        elif any(
            term in lowered
            for term in ("fdc", "spc", "报警", "alarm", "lot", "wafer", "制造数据", "趋势", "良率")
        ):
            primary = PrimaryIntent.DATA_QUERY
        else:
            primary = PrimaryIntent.KNOWLEDGE_QUERY

        tasks = self._infer_tasks(request, primary)
        suggested = self._suggest_route(primary, request, tasks)
        operations = [
            SlotOperation(operation=SlotOperationKind.SET, slot_name=name, value=value)
            for name, value in explicit.items()
        ]
        operations.extend(
            SlotOperation(
                operation=SlotOperationKind.INHERIT,
                slot_name=name,
                value=value,
                source_message_id=valid_context[name]["source_message_id"],
            )
            for name, value in inherited.items()
        )
        context_ids = []
        if inherited:
            context_ids = list(
                dict.fromkeys(item["source_message_id"] for item in valid_context.values())
            )[:8]
        confidence = 0.82 if tasks else 0.65
        return ConversationUnderstanding(
            classifier_source="deterministic_fallback",
            interaction_mode=mode,
            primary_intent=primary,
            task_items=tasks,
            affect=self._affect(request),
            slot_operations=operations,
            explicit_slots=explicit,
            inherited_slots=inherited,
            context_message_ids=context_ids,
            standalone_query=self._standalone_query(request, request, inherited),
            suggested_route=suggested,
            confidence=confidence,
        )

    @staticmethod
    def _direct_understanding(
        mode: InteractionMode,
        primary: PrimaryIntent,
        target: IntentTarget,
        action: IntentTaskAction,
        route: AgentRoute,
        *,
        context_message_ids: list[str] | None = None,
        affect: AffectSignals | None = None,
    ) -> ConversationUnderstanding:
        return ConversationUnderstanding(
            classifier_source="l0",
            interaction_mode=mode,
            primary_intent=primary,
            task_items=[
                IntentTaskItem(
                    task_id="task_1",
                    primary_intent=primary,
                    target_type=target,
                    action=action,
                )
            ],
            affect=affect or AffectSignals(),
            context_message_ids=context_message_ids or [],
            suggested_route=route,
            confidence=0.99,
        )

    @staticmethod
    def extract_explicit_slots(request: str) -> dict[str, str]:
        slots: dict[str, str] = {}
        for name, pattern in ENTITY_PATTERNS.items():
            match = pattern.search(request)
            if match:
                slots[name] = match.group(0).replace(" ", "-").upper()
        chamber = CHAMBER_PATTERN.search(request)
        if chamber:
            slots["chamber"] = chamber.group(1).upper()
        time_range = TIME_PATTERN.search(request)
        if time_range:
            slots["time_range"] = time_range.group(0).strip()
        return slots

    @staticmethod
    def _infer_tasks(request: str, primary: PrimaryIntent) -> list[IntentTaskItem]:
        lowered = request.lower()
        candidates: list[tuple[PrimaryIntent, IntentTarget, IntentTaskAction, TaskExecutionDecision]] = []

        if any(term in lowered for term in MUTATION_TERMS) and not any(
            term in lowered for term in ("查", "查询", "看", "分析", "并且", "然后", "、")
        ):
            return [
                IntentTaskItem(
                    task_id="task_1",
                    primary_intent=PrimaryIntent.ACTION_REQUEST,
                    target_type=IntentTarget.RECIPE if "recipe" in lowered or "配方" in lowered else IntentTarget.GENERAL,
                    action=IntentTaskAction.EXECUTE,
                    execution_policy=TaskExecutionDecision.REFUSE,
                )
            ]

        if primary is PrimaryIntent.KNOWLEDGE_QUERY and any(
            term in lowered for term in ALARM_TERMS
        ) and any(term in lowered for term in ("含义", "解释", "定义", "代表什么", "手册")):
            candidates.append(
                (
                    PrimaryIntent.KNOWLEDGE_QUERY,
                    IntentTarget.ALARM,
                    IntentTaskAction.EXPLAIN,
                    TaskExecutionDecision.EXECUTE,
                )
            )
        if primary is PrimaryIntent.KNOWLEDGE_QUERY and any(
            term in lowered for term in EXTERNAL_TERMS
        ):
            candidates.append(
                (
                    PrimaryIntent.KNOWLEDGE_QUERY,
                    IntentTarget.GENERAL,
                    IntentTaskAction.LOOKUP,
                    TaskExecutionDecision.EXECUTE,
                )
            )

        data_requested = primary in {PrimaryIntent.DATA_QUERY, PrimaryIntent.INVESTIGATION} or (
            primary is PrimaryIntent.ACTION_REQUEST
            and any(term in lowered for term in ("查", "查询", "看"))
        )
        if data_requested:
            if any(term in lowered for term in ("fdc", "报警", "alarm")):
                candidates.append((PrimaryIntent.DATA_QUERY, IntentTarget.FDC, IntentTaskAction.LOOKUP, TaskExecutionDecision.EXECUTE))
            elif "spc" in lowered or "趋势" in lowered:
                candidates.append((PrimaryIntent.DATA_QUERY, IntentTarget.SPC, IntentTaskAction.LOOKUP, TaskExecutionDecision.EXECUTE))
            elif any(term in lowered for term in ("lot", "wafer", "晶圆图", "良率")):
                candidates.append((PrimaryIntent.DATA_QUERY, IntentTarget.LOT, IntentTaskAction.LOOKUP, TaskExecutionDecision.EXECUTE))

        if "sop" in lowered:
            action = IntentTaskAction.COMPARE if any(term in lowered for term in ("对照", "比较")) else IntentTaskAction.LOOKUP
            candidates.append((PrimaryIntent.KNOWLEDGE_QUERY, IntentTarget.SOP, action, TaskExecutionDecision.EXECUTE))
        if "recipe" in lowered or "配方" in lowered:
            mutation = any(term in lowered for term in MUTATION_TERMS)
            action = IntentTaskAction.EXECUTE if mutation else IntentTaskAction.EXPLAIN
            decision = TaskExecutionDecision.REFUSE if mutation else TaskExecutionDecision.EXECUTE
            candidates.append((PrimaryIntent.ACTION_REQUEST if mutation else PrimaryIntent.KNOWLEDGE_QUERY, IntentTarget.RECIPE, action, decision))

        if any(
            term in lowered
            for term in ("根因", "原因", "为什么", "排查", "诊断", "分析原因", "给排查建议", "良率下降")
        ):
            candidates.append((PrimaryIntent.INVESTIGATION, IntentTarget.CASE, IntentTaskAction.DIAGNOSE, TaskExecutionDecision.EXECUTE))
        if any(term in lowered for term in ("生成报告", "写报告", "整理报告")):
            candidates.append((PrimaryIntent.CONTENT_TASK, IntentTarget.REPORT, IntentTaskAction.GENERATE, TaskExecutionDecision.DEFER))
        if any(term in lowered for term in MUTATION_TERMS) and not any(
            item[1] is IntentTarget.RECIPE and item[3] is TaskExecutionDecision.REFUSE
            for item in candidates
        ):
            candidates.append(
                (
                    PrimaryIntent.ACTION_REQUEST,
                    IntentTarget.GENERAL,
                    IntentTaskAction.EXECUTE,
                    TaskExecutionDecision.REFUSE,
                )
            )
        if any(term in lowered for term in ("简化", "简单解释", "说简单", "总结", "翻译", "改写")):
            action = IntentTaskAction.SIMPLIFY
            if "总结" in lowered:
                action = IntentTaskAction.SUMMARIZE
            elif "翻译" in lowered:
                action = IntentTaskAction.TRANSLATE
            candidates.append((PrimaryIntent.CONTENT_TASK, IntentTarget.PREVIOUS_ANSWER, action, TaskExecutionDecision.EXECUTE))

        if not candidates:
            target = IntentTarget.SOP if primary is PrimaryIntent.KNOWLEDGE_QUERY else IntentTarget.GENERAL
            action = IntentTaskAction.LOOKUP if primary is not PrimaryIntent.CONVERSATION else IntentTaskAction.EXPLAIN
            candidates.append((primary, target, action, TaskExecutionDecision.EXECUTE))

        unique: list[tuple[PrimaryIntent, IntentTarget, IntentTaskAction, TaskExecutionDecision]] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        tasks: list[IntentTaskItem] = []
        for index, (intent, target, action, decision) in enumerate(unique[:3], start=1):
            depends_on = []
            if intent is PrimaryIntent.INVESTIGATION and index > 1:
                depends_on = [item.task_id for item in tasks if item.primary_intent in {PrimaryIntent.DATA_QUERY, PrimaryIntent.KNOWLEDGE_QUERY}][-2:]
            tasks.append(
                IntentTaskItem(
                    task_id=f"task_{index}",
                    primary_intent=intent,
                    target_type=target,
                    action=action,
                    depends_on=depends_on,
                    execution_policy=decision,
                )
            )
        return tasks

    @staticmethod
    def _suggest_route(
        primary: PrimaryIntent,
        request: str,
        tasks: list[IntentTaskItem],
    ) -> AgentRoute:
        lowered = request.lower()
        if tasks and all(item.execution_policy is TaskExecutionDecision.REFUSE for item in tasks):
            return AgentRoute.REFUSE
        intents = {item.primary_intent for item in tasks}
        if PrimaryIntent.INVESTIGATION in intents or (
            PrimaryIntent.DATA_QUERY in intents and PrimaryIntent.KNOWLEDGE_QUERY in intents
        ):
            return AgentRoute.RAG_AND_TOOL
        if any(term in lowered for term in EXTERNAL_TERMS):
            return AgentRoute.RAG_AND_WEB
        if primary is PrimaryIntent.DATA_QUERY:
            return AgentRoute.TOOL_ONLY
        if primary in {PrimaryIntent.CONVERSATION, PrimaryIntent.CONTENT_TASK}:
            return AgentRoute.HISTORY_DIRECT
        return AgentRoute.INTERNAL_RAG

    @staticmethod
    def _affect(request: str) -> AffectSignals:
        lowered = request.lower()
        negative = any(term in lowered for term in ("烦", "太慢", "糟糕", "不满意", "看不懂"))
        positive = any(term in lowered for term in ("谢谢", "很好", "明白", "不错"))
        urgent = any(term in lowered for term in ("赶紧", "马上", "紧急", "尽快", "立刻"))
        return AffectSignals(
            sentiment="negative" if negative else "positive" if positive else "neutral",
            urgency="urgent" if urgent else "normal",
            complaint_signal=negative,
        )

    @staticmethod
    def _normalize_tasks(tasks: list[_RawTask]) -> list[IntentTaskItem]:
        normalized: list[IntentTaskItem] = []
        valid_ids: set[str] = set()
        for index, raw in enumerate(tasks[:3], start=1):
            task_id = f"task_{index}"
            depends = [item for item in raw.depends_on if item in valid_ids][-2:]
            normalized.append(
                IntentTaskItem(
                    task_id=task_id,
                    primary_intent=raw.primary_intent,
                    target_type=raw.target_type,
                    action=raw.action,
                    depends_on=depends,
                    execution_policy=raw.execution_policy,
                )
            )
            valid_ids.add(task_id)
        return normalized

    @staticmethod
    def _normalize_operations(
        operations: list[_RawSlotOperation],
        explicit: dict[str, str],
        valid_context: dict[str, dict[str, str]],
    ) -> list[SlotOperation]:
        normalized: list[SlotOperation] = []
        for raw in operations:
            if raw.slot_name not in SLOT_NAMES:
                continue
            if raw.operation in {SlotOperationKind.SET, SlotOperationKind.CORRECT}:
                value = explicit.get(raw.slot_name)
                if not value:
                    continue
                normalized.append(
                    SlotOperation(operation=raw.operation, slot_name=raw.slot_name, value=value)
                )
            elif raw.operation is SlotOperationKind.INHERIT and raw.slot_name in valid_context:
                item = valid_context[raw.slot_name]
                normalized.append(
                    SlotOperation(
                        operation=raw.operation,
                        slot_name=raw.slot_name,
                        value=item["value"],
                        source_message_id=item["source_message_id"],
                    )
                )
            elif raw.operation is SlotOperationKind.CLEAR:
                normalized.append(
                    SlotOperation(operation=raw.operation, slot_name=raw.slot_name)
                )
        return normalized[:12]

    @staticmethod
    def _valid_context_slots(context: dict[str, Any]) -> dict[str, dict[str, str]]:
        active = context.get("active_context", {})
        slots = active.get("slots", {}) if isinstance(active, dict) else {}
        if not isinstance(slots, dict):
            return {}
        return {
            str(name): {
                "value": str(item.get("value", "")),
                "source_message_id": str(item.get("source_message_id", "")),
            }
            for name, item in slots.items()
            if name in SLOT_NAMES
            and isinstance(item, dict)
            and item.get("valid") is True
            and item.get("value")
            and item.get("source_message_id")
        }

    @staticmethod
    def _valid_context_message_ids(context: dict[str, Any]) -> set[str]:
        return {
            str(item.get("message_id"))
            for item in context.get("recent_messages", [])
            if isinstance(item, dict) and item.get("message_id")
        }

    @staticmethod
    def _last_message_id(context: dict[str, Any], role: str) -> str | None:
        for item in reversed(context.get("recent_messages", [])):
            if isinstance(item, dict) and item.get("role") == role and item.get("message_id"):
                return str(item["message_id"])
        return None

    @staticmethod
    def _is_history_recall_request(content: str) -> bool:
        return HISTORY_RECALL_PATTERN.fullmatch(content.strip()) is not None

    @classmethod
    def _last_substantive_user_message_id(cls, context: dict[str, Any]) -> str | None:
        for item in reversed(context.get("recent_messages", [])):
            if not isinstance(item, dict) or item.get("role") != "user" or not item.get("message_id"):
                continue
            if cls._is_history_recall_request(str(item.get("content", ""))):
                continue
            return str(item["message_id"])
        return None

    @staticmethod
    def _value_is_grounded(value: str, request: str) -> bool:
        normalized_value = re.sub(r"\s+", "", value).lower()
        normalized_request = re.sub(r"\s+", "", request).lower()
        return bool(normalized_value and normalized_value in normalized_request)

    @staticmethod
    def _slot_value_is_grounded(slot_name: str, value: str, request: str) -> bool:
        if not ConversationUnderstandingService._value_is_grounded(value, request):
            return False
        candidate = value.strip()
        if slot_name in ENTITY_PATTERNS:
            return ENTITY_PATTERNS[slot_name].fullmatch(candidate) is not None
        if slot_name == "chamber":
            match = CHAMBER_PATTERN.search(request)
            return match is not None and match.group(1).casefold() == candidate.casefold()
        if slot_name == "time_range":
            return any(
                match.group(0).strip().casefold() == candidate.casefold()
                for match in TIME_PATTERN.finditer(request)
            )
        return True

    @staticmethod
    def _standalone_query(candidate: str, request: str, inherited: dict[str, str]) -> str:
        value = candidate.strip() or request.strip()
        if inherited and any(term in request.lower() for term in REFERENCE_TERMS):
            suffix = " ".join(f"{name}={slot}" for name, slot in sorted(inherited.items()))
            if suffix and suffix.lower() not in value.lower():
                value = f"{value}\n已确认上下文：{suffix}"
        return value[:8000]
