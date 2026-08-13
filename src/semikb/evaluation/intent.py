"""Versioned offline evaluation for controlled conversation understanding and routing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.contracts.models import (
    ActorScope,
    AgentRoute,
    InteractionMode,
    PrimaryIntent,
    SlotOperationKind,
    TaskExecutionDecision,
)


class IntentEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    utterance: str
    context: dict[str, Any] = Field(default_factory=dict)
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    clarification_pending: bool = False
    expected_interaction_mode: InteractionMode
    expected_primary_intent: PrimaryIntent
    expected_route: AgentRoute
    expected_task_count: int = Field(ge=1, le=3)
    expected_refused_task_count: int = Field(default=0, ge=0, le=3)
    expected_slot_operation: SlotOperationKind | None = None
    tags: list[str] = Field(default_factory=list)


class IntentEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str = "semikb-intent-v1"
    source_kind: str = "synthetic_review_required"
    description: str
    cases: list[IntentEvaluationCase] = Field(min_length=80, max_length=120)

    @property
    def dataset_hash(self) -> str:
        payload = json.dumps(
            [case.model_dump(mode="json") for case in self.cases],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> IntentEvaluationDataset:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class IntentEvaluationResult(BaseModel):
    dataset_version: str
    dataset_hash: str
    evaluated_cases: int
    metrics: dict[str, float]
    failures: list[dict[str, Any]] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)


class IntentEvaluationRunner:
    def __init__(
        self,
        understanding: ConversationUnderstandingService,
        policy: RoutePolicy | None = None,
    ) -> None:
        self.understanding = understanding
        self.policy = policy or RoutePolicy()

    async def run(
        self,
        dataset: IntentEvaluationDataset,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> IntentEvaluationResult:
        selected = dataset.cases[offset:]
        cases = selected[:limit] if limit else selected
        correct_mode = 0
        correct_intent = 0
        expected_routes: Counter[AgentRoute] = Counter()
        actual_routes: Counter[AgentRoute] = Counter()
        true_routes: Counter[AgentRoute] = Counter()
        missed_tasks = 0
        expected_tasks = 0
        unnecessary_retrieval = 0
        direct_expected = 0
        wrong_reuse = 0
        reuse_predictions = 0
        wrong_clarify = 0
        non_clarify_expected = 0
        correction_failures = 0
        correction_cases = 0
        dangerous_failures = 0
        dangerous_cases = 0
        failures: list[dict[str, Any]] = []
        sources: Counter[str] = Counter()
        retrieval_routes = {
            AgentRoute.REUSE_EVIDENCE,
            AgentRoute.INTERNAL_RAG,
            AgentRoute.RAG_AND_TOOL,
            AgentRoute.RAG_AND_WEB,
        }

        for case in cases:
            result = await self.understanding.understand(
                case.utterance,
                case.context,
                clarification_pending=case.clarification_pending,
            )
            interpreted = result.understanding
            plan = self.policy.decide(
                interpreted,
                case.actor_scope,
                case.context,
                case.utterance,
            )
            sources[interpreted.classifier_source] += 1
            mode_ok = interpreted.interaction_mode is case.expected_interaction_mode
            intent_ok = interpreted.primary_intent is case.expected_primary_intent
            route_ok = plan.route is case.expected_route
            correct_mode += int(mode_ok)
            correct_intent += int(intent_ok)
            expected_routes[case.expected_route] += 1
            actual_routes[plan.route] += 1
            true_routes[case.expected_route] += int(route_ok)

            expected_tasks += case.expected_task_count
            missed_tasks += max(case.expected_task_count - len(interpreted.task_items), 0)
            if case.expected_route in {
                AgentRoute.HISTORY_DIRECT,
                AgentRoute.CHAT_DIRECT,
                AgentRoute.TOOL_ONLY,
                AgentRoute.CLARIFY,
                AgentRoute.REFUSE,
            }:
                direct_expected += 1
                unnecessary_retrieval += int(plan.route in retrieval_routes)
            if plan.route is AgentRoute.REUSE_EVIDENCE:
                reuse_predictions += 1
                wrong_reuse += int(case.expected_route is not AgentRoute.REUSE_EVIDENCE)
            if case.expected_route is not AgentRoute.CLARIFY:
                non_clarify_expected += 1
                wrong_clarify += int(plan.route is AgentRoute.CLARIFY)
            if case.expected_slot_operation is not None:
                correction_cases += 1
                correction_failures += int(
                    not any(
                        item.operation is case.expected_slot_operation
                        for item in interpreted.slot_operations
                    )
                )
            if case.expected_refused_task_count:
                dangerous_cases += 1
                actual_refused = sum(
                    item.decision is TaskExecutionDecision.REFUSE
                    for item in plan.task_decisions
                )
                dangerous_failures += int(actual_refused < case.expected_refused_task_count)

            if not (mode_ok and intent_ok and route_ok) or len(interpreted.task_items) < case.expected_task_count:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "expected": {
                            "interaction_mode": case.expected_interaction_mode,
                            "primary_intent": case.expected_primary_intent,
                            "route": case.expected_route,
                            "task_count": case.expected_task_count,
                        },
                        "actual": {
                            "interaction_mode": interpreted.interaction_mode,
                            "primary_intent": interpreted.primary_intent,
                            "route": plan.route,
                            "task_count": len(interpreted.task_items),
                            "reason_codes": plan.reason_codes,
                        },
                    }
                )

        route_precisions = [
            true_routes[route] / actual_routes[route]
            for route in AgentRoute
            if actual_routes[route]
        ]
        route_recalls = [
            true_routes[route] / expected_routes[route]
            for route in AgentRoute
            if expected_routes[route]
        ]
        total = max(len(cases), 1)
        metrics = {
            "interaction_mode_accuracy": correct_mode / total,
            "primary_intent_accuracy": correct_intent / total,
            "route_accuracy": sum(true_routes.values()) / total,
            "route_macro_precision": sum(route_precisions) / max(len(route_precisions), 1),
            "route_macro_recall": sum(route_recalls) / max(len(route_recalls), 1),
            "multi_task_miss_rate": missed_tasks / max(expected_tasks, 1),
            "unnecessary_retrieval_rate": unnecessary_retrieval / max(direct_expected, 1),
            "wrong_evidence_reuse_rate": wrong_reuse / max(reuse_predictions, 1),
            "wrong_clarification_rate": wrong_clarify / max(non_clarify_expected, 1),
            "slot_correction_failure_rate": correction_failures / max(correction_cases, 1),
            "dangerous_execution_miss_rate": dangerous_failures / max(dangerous_cases, 1),
        }
        return IntentEvaluationResult(
            dataset_version=dataset.dataset_version,
            dataset_hash=dataset.dataset_hash,
            evaluated_cases=len(cases),
            metrics={key: round(value, 6) for key, value in metrics.items()},
            failures=failures,
            source_counts=dict(sources),
        )
