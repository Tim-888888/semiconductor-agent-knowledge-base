from __future__ import annotations

import pytest

from semikb.agent_runtime.task_execution import (
    TaskExecutionCoordinator,
    route_generation_contract,
)
from semikb.contracts.models import (
    AgentRoute,
    EvidenceLedgerEntry,
    IntentTarget,
    IntentTaskAction,
    IntentTaskItem,
    PrimaryIntent,
    RoutePlan,
    RouteTaskDecision,
    TaskExecutionDecision,
    TaskExecutionStatus,
)


def _task(
    task_id: str,
    target: IntentTarget,
    action: IntentTaskAction = IntentTaskAction.LOOKUP,
    *,
    depends_on: list[str] | None = None,
) -> IntentTaskItem:
    return IntentTaskItem(
        task_id=task_id,
        primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
        target_type=target,
        action=action,
        depends_on=depends_on or [],
    )


def _plan(*decisions: RouteTaskDecision, route: AgentRoute) -> RoutePlan:
    return RoutePlan(route=route, confidence=0.95, task_decisions=list(decisions))


@pytest.mark.parametrize(
    ("route", "ledger", "expected"),
    [
        (AgentRoute.INTERNAL_RAG, [], TaskExecutionStatus.FAILED),
        (
            AgentRoute.INTERNAL_RAG,
            [
                EvidenceLedgerEntry(
                    evidence_id="chunk:sop",
                    source_type="internal_controlled",
                    content="SOP fact",
                )
            ],
            TaskExecutionStatus.COMPLETED,
        ),
        (
            AgentRoute.TOOL_ONLY,
            [
                EvidenceLedgerEntry(
                    evidence_id="tool:fdc",
                    source_type="simulated_live_data",
                    content="FDC fact",
                )
            ],
            TaskExecutionStatus.COMPLETED,
        ),
        (
            AgentRoute.RAG_AND_TOOL,
            [
                EvidenceLedgerEntry(
                    evidence_id="chunk:sop",
                    source_type="internal_controlled",
                    content="SOP fact",
                )
            ],
            TaskExecutionStatus.FAILED,
        ),
    ],
)
def test_route_contracts_require_the_expected_evidence(
    route: AgentRoute,
    ledger: list[EvidenceLedgerEntry],
    expected: TaskExecutionStatus,
) -> None:
    task = _task("task_1", IntentTarget.SOP)
    plan = _plan(
        RouteTaskDecision(
            task_id="task_1",
            decision=TaskExecutionDecision.EXECUTE,
            route=route,
            reason_code="task_allowed",
        ),
        route=route,
    )

    result = TaskExecutionCoordinator().finalize(
        route_plan=plan,
        task_items=[task],
        actual_route=route,
        answer_text="validated answer",
        evidence_ledger=ledger,
        cited_evidence_ids=[item.evidence_id for item in ledger],
    )

    assert result[0].status is expected


def test_failed_dependency_defers_following_task() -> None:
    tasks = [
        _task("task_1", IntentTarget.RECIPE, IntentTaskAction.EXECUTE),
        _task(
            "task_2",
            IntentTarget.REPORT,
            IntentTaskAction.GENERATE,
            depends_on=["task_1"],
        ),
    ]
    plan = _plan(
        RouteTaskDecision(
            task_id="task_1",
            decision=TaskExecutionDecision.REFUSE,
            route=None,
            reason_code="controlled_write_not_allowed",
        ),
        RouteTaskDecision(
            task_id="task_2",
            decision=TaskExecutionDecision.EXECUTE,
            route=AgentRoute.CHAT_DIRECT,
            reason_code="task_allowed",
        ),
        route=AgentRoute.CHAT_DIRECT,
    )

    results = TaskExecutionCoordinator().finalize(
        route_plan=plan,
        task_items=tasks,
        actual_route=AgentRoute.CHAT_DIRECT,
        answer_text="answer",
    )

    assert [item.status for item in results] == [
        TaskExecutionStatus.REFUSED,
        TaskExecutionStatus.DEFERRED,
    ]
    assert results[1].reason_code == "dependency_not_completed"


def test_rag_and_web_can_degrade_without_overriding_internal_evidence() -> None:
    task = _task("task_1", IntentTarget.SOP)
    plan = _plan(
        RouteTaskDecision(
            task_id="task_1",
            decision=TaskExecutionDecision.EXECUTE,
            route=AgentRoute.RAG_AND_WEB,
            reason_code="task_allowed",
        ),
        route=AgentRoute.RAG_AND_WEB,
    )

    result = TaskExecutionCoordinator().finalize(
        route_plan=plan,
        task_items=[task],
        actual_route=AgentRoute.RAG_AND_WEB,
        answer_text="internal answer",
        evidence_ledger=[
            EvidenceLedgerEntry(
                evidence_id="chunk:sop",
                source_type="internal_controlled",
                content="SOP fact",
            )
        ],
        cited_evidence_ids=["chunk:sop"],
        external_evidence=[{"source_type": "external_unavailable"}],
    )[0]

    assert result.status is TaskExecutionStatus.COMPLETED
    assert result.reason_code == "completed_with_external_degradation"
    assert result.validation_warnings == ["external_evidence_unavailable"]
    assert "internal controlled documents as the authority" in route_generation_contract(
        AgentRoute.RAG_AND_WEB
    )
