"""Server-owned mapping from audited routes to message presentation modes."""

from __future__ import annotations

from typing import Any

from semikb.contracts.models import (
    AgentAnswer,
    AgentRoute,
    EvidenceLedgerEntry,
    MessagePresentation,
    MessageRenderMode,
    TaskExecutionResult,
)

STRUCTURED_CARD_ROUTES = frozenset(
    {
        AgentRoute.REUSE_EVIDENCE,
        AgentRoute.INTERNAL_RAG,
        AgentRoute.TOOL_ONLY,
        AgentRoute.RAG_AND_TOOL,
        AgentRoute.RAG_AND_WEB,
    }
)


def build_message_presentation(
    *,
    route: AgentRoute | str | None,
    answer: AgentAnswer | dict[str, Any] | None,
    status: str | None,
    trace_id: str | None,
    verification_warnings: list[str] | None = None,
    task_results: list[TaskExecutionResult | dict[str, Any]] | None = None,
    image_asset_ids: list[str] | None = None,
    evidence_ledger: list[EvidenceLedgerEntry | dict[str, Any]] | None = None,
) -> MessagePresentation:
    """Choose presentation deterministically; the model never controls this decision."""

    normalized_route: AgentRoute | None = None
    route_value: str | None = None
    if route is not None:
        route_value = str(route)
        try:
            normalized_route = AgentRoute(route_value)
        except ValueError:
            normalized_route = None

    parsed_answer = (
        answer
        if isinstance(answer, AgentAnswer)
        else AgentAnswer.model_validate(answer)
        if answer
        else None
    )
    structured = parsed_answer is not None and (
        normalized_route in STRUCTURED_CARD_ROUTES
        or normalized_route is None
    )
    parsed_evidence: list[EvidenceLedgerEntry] = []
    seen_evidence_ids: set[str] = set()
    if structured:
        for item in evidence_ledger or []:
            entry = EvidenceLedgerEntry.model_validate(item)
            if entry.evidence_id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(entry.evidence_id)
            parsed_evidence.append(entry)
    return MessagePresentation(
        mode=(
            MessageRenderMode.STRUCTURED_CARD
            if structured
            else MessageRenderMode.BUBBLE
        ),
        route_decision=route_value,
        status=status,
        answer=parsed_answer if structured else None,
        trace_id=trace_id if structured else None,
        verification_warnings=list(verification_warnings or []) if structured else [],
        task_results=[TaskExecutionResult.model_validate(item) for item in task_results or []],
        image_asset_ids=list(dict.fromkeys(image_asset_ids or [])) if structured else [],
        evidence_ledger=parsed_evidence,
    )
