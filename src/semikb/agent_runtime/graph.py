"""Small LangGraph state machine for the controlled conversation flow.

The in-memory demo checkpointer is intentional: a production deployment replaces it
with the MongoDB checkpointer adapter while preserving the same ``thread_id`` key.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ConversationState(TypedDict, total=False):
    content: str
    missing_fields: list[str]
    action: Literal["clarify", "retrieve"]


class ConversationGraph:
    """Routes a message before any downstream retrieval or tool invocation."""

    def __init__(self) -> None:
        workflow = StateGraph(ConversationState)
        workflow.add_node("inspect", self._inspect)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_edge(START, "inspect")
        workflow.add_conditional_edges(
            "inspect",
            self._route,
            {"clarify": "clarify", "retrieve": "retrieve"},
        )
        workflow.add_edge("clarify", END)
        workflow.add_edge("retrieve", END)
        self._graph = workflow.compile(checkpointer=InMemorySaver())

    def decide(self, thread_id: str, content: str, missing_fields: list[str]) -> ConversationState:
        """Persist the branch decision under the same graph thread identifier."""

        return self._graph.invoke(
            {"content": content, "missing_fields": missing_fields},
            config={"configurable": {"thread_id": thread_id}},
        )

    @staticmethod
    def _inspect(state: ConversationState) -> ConversationState:
        return {"action": "clarify" if state.get("missing_fields") else "retrieve"}

    @staticmethod
    def _route(state: ConversationState) -> Literal["clarify", "retrieve"]:
        return state["action"]

    @staticmethod
    def _clarify(_: ConversationState) -> ConversationState:
        return {}

    @staticmethod
    def _retrieve(_: ConversationState) -> ConversationState:
        return {}
