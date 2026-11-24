"""Global LangGraph state and protocol logging for the Sprints orchestrator."""
from __future__ import annotations

import logging
from typing import Annotated, Any, Iterable, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.logic import format_protocol_message

PROTOCOL_LOGGER = logging.getLogger("sprints.protocol")
FINAL_SENDER = "sprints_assistant"


class SprintsState(TypedDict, total=False):
    """Global state shared by the orchestrator and both worker agents."""
    messages: Annotated[list[AnyMessage], add_messages]
    learner_id: str
    intent: str
    search_query: str
    retrieved_context: list[dict[str, Any]]
    context_sufficient: bool
    rag_attempts: int
    retrieval_latency_ms: float
    learner_profile: dict[str, Any]
    placement_recommendation: dict[str, Any]
    code_submission: str
    repl_result: dict[str, Any]
    assessment: dict[str, Any]
    judge_verdict: dict[str, Any]
    refinement_iterations: int
    human_decision: dict[str, Any]
    draft_answer: str
    final_answer: str


def fresh_turn_defaults() -> dict[str, Any]:
    """Return values reset for each new turn while conversation state persists."""
    return {
        "intent": "", "search_query": "", "retrieved_context": [], "context_sufficient": False,
        "rag_attempts": 0, "retrieval_latency_ms": 0.0, "learner_profile": {},
        "placement_recommendation": {}, "code_submission": "", "repl_result": {},
        "assessment": {}, "judge_verdict": {}, "refinement_iterations": 0,
        "human_decision": {}, "draft_answer": "", "final_answer": "",
    }


def log_messages(node: str, messages: Iterable[Any]) -> None:
    """Write one verbose log line per protocol message produced by a node."""
    for msg in messages:
        PROTOCOL_LOGGER.info("%-22s %s", node, format_protocol_message(msg))


def latest_user_text(state: SprintsState) -> str:
    """Return the most recent HumanMessage text."""
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


def recent_history(state: SprintsState, limit: int = 6) -> str:
    """Return a compact transcript of recent learner and assistant turns."""
    lines = []
    for msg in state.get("messages", []):
        kind = getattr(msg, "type", "")
        if kind == "human":
            lines.append(f"Learner: {msg.content}")
        elif kind == "ai" and getattr(msg, "name", None) == FINAL_SENDER:
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines[-limit:]) or "(no earlier turns)"
