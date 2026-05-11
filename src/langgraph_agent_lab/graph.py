"""Graph construction.

Import-safe: LangGraph is imported only inside :func:`build_graph` so unit tests covering schema
and metrics still run when langgraph is unavailable.

Architecture:

    START -> intake -> classify -> {answer | tool | clarify | risky_action | retry}
    tool -> evaluate -> {answer | retry}                      (retry loop, bounded)
    risky_action -> approval -> {tool | clarify}              (HITL gate)
    retry -> {tool | dead_letter}                             (bounded by max_attempts)
    {answer | clarify | dead_letter} -> finalize -> END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from .routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Explicit path maps so `graph.get_graph().draw_mermaid()` renders every conditional edge.
_CLASSIFY_DESTINATIONS = ["answer", "tool", "clarify", "risky_action", "retry"]
_EVALUATE_DESTINATIONS = ["answer", "retry"]
_APPROVAL_DESTINATIONS = ["tool", "clarify"]
_RETRY_DESTINATIONS = ["tool", "dead_letter"]


def build_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - helpful install error
        raise RuntimeError(
            "LangGraph is required. Run: pip install -e '.[dev]' or pip install langgraph"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("classify", classify_node)
    graph.add_node("answer", answer_node)
    graph.add_node("tool", tool_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("clarify", ask_clarification_node)
    graph.add_node("risky_action", risky_action_node)
    graph.add_node("approval", approval_node)
    graph.add_node("retry", retry_or_fallback_node)
    graph.add_node("dead_letter", dead_letter_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")
    graph.add_conditional_edges("classify", route_after_classify, _CLASSIFY_DESTINATIONS)
    graph.add_edge("tool", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, _EVALUATE_DESTINATIONS)
    graph.add_edge("clarify", "finalize")
    graph.add_edge("risky_action", "approval")
    graph.add_conditional_edges("approval", route_after_approval, _APPROVAL_DESTINATIONS)
    graph.add_conditional_edges("retry", route_after_retry, _RETRY_DESTINATIONS)
    graph.add_edge("answer", "finalize")
    graph.add_edge("dead_letter", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
