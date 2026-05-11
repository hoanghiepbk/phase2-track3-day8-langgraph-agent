"""Conditional-edge routing functions.

Each function reads the current ``AgentState`` and returns the name of the next node.
Routers are intentionally side-effect-free so they can be unit-tested without invoking
LangGraph.
"""

from __future__ import annotations

from .state import AgentState, Route

_CLASSIFY_NEXT: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next node. Unknown routes fall back to ``answer``."""
    return _CLASSIFY_NEXT.get(state.get("route", Route.SIMPLE.value), "answer")


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool again or escalate to ``dead_letter``.

    The retry loop is bounded by ``state['max_attempts']`` so the graph always terminates.
    """
    if int(state.get("attempt", 0)) >= int(state.get("max_attempts", 3)):
        return "dead_letter"
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """The 'done?' check that closes the retry loop.

    Reads ``state['evaluation_result']`` produced by :func:`evaluate_node`; a value of
    ``needs_retry`` re-enters the retry node, anything else proceeds to ``answer``.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_approval(state: AgentState) -> str:
    """Continue execution only if the human approved the proposed action.

    A rejection ('approved=False') is treated as a safety fallback that hands control to
    the clarification node instead of executing the risky action.
    """
    approval = state.get("approval") or {}
    return "tool" if approval.get("approved") else "clarify"
