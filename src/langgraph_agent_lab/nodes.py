"""Node implementations for the LangGraph workflow.

Each node returns a partial state update (never mutates input state).
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event

# Word-stem patterns with explicit \b leading boundary so "delete" does not match "deletion"
# substrings inside other tokens, while still catching common inflections (deleted, sending, ...).
_RISKY_PATTERN = re.compile(
    r"\b(?:refund|delete|send|cancel|remove|revoke|wipe|purge|destroy|terminate|erase)\w*",
    re.IGNORECASE,
)
_TOOL_PATTERN = re.compile(
    r"\b(?:status|order|lookup|check|track|find|search|query|locate|retrieve|fetch)\w*",
    re.IGNORECASE,
)
_ERROR_PATTERN = re.compile(
    r"\b(?:timeout|fail|error|crash|unavailable|exception|outage|broken)\w*",
    re.IGNORECASE,
)
_PRONOUN_PATTERN = re.compile(r"\b(?:it|this|that|them|those|something)\b", re.IGNORECASE)
_MISSING_INFO_MAX_TOKENS = 5


def _tokenize(query: str) -> list[str]:
    """Lowercase, punctuation-stripped tokens used for length heuristics."""
    return re.findall(r"[a-z0-9]+", query.lower())


def intake_node(state: AgentState) -> dict:
    """Normalize the raw query: trim whitespace and collapse runs of spaces."""
    raw = state.get("query", "")
    normalized = re.sub(r"\s+", " ", raw).strip()
    return {
        "query": normalized,
        "messages": [f"intake:{normalized[:40]}"],
        "events": [make_event("intake", "completed", "query normalized", length=len(normalized))],
    }


def classify_node(state: AgentState) -> dict:
    """Route the query using word-boundary keyword matching.

    Priority (highest first): risky > tool > missing_info > error > simple.
    Word boundaries prevent substring false-positives ("it" inside "item").
    """
    query = state.get("query", "")
    tokens = _tokenize(query)

    if _RISKY_PATTERN.search(query):
        route, risk_level = Route.RISKY, "high"
    elif _TOOL_PATTERN.search(query):
        route, risk_level = Route.TOOL, "low"
    elif len(tokens) < _MISSING_INFO_MAX_TOKENS and _PRONOUN_PATTERN.search(query):
        route, risk_level = Route.MISSING_INFO, "low"
    elif _ERROR_PATTERN.search(query):
        route, risk_level = Route.ERROR, "medium"
    else:
        route, risk_level = Route.SIMPLE, "low"

    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                token_count=len(tokens),
                risk_level=risk_level,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask a targeted clarification question instead of guessing."""
    query = state.get("query", "")
    if "order" in query.lower():
        question = "Which order id is affected? Please share the order number."
    else:
        question = (
            "Your request is missing context. Please specify the entity (order/account/ticket id) "
            "and the action you want us to take."
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event("clarify", "completed", "missing information requested", question=question)
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock backend tool.

    For error-route scenarios, the first two attempts return a transient failure so the graph
    must traverse the retry loop. After that the tool returns a success payload, allowing the
    evaluate node to advance to answer.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario_id}"
        event_type = "transient_failure"
    else:
        result = f"OK: lookup_result scenario={scenario_id} attempt={attempt}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [
            make_event("tool", event_type, f"tool executed attempt={attempt}", attempt=attempt)
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Stage a risky action together with evidence for the approver."""
    query = state.get("query", "")
    proposed = (
        f"Proposed action: '{query[:80]}'. Risk={state.get('risk_level', 'high')}. "
        "Awaiting human approval before execution."
    )
    return {
        "proposed_action": proposed,
        "events": [
            make_event(
                "risky_action",
                "pending_approval",
                "approval required",
                risk_level=state.get("risk_level", "high"),
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval gate.

    When ``LANGGRAPH_INTERRUPT=true`` the node calls :func:`langgraph.types.interrupt`,
    suspending the graph until a :class:`langgraph.types.Command` resumes it with the
    reviewer's :class:`ApprovalDecision` (supports both ``approved=True`` and ``False``;
    a rejection falls through to ``route_after_approval`` and is routed to ``clarify``).

    Without the env var (default in tests/CI) a mock approval is returned so the graph
    can be exercised offline without a UI.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "risk_level": state.get("risk_level"),
            }
        )
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt. Bounded by max_attempts in route_after_retry."""
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    return {
        "attempt": attempt,
        "errors": [f"transient failure attempt={attempt}/{max_attempts}"],
        "events": [
            make_event(
                "retry",
                "completed",
                f"retry attempt {attempt}/{max_attempts}",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool_results and approval, when available."""
    tool_results = state.get("tool_results") or []
    approval = state.get("approval") or {}
    parts = []
    if approval.get("approved"):
        parts.append(f"[approved by {approval.get('reviewer', 'reviewer')}]")
    if tool_results:
        parts.append(f"Tool says: {tool_results[-1]}")
    else:
        parts.append("Direct response: your request can be handled without a tool call.")
    answer = " ".join(parts)
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """The 'done?' check that gates the retry loop.

    Marks the latest tool result as either ``needs_retry`` (contains ERROR prefix) or
    ``success``. The routing function `route_after_evaluate` reads this field.
    """
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""
    needs_retry = latest.startswith("ERROR")
    return {
        "evaluation_result": "needs_retry" if needs_retry else "success",
        "events": [
            make_event(
                "evaluate",
                "completed",
                "needs_retry" if needs_retry else "success",
                latest_result=latest[:80],
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Terminal node for unresolvable failures after max_attempts retries."""
    attempt = int(state.get("attempt", 0))
    return {
        "final_answer": (
            f"Request could not be completed after {attempt} attempts. "
            "Escalated to manual review (dead-letter queue)."
        ),
        "errors": [f"dead_letter attempts={attempt}"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded attempt={attempt}",
                attempt=attempt,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit the closing audit event so the run-trace is bracketed by intake -> finalize."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
