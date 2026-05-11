"""Parallel fan-out demonstration using :class:`langgraph.types.Send`.

The lab's main graph uses sequential tool calls. This script demonstrates the alternative
pattern: dispatch two tool calls concurrently from a single dispatcher and merge their
results through an ``Annotated[list, add]`` reducer.

Architecture::

    START -> plan -> [dispatch_send] --> worker  (one task per Send payload, run concurrently)
                                       \\
                                        --> worker
                                                  \\-> merge -> END
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Send  # noqa: E402


class FanoutState(TypedDict, total=False):
    query: str
    tools: list[str]
    tool_results: Annotated[list[str], add]  # append-only via reducer
    events: Annotated[list[str], add]
    final_answer: str


def plan_node(state: FanoutState) -> dict:
    tools = ["order_lookup", "shipment_tracker"]
    return {
        "tools": tools,
        "events": [f"plan: dispatching {len(tools)} tools in parallel"],
    }


def dispatch_send(state: FanoutState) -> list[Send]:
    """Return one Send per tool so LangGraph schedules them concurrently."""
    return [
        Send("worker", {"query": state["query"], "tool": tool})
        for tool in state.get("tools", [])
    ]


def worker_node(payload: dict) -> dict:
    """Each worker runs as its own task. Sleeps briefly to make parallelism observable."""
    tool = payload["tool"]
    query = payload["query"]
    started = time.perf_counter()
    time.sleep(0.2)  # simulate I/O latency
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "tool_results": [f"{tool}: result for '{query}' (sim_latency_ms={elapsed_ms})"],
        "events": [f"worker[{tool}] finished in ~{elapsed_ms}ms"],
    }


def merge_node(state: FanoutState) -> dict:
    results = state.get("tool_results", []) or []
    summary = " | ".join(results)
    return {
        "final_answer": f"Merged from {len(results)} tools: {summary}",
        "events": [f"merge: combined {len(results)} tool results"],
    }


def build_fanout_graph():
    graph = StateGraph(FanoutState)
    graph.add_node("plan", plan_node)
    graph.add_node("worker", worker_node)
    graph.add_node("merge", merge_node)

    graph.add_edge(START, "plan")
    # The dispatcher returns Send objects; LangGraph fans out one parallel task per Send.
    graph.add_conditional_edges("plan", dispatch_send, ["worker"])
    graph.add_edge("worker", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


def run() -> None:
    graph = build_fanout_graph()

    # Export Mermaid for the report.
    diagram_path = Path("outputs/graph_fanout.mmd")
    diagram_path.parent.mkdir(parents=True, exist_ok=True)
    diagram_path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")

    print(f"Wrote fan-out diagram: {diagram_path}")

    started = time.perf_counter()
    result = graph.invoke({"query": "order 12345 status and shipment"})
    wall_ms = int((time.perf_counter() - started) * 1000)

    print(f"\nfan-out wall-clock: {wall_ms} ms "
          f"(sequential would be ~400ms; <400ms proves true parallelism)")
    print(f"tool_results count: {len(result.get('tool_results', []))}")
    for entry in result.get("tool_results", []):
        print(f"  - {entry}")
    print(f"\nfinal_answer: {result.get('final_answer')}")
    print(f"events:")
    for evt in result.get("events", []):
        print(f"  * {evt}")

    assert wall_ms < 400, (
        f"Expected wall-clock < 400ms for true parallel execution, got {wall_ms}ms"
    )
    assert len(result.get("tool_results", [])) == 2, "Expected 2 merged tool results"
    print("\nFan-out OK: workers ran concurrently and merged via add reducer.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    log_handle = None
    original_stdout = sys.stdout
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log.open("w", encoding="utf-8")

        class _Tee(io.TextIOBase):
            def write(self, data: str) -> int:  # type: ignore[override]
                original_stdout.write(data)
                log_handle.write(data)
                return len(data)

            def flush(self) -> None:  # type: ignore[override]
                original_stdout.flush()
                log_handle.flush()

        sys.stdout = _Tee()

    try:
        run()
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
