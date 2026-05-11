"""Real LangGraph human-in-the-loop interrupt demo.

When ``LANGGRAPH_INTERRUPT=true``, the approval node calls :func:`langgraph.types.interrupt`,
which suspends the graph mid-run. A second :func:`graph.invoke` call with a
:class:`langgraph.types.Command` resumes the same thread with the approver's decision.

This script automates both halves so the evidence is reproducible without a UI.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["LANGGRAPH_INTERRUPT"] = "true"

from langgraph.types import Command  # noqa: E402

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.state import Route, Scenario, initial_state  # noqa: E402

DB_PATH = "outputs/hitl_checkpoints.db"
THREAD_ID = "thread-hitl_demo"
SCENARIO = Scenario(
    id="hitl_demo",
    query="Refund this customer and send confirmation email",
    expected_route=Route.RISKY,
    requires_approval=True,
)


def _config() -> dict:
    return {"configurable": {"thread_id": THREAD_ID}}


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run() -> None:
    Path("outputs").mkdir(exist_ok=True)
    # Start each run from a clean DB so the demo is deterministic.
    for suffix in ("", "-wal", "-shm"):
        p = Path(DB_PATH + suffix)
        if p.exists():
            p.unlink()

    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    state = initial_state(SCENARIO)

    _print_header("PHASE 1 - INVOKE: expect graph to suspend at approval node")
    first = graph.invoke(state, config=_config())
    snapshot = graph.get_state(_config())
    print(f"after first invoke -- next nodes : {snapshot.next}")
    print(f"after first invoke -- final_answer present? {'final_answer' in first}")
    interrupts = (snapshot.tasks or [])
    for task in interrupts:
        for itr in (getattr(task, "interrupts", None) or []):
            print(f"  interrupt payload: {itr.value}")
    print(f"events captured before approval: {len(first.get('events', []))}")

    _print_header("PHASE 2 - RESUME with approver decision via Command(resume=...)")
    decision = {"approved": True, "reviewer": "hoang-hiep", "comment": "verified refund eligibility"}
    print(f"sending approval decision: {decision}")
    second = graph.invoke(Command(resume=decision), config=_config())

    print(f"after resume -- route       : {second.get('route')}")
    print(f"after resume -- approval    : {second.get('approval')}")
    print(f"after resume -- final_answer: {second.get('final_answer')}")
    print(f"after resume -- events count: {len(second.get('events', []))}")

    history = list(graph.get_state_history(_config()))
    print(f"\ntotal checkpoints in DB     : {len(history)}")
    approval_events = [
        evt
        for evt in second.get("events", [])
        if isinstance(evt, dict) and evt.get("node") == "approval"
    ]
    print(f"approval events in history  : {len(approval_events)}")
    assert second.get("approval", {}).get("approved") is True, "approval was not persisted"
    assert second.get("final_answer"), "final_answer must be set after resume"
    print("\nReal HITL flow OK: interrupt suspended the run, Command(resume=...) resumed it.")


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
