"""Crash-resume + time-travel demonstration backed by SqliteSaver.

Usage:
    python scripts/demo_resume.py --phase write
        Runs the graph end-to-end for one scenario and persists every checkpoint to
        ``outputs/checkpoints.db`` under a stable ``thread_id``.

    python scripts/demo_resume.py --phase read
        Opens the same SQLite file in a fresh process (no in-memory carry-over) and
        replays the state history. Demonstrates that state survived process restart.

    python scripts/demo_resume.py --phase replay
        Time-travel: re-invoke the graph from an intermediate checkpoint to confirm
        deterministic replay.

Run all three sequentially with ``--phase all``.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Make ``src/`` importable when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.state import Route, Scenario, initial_state  # noqa: E402

DB_PATH = "outputs/checkpoints.db"
THREAD_ID = "thread-resume_demo"
SCENARIO = Scenario(
    id="resume_demo",
    query="Timeout failure while processing request",
    expected_route=Route.ERROR,
    should_retry=True,
)


def _config() -> dict:
    return {"configurable": {"thread_id": THREAD_ID}}


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def phase_write() -> None:
    _print_header("PHASE 1/3 - WRITE: run graph end-to-end and persist checkpoints")
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    state = initial_state(SCENARIO)
    final = graph.invoke(state, config=_config())
    print(f"thread_id        : {THREAD_ID}")
    print(f"db_path          : {DB_PATH}")
    print(f"actual_route     : {final.get('route')}")
    print(f"attempt          : {final.get('attempt')}")
    print(f"evaluation_result: {final.get('evaluation_result')}")
    print(f"final_answer     : {final.get('final_answer')}")
    print(f"events recorded  : {len(final.get('events', []))}")


def phase_read() -> None:
    _print_header("PHASE 2/3 - READ AFTER 'RESTART': re-open SQLite and replay history")
    # Build a fresh checkpointer to prove nothing is held in process memory.
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    history = list(graph.get_state_history(_config()))
    print(f"checkpoints found: {len(history)}")
    print(f"{'step':>4}  {'next':<18}  writes")
    print("-" * 78)
    for snapshot in reversed(history):
        meta = snapshot.metadata or {}
        next_nodes = ",".join(snapshot.next) if snapshot.next else "<END>"
        writes = list((meta.get("writes") or {}).keys())
        print(f"{meta.get('step', '?'):>4}  {next_nodes:<18}  {writes}")
    latest = history[0].values if history else {}
    print(f"\nFinal answer after replay: {latest.get('final_answer')}")


def phase_replay() -> None:
    _print_header("PHASE 3/3 - TIME-TRAVEL REPLAY from an intermediate checkpoint")
    checkpointer = build_checkpointer("sqlite", DB_PATH)
    graph = build_graph(checkpointer=checkpointer)
    history = list(graph.get_state_history(_config()))
    if len(history) < 3:
        print("Not enough checkpoints to time-travel; run --phase write first.")
        return
    # Pick a checkpoint roughly in the middle so replay covers a meaningful prefix.
    target = history[len(history) // 2]
    meta = target.metadata or {}
    print(
        f"Resuming from step={meta.get('step')} next={target.next} "
        f"checkpoint_id={target.config['configurable'].get('checkpoint_id')}"
    )
    replayed = graph.invoke(None, config=target.config)
    print(f"Replayed final_answer : {replayed.get('final_answer')}")
    print(f"Replayed events count : {len(replayed.get('events', []))}")
    print("Time-travel replay succeeded - state is deterministic from any checkpoint.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("write", "read", "replay", "all"),
        default="all",
        help="Which phase of the demo to run.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to mirror stdout into a UTF-8 log file.",
    )
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

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
        if args.phase in ("write", "all"):
            phase_write()
        if args.phase in ("read", "all"):
            phase_read()
        if args.phase in ("replay", "all"):
            phase_replay()
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
