"""CLI for the Day 08 LangGraph lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml
from langchain_core.runnables import RunnableConfig

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _dump_state(state: dict, scenario_id: str, output_dir: Path) -> None:
    """Persist the final state for one scenario as evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"state_{scenario_id}.json"
    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
    dump_states: Annotated[bool, typer.Option("--dump-states/--no-dump-states")] = True,
) -> None:
    """Execute all scenarios, write metrics JSON, and dump per-scenario state evidence."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    state_dir = output.parent / "states"
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}
        t0 = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        metric = metric_from_state(
            final_state, scenario.expected_route.value, scenario.requires_approval
        )
        metric.latency_ms = latency_ms
        metrics.append(metric)

        if dump_states:
            _dump_state(final_state, scenario.id, state_dir)
        typer.echo(
            f"[{scenario.id}] route={metric.actual_route} success={metric.success} "
            f"retries={metric.retry_count} interrupts={metric.interrupt_count} "
            f"latency_ms={latency_ms}"
        )

    report = summarize_metrics(metrics)
    if cfg.get("resume_success"):
        report.resume_success = True
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(
        f"\nWrote metrics to {output} (success_rate={report.success_rate:.2%}, "
        f"total_retries={report.total_retries}, total_interrupts={report.total_interrupts})"
    )


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON against the MetricsReport schema."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(
        f"Metrics valid. scenarios={report.total_scenarios} success_rate={report.success_rate:.2%} "
        f"avg_nodes={report.avg_nodes_visited:.2f} retries={report.total_retries} "
        f"interrupts={report.total_interrupts} resume_success={report.resume_success}"
    )


@app.command("export-graph")
def export_graph(
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/graph.mmd"),
) -> None:
    """Export the compiled graph as a Mermaid diagram for the report."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    mermaid = graph.get_graph().draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(mermaid, encoding="utf-8")
    typer.echo(f"Wrote Mermaid diagram to {output} ({len(mermaid)} chars)")


if __name__ == "__main__":
    app()
