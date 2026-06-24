import yaml  # type: ignore
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_single_agent_baseline(query: str) -> ResearchState:
    llm = LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    sys_prompt = "You are a helpful research assistant. Answer the user's research query directly and comprehensively."
    user_prompt = f"Query: {query}"
    
    response = llm.complete(sys_prompt, user_prompt)
    state.final_answer = response.content.strip()
    
    state.add_trace_event("llm_call", {
        "agent": "single_agent_baseline",
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    })
    
    return state


def _run_multi_agent_workflow(query: str) -> ResearchState:
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline call."""
    _init()
    console.print(f"[bold blue]Running Single-Agent Baseline for query:[/bold blue] '{query}'")
    
    state, metrics = run_benchmark("baseline", query, _run_single_agent_baseline)
    
    console.print(Panel.fit(state.final_answer or "No output", title="Baseline Answer"))
    console.print(f"[bold green]Latency:[/bold green] {metrics.latency_seconds:.2f}s | [bold green]Cost:[/bold green] ${metrics.estimated_cost_usd or 0.0:.6f} | [bold green]Score:[/bold green] {metrics.quality_score or 'N/A'}")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print(f"[bold blue]Running Multi-Agent Workflow for query:[/bold blue] '{query}'")
    
    state, metrics = run_benchmark("multi_agent", query, _run_multi_agent_workflow)
    
    console.print(Panel.fit(state.final_answer or "No output", title="Multi-Agent Final Answer"))
    console.print(f"[bold green]Latency:[/bold green] {metrics.latency_seconds:.2f}s | [bold green]Cost:[/bold green] ${metrics.estimated_cost_usd or 0.0:.6f} | [bold green]Score:[/bold green] {metrics.quality_score or 'N/A'}")
    console.print(f"[bold yellow]Route History:[/bold yellow] {' -> '.join(state.route_history)}")


@app.command("benchmark")
def benchmark() -> None:
    """Run single-agent and multi-agent workflows on benchmark queries and generate report."""
    _init()
    
    config_path = Path("configs/lab_default.yaml")
    queries = []
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                queries = config_data.get("benchmark", {}).get("queries", [])
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            
    if not queries:
        queries = [
            "Research GraphRAG state-of-the-art and write a 500-word summary",
            "Compare single-agent and multi-agent workflows for customer support",
            "Summarize production guardrails for LLM agents"
        ]
        
    all_metrics = []
    
    console.print(Panel("Starting Benchmark Runner (Single-Agent vs Multi-Agent)", style="bold magenta"))
    
    for idx, query in enumerate(queries):
        console.print(f"\n[bold cyan]Query #{idx+1}:[/bold cyan] '{query}'")
        
        console.print("  [dim]Running Baseline...[/dim]")
        _, base_metrics = run_benchmark(f"Baseline - Query {idx+1}", query, _run_single_agent_baseline)
        all_metrics.append(base_metrics)
        
        console.print("  [dim]Running Multi-Agent...[/dim]")
        _, multi_metrics = run_benchmark(f"Multi-Agent - Query {idx+1}", query, _run_multi_agent_workflow)
        all_metrics.append(multi_metrics)
        
    report_content = render_markdown_report(all_metrics)
    store = LocalArtifactStore()
    report_file = store.write_text("benchmark_report.md", report_content)
    
    console.print(f"\n[bold green][SUCCESS] Benchmark complete! Report saved to {report_file}[/bold green]")
    
    table = Table(title="Benchmark Comparison")
    table.add_column("Run Name", style="cyan")
    table.add_column("Latency (s)", justify="right", style="green")
    table.add_column("Cost (USD)", justify="right", style="yellow")
    table.add_column("Quality Score", justify="right", style="magenta")
    table.add_column("Notes", style="white")
    
    for m in all_metrics:
        cost = f"${m.estimated_cost_usd:.5f}" if m.estimated_cost_usd is not None else "N/A"
        quality = f"{m.quality_score:.1f}" if m.quality_score is not None else "N/A"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}s", cost, quality, m.notes)
        
    console.print(table)


if __name__ == "__main__":
    app()
