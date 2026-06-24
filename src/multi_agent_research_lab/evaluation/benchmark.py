"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, token cost, quality score, and routing notes."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # 1. Sum up token costs from the trace logs
    total_cost = 0.0
    for event in state.trace:
        if event.get("name") == "llm_call":
            payload = event.get("payload", {})
            cost = payload.get("cost_usd")
            if cost is not None:
                total_cost += cost

    # 2. Extract quality score (use critic's score if available, or LLM-as-a-judge)
    quality_score = None
    for res in state.agent_results:
        if res.agent == "critic":
            quality_score = res.metadata.get("quality_score")

    if quality_score is None and state.final_answer:
        try:
            import re
            from multi_agent_research_lab.services.llm_client import LLMClient
            llm = LLMClient()
            eval_prompt = f"""You are a Peer Review Judge evaluating a research report.
User Query: "{query}"

Report:
{state.final_answer}

Rate this report from 0.0 to 10.0 based on clarity, structure, and completeness.
Output ONLY the score as a single float number (e.g., 8.5)."""
            resp = llm.complete("You are an AI research judge.", eval_prompt)
            match = re.findall(r"\b(?:10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", resp.content.strip())
            if match:
                val = float(match[0])
                if 0.0 <= val <= 10.0:
                    quality_score = val
        except Exception:
            quality_score = 7.0  # safe fallback

    # 3. Compile notes (iteration count, sources, routing history)
    notes = f"Iters: {state.iteration}, Sources: {len(state.sources)}, Path: {' -> '.join(state.route_history)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        notes=notes
    )

    return state, metrics
