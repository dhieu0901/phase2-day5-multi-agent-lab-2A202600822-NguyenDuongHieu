"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import json
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()

        # Iteration guardrail: if iteration limit reached, stop or compile final answer
        if state.iteration >= settings.max_iterations:
            next_agent = "done" if state.final_answer else "writer"
            state.record_route(next_agent)
            state.add_trace_event("supervisor_decision", {
                "next": next_agent,
                "reasoning": f"Max iterations ({settings.max_iterations}) reached. Forcing completion."
            })
            return state

        # Build prompt listing state context
        system_prompt = f"""You are the Supervisor of a Multi-Agent Research System.
Your job is to coordinate a team of research agents to answer the user query: "{state.request.query}"

The agents available are:
1. researcher: Gathers facts, details, and sources. Call researcher when you need to gather new info, fetch external data, or expand the source base.
2. analyst: Synthesizes research notes, finds contradictions, analyzes key claims, and organizes insights. Call analyst after researcher has gathered notes.
3. writer: Produces the final response with markdown format and citations. Call writer only after analyst has completed their analysis.
4. critic: Fact-checks the writer's final answer, verifies citations, and identifies hallucinations. Call critic right after writer produces a final answer.
5. done: Stops the workflow. Call done when the final answer has been generated and validated.

You must respond in JSON format with two keys:
- "next": one of "researcher", "analyst", "writer", "critic", "done"
- "reasoning": a brief explanation of your decision.

Current workflow state:
- Iteration count: {state.iteration}
- Route history: {state.route_history}
- Sources gathered: {len(state.sources)}
- Research notes: {state.research_notes or "Empty"}
- Analysis notes: {state.analysis_notes or "Empty"}
- Final answer: {state.final_answer or "Empty"}
"""

        user_prompt = "Decide the next agent to invoke. Output JSON only."

        llm = LLMClient()
        try:
            response = llm.complete(system_prompt, user_prompt)
            # Record LLM call metrics
            state.add_trace_event("llm_call", {
                "agent": self.name,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })

            data = json.loads(response.content.strip())
            next_agent = data.get("next", "").strip().lower()
            reasoning = data.get("reasoning", "")

            valid_agents = ["researcher", "analyst", "writer", "critic", "done"]
            if next_agent not in valid_agents:
                raise ValueError(f"Invalid next agent name: {next_agent}")
        except Exception as e:
            # Fallback routing
            reasoning = f"Fallback routing activated due to parsing error: {e}"
            if not state.research_notes:
                next_agent = "researcher"
            elif not state.analysis_notes:
                next_agent = "analyst"
            elif not state.final_answer:
                next_agent = "writer"
            elif "critic" not in state.route_history:
                next_agent = "critic"
            else:
                next_agent = "done"

        # Update state history and trace
        state.record_route(next_agent)
        state.add_trace_event("supervisor_decision", {
            "next": next_agent,
            "reasoning": reasoning
        })

        return state
