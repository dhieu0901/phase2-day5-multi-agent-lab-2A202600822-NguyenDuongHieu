"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        llm = LLMClient()

        sources_str = "\n".join(
            f"- [{idx+1}] {doc.title} ({doc.url or 'No URL'})"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = "You are an Analyst Agent. Your task is to analyze research notes and sources to extract key claims, identify contradictions or gaps, and structure insights."
        user_prompt = f"""User Query: "{state.request.query}"

Research Notes:
{state.research_notes or "None"}

Sources gathered:
{sources_str}

Please perform a structured critical analysis. Highlight:
1. Core Claims & Findings
2. Quality and Strength of Evidence
3. Gaps, Uncertainties, or Contrasting Viewpoints
4. Key Takeaways formatted for the target audience: {state.request.audience}
"""

        response = llm.complete(system_prompt, user_prompt)
        state.analysis_notes = response.content.strip()

        state.add_trace_event("llm_call", {
            "agent": self.name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        })

        state.add_trace_event("analyst_execution", {
            "analysis_length": len(state.analysis_notes)
        })

        return state
