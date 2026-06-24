"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        llm = LLMClient()

        sources_str = "\n".join(
            f"[{idx+1}] {doc.title} - {doc.url or 'No URL'}"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = "You are a Writer Agent. Your task is to write a detailed, well-structured final report answering the user's query, utilizing the research and analysis notes. Include clean inline citations like [1] or [2] referencing sources."
        user_prompt = f"""User Query: "{state.request.query}"
Target Audience: {state.request.audience}

Research Notes:
{state.research_notes or "None"}

Analysis Notes:
{state.analysis_notes or "None"}

Available Sources:
{sources_str}

Please generate the final answer. Ensure that:
- It is comprehensive and professionally structured.
- It includes clear sections with markdown headings.
- Sources are cited inline using bracket numbers matching the source list (e.g., [1]).
- A "References" section is included at the end listing the details of the sources.
"""

        response = llm.complete(system_prompt, user_prompt)
        state.final_answer = response.content.strip()

        state.add_trace_event("llm_call", {
            "agent": self.name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        })

        state.add_trace_event("writer_execution", {
            "final_answer_length": len(state.final_answer)
        })

        return state
