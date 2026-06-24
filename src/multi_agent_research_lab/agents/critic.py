"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentResult, AgentName
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        llm = LLMClient()

        sources_str = "\n".join(
            f"[{idx+1}] {doc.title} - {doc.url or 'No URL'}\nContent: {doc.snippet}"
            for idx, doc in enumerate(state.sources)
        )

        system_prompt = "You are a Critic Agent. Your task is to fact-check the writer's final answer, verify citation coverage, check for hallucinations, and rate quality."
        user_prompt = f"""User Query: "{state.request.query}"

Final Answer under review:
{state.final_answer or "No final answer generated yet."}

Sources Reference:
{sources_str}

Please perform a rigorous review. Highlight:
1. Factual accuracy & source alignment.
2. Citation coverage (are all key claims properly referenced?).
3. Any detected hallucinations or unsupported claims.
4. Constructive feedback for improvement.
5. Final quality rating on a scale of 0.0 to 10.0.
"""

        response = llm.complete(system_prompt, user_prompt)
        critic_notes = response.content.strip()

        # Try to parse or extract a score
        score = 8.5
        for line in critic_notes.split("\n"):
            if "score" in line.lower() or "rating" in line.lower():
                import re
                matches = re.findall(r"\b(?:10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", line)
                if matches:
                    try:
                        val = float(matches[0])
                        if 0.0 <= val <= 10.0:
                            score = val
                            break
                    except ValueError:
                        pass

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critic_notes,
                metadata={"quality_score": score}
            )
        )

        state.add_trace_event("llm_call", {
            "agent": self.name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        })

        state.add_trace_event("critic_execution", {
            "critic_feedback_length": len(critic_notes),
            "quality_score": score
        })

        return state
