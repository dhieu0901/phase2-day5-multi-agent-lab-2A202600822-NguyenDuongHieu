from unittest.mock import patch
from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def test_supervisor_routes_to_researcher_initially() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    
    with patch("multi_agent_research_lab.services.llm_client.LLMClient.complete") as mock_complete:
        mock_complete.return_value = LLMResponse(
            content='{"next": "researcher", "reasoning": "Need to research multi-agent systems."}',
            input_tokens=10,
            output_tokens=10,
            cost_usd=0.0001
        )
        
        result = SupervisorAgent().run(state)
        assert result.iteration == 1
        assert result.route_history == ["researcher"]
        assert len(result.trace) == 2  # llm_call and supervisor_decision
        assert result.trace[1]["payload"]["next"] == "researcher"
