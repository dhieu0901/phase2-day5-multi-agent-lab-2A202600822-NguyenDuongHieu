"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from langgraph.graph import StateGraph, END
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent,
)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> StateGraph[ResearchState]:
        """Create a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        def supervisor_node(state: ResearchState) -> ResearchState:
            agent = SupervisorAgent()
            return agent.run(state)

        def researcher_node(state: ResearchState) -> ResearchState:
            agent = ResearcherAgent()
            return agent.run(state)

        def analyst_node(state: ResearchState) -> ResearchState:
            agent = AnalystAgent()
            return agent.run(state)

        def writer_node(state: ResearchState) -> ResearchState:
            agent = WriterAgent()
            return agent.run(state)

        def critic_node(state: ResearchState) -> ResearchState:
            agent = CriticAgent()
            return agent.run(state)

        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("writer", writer_node)
        workflow.add_node("critic", critic_node)

        workflow.set_entry_point("supervisor")

        def router(state: ResearchState) -> str:
            if not state.route_history:
                return "done"
            next_agent = state.route_history[-1]
            if next_agent == "done":
                return END
            return next_agent

        workflow.add_conditional_edges(
            "supervisor",
            router,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END
            }
        )

        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        return workflow

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        graph = self.build().compile()
        result = graph.invoke(state)

        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        else:
            raise ValueError(f"Unexpected graph run result type: {type(result)}")
