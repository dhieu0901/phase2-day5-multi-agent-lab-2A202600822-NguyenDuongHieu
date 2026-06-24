"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        llm = LLMClient()
        search_client = SearchClient()

        # Step 1: Generate search query using LLM
        query_gen_sys = "You are a Researcher Agent. Generate a single, concise search query to look up details for the user query."
        query_gen_user = f"""User Query: "{state.request.query}"
Current Research Notes: {state.research_notes or "None"}
Current Sources: {[s.title for s in state.sources]}

Generate a search query. Respond with ONLY the query text."""

        response_q = llm.complete(query_gen_sys, query_gen_user)
        search_query = response_q.content.strip().strip('"').strip("'")
        
        state.add_trace_event("llm_call", {
            "agent": self.name,
            "purpose": "query_generation",
            "input_tokens": response_q.input_tokens,
            "output_tokens": response_q.output_tokens,
            "cost_usd": response_q.cost_usd,
        })

        # Step 2: Execute search
        results = search_client.search(search_query, max_results=state.request.max_sources)
        
        # Step 3: Deduplicate and record sources
        existing_urls = {s.url for s in state.sources if s.url}
        existing_titles = {s.title.lower() for s in state.sources}
        new_sources_added = 0
        for doc in results:
            if doc.url and doc.url in existing_urls:
                continue
            if doc.title.lower() in existing_titles:
                continue
            state.sources.append(doc)
            if doc.url:
                existing_urls.add(doc.url)
            existing_titles.add(doc.title.lower())
            new_sources_added += 1

        # Step 4: Synthesize updated research notes
        sources_str = "\n\n".join(
            f"Source [{idx+1}]: {doc.title}\nURL: {doc.url}\nContent: {doc.snippet}"
            for idx, doc in enumerate(state.sources)
        )

        synthesis_sys = "You are a Researcher Agent. Synthesize structured, detailed research notes from the provided sources. Cite sources using [Source Name] or [Idx]."
        synthesis_user = f"""User Query: "{state.request.query}"
Existing Research Notes: {state.research_notes or "None"}

Sources gathered:
{sources_str}

Please generate updated, comprehensive research notes based on all available sources."""

        response_s = llm.complete(synthesis_sys, synthesis_user)
        state.research_notes = response_s.content.strip()

        state.add_trace_event("llm_call", {
            "agent": self.name,
            "purpose": "notes_synthesis",
            "input_tokens": response_s.input_tokens,
            "output_tokens": response_s.output_tokens,
            "cost_usd": response_s.cost_usd,
        })

        state.add_trace_event("researcher_execution", {
            "search_query": search_query,
            "new_sources_added": new_sources_added,
            "total_sources": len(state.sources)
        })

        return state
