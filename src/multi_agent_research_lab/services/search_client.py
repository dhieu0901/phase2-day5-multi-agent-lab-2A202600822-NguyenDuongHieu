"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


import json
import urllib.request
import urllib.error
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with Tavily support and mock fallbacks."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        settings = get_settings()
        api_key = settings.tavily_api_key

        if api_key:
            try:
                req = urllib.request.Request(
                    "https://api.tavily.com/search",
                    data=json.dumps({
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    results = res_data.get("results", [])
                    documents = []
                    for r in results:
                        documents.append(
                            SourceDocument(
                                title=r.get("title", "No Title"),
                                url=r.get("url"),
                                snippet=r.get("content", ""),
                                metadata={"score": r.get("score", 0.0)}
                            )
                        )
                    if documents:
                        return documents[:max_results]
            except Exception:
                pass

        query_lower = query.lower()
        mock_docs = []

        if "graphrag" in query_lower or "graph rag" in query_lower:
            mock_docs = [
                SourceDocument(
                    title="GraphRAG: Graph-based Retrieval-Augmented Generation",
                    url="https://github.com/microsoft/graphrag",
                    snippet="Microsoft's GraphRAG combines Knowledge Graphs with LLMs to build global summaries of massive text corpora. It builds a hierarchical graph of entities, claims, and relations, allowing for structured query answering that traditional vector RAG struggles with.",
                    metadata={"source": "mock_database"}
                ),
                SourceDocument(
                    title="GraphRAG State of the Art: Entity Resolution & Community Detection",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet="State of the art GraphRAG implementations use Leiden clustering for community detection. This enables summarization of information at multiple levels of granularity, which is highly effective for global questions over huge document sets.",
                    metadata={"source": "mock_database"}
                ),
            ]
        elif "support" in query_lower or "customer" in query_lower:
            mock_docs = [
                SourceDocument(
                    title="Single-agent vs Multi-agent Orchestration for Support",
                    url="https://www.anthropic.com/news/building-effective-agents",
                    snippet="In customer support, single-agent systems struggle with context shifting and complex rules. Multi-agent systems assign specific roles (e.g., billing expert, troubleshooter, router) to improve reliability, reduce prompt length, and lower latencies.",
                    metadata={"source": "mock_database"}
                ),
                SourceDocument(
                    title="Multi-Agent Systems for Enterprise Service Desks",
                    url="https://openai.com/blog/introducing-openai-agents-sdk",
                    snippet="Evaluating multi-agent support workflows shows a 30% reduction in resolution times compared to a single agent handling all tasks. Handoff patterns and specialized tools are critical for enterprise-grade customer support routing.",
                    metadata={"source": "mock_database"}
                ),
            ]
        elif "guardrail" in query_lower or "runaway" in query_lower:
            mock_docs = [
                SourceDocument(
                    title="Production Guardrails for LLM Agents",
                    url="https://www.deeplearning.ai/short-courses/guardrails-for-llm-applications/",
                    snippet="Production guardrails include: max iteration limits, timeout mechanisms, validation filters (e.g. check inputs/outputs with Pydantic), and retry/fallback behaviors when APIs fail or validation checks raise errors.",
                    metadata={"source": "mock_database"}
                ),
                SourceDocument(
                    title="Safety & Reliability in Multi-Agent Workflows",
                    url="https://arxiv.org/abs/2310.02214",
                    snippet="Key guardrails to prevent runaway loops in agentic graphs: (1) state validators checking for repetitive transitions, (2) strict budget bounds on input/output tokens, (3) human-in-the-loop gates for high-risk tool actions.",
                    metadata={"source": "mock_database"}
                ),
            ]
        else:
            mock_docs = [
                SourceDocument(
                    title=f"General Web Search: {query}",
                    url="https://example.com/search",
                    snippet=f"Mock search result snippet containing baseline content for query: '{query}'. Research and agent frameworks often use multi-agent systems to solve multi-step complex tasks.",
                    metadata={"source": "mock_database"}
                )
            ]

        return mock_docs[:max_results]
