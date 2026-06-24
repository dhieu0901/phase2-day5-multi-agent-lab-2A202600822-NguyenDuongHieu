"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _complete_with_retry(self, system_prompt: str, user_prompt: str) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30.0,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        try:
            response = self._complete_with_retry(system_prompt, user_prompt)
        except Exception as e:
            raise AgentExecutionError(f"LLM call failed after retries: {e}") from e

        content = response.choices[0].message.content or ""
        
        input_tokens = None
        output_tokens = None
        cost_usd = None
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            # Standard gpt-4o-mini pricing: input: $0.15 / 1M tokens, output: $0.60 / 1M tokens
            cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
            
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
