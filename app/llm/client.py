import os
import time

import structlog
from anthropic import AsyncAnthropic

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
_DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "256"))

SYSTEM_PROMPT = (
    "Eres un asistente de voz conversacional. "
    "Responde siempre en español. "
    "Responde conversacionalmente en máximo 2 frases. "
    "Sé directo y natural, como en una conversación hablada."
)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        logger.info("llm.client_ready", model=model, max_tokens=max_tokens)

    async def complete(self, messages: list[dict]) -> str:
        t0 = time.perf_counter()

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.content[0].text

        logger.info(
            "llm.completed",
            model=self.model,
            latency_ms=round(latency_ms, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            text_preview=text[:80],
        )

        return text
