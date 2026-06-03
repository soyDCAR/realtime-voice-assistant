import os
import time

import structlog
from groq import AsyncGroq

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "256"))

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
        self._client = AsyncGroq(api_key=api_key or os.environ["GROQ_API_KEY"])
        logger.info("llm.client_ready", model=model, max_tokens=max_tokens)

    async def complete(self, messages: list[dict]) -> str:
        t0 = time.perf_counter()

        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        text = response.choices[0].message.content

        logger.info(
            "llm.completed",
            model=self.model,
            latency_ms=round(latency_ms, 1),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            text_preview=text[:80],
        )

        return text
