import structlog

from app.llm.client import LLMClient

logger = structlog.get_logger(__name__)


class ConversationManager:
    def __init__(self, client: LLMClient, max_history: int = 20) -> None:
        self._client = client
        self._max_history = max_history
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        logger.info("conversation.cleared")

    async def chat(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        logger.info(
            "conversation.turn_start",
            history_length=len(self._history),
            user_preview=user_text[:80],
        )

        response_text = await self._client.complete(self._history)

        self._history.append({"role": "assistant", "content": response_text})

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        logger.info(
            "conversation.turn_done",
            history_length=len(self._history),
            response_preview=response_text[:80],
        )

        return response_text
