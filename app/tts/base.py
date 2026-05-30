from abc import ABC, abstractmethod


class BaseTTSEngine(ABC):
    """Contract that every TTS backend must fulfil.

    synthesize() takes a text string and returns raw PCM/WAV audio bytes
    ready to be sent over WebSocket to the browser.
    """

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV format).

        Args:
            text: The text to speak.

        Returns:
            Raw WAV audio bytes.
        """
        ...
