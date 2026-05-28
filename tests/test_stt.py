import pytest
from pathlib import Path

from app.stt.engine import Transcriber

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_WAV = FIXTURES_DIR / "sample.wav"


@pytest.fixture(scope="module")
def transcriber():
    return Transcriber(model_size="base", device="cpu", language="es")


def test_transcribe_returns_text(transcriber):
    result = transcriber.transcribe(SAMPLE_WAV)

    assert isinstance(result, str)
    assert len(result) > 0


def test_transcribe_contains_expected_words(transcriber):
    result = transcriber.transcribe(SAMPLE_WAV).lower()

    assert "hola" in result
    assert "prueba" in result


def test_transcribe_missing_file_raises(transcriber):
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(FIXTURES_DIR / "nonexistent.wav")
