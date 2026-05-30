import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import LLMClient
from app.llm.conversation import ConversationManager


def _make_mock_response(text: str) -> MagicMock:
    """Build a fake Anthropic API response object."""
    content_block = MagicMock()
    content_block.text = text

    usage = MagicMock()
    usage.input_tokens = 42
    usage.output_tokens = 12

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


@pytest.fixture
def mock_anthropic(mocker):
    """Patch AsyncAnthropic so no real HTTP calls are made."""
    mock_create = AsyncMock(
        return_value=_make_mock_response("Hola, ¿en qué te puedo ayudar?")
    )
    mocker.patch(
        "app.llm.client.AsyncAnthropic",
        return_value=MagicMock(
            messages=MagicMock(create=mock_create)
        ),
    )
    return mock_create


@pytest.fixture
def llm_client(mock_anthropic):
    return LLMClient(api_key="test-key-fake")


@pytest.fixture
def conversation(llm_client):
    return ConversationManager(client=llm_client)


# ── LLMClient tests ────────────────────────────────────────────────────────────

async def test_complete_returns_text(llm_client, mock_anthropic):
    result = await llm_client.complete([{"role": "user", "content": "Hola"}])

    assert isinstance(result, str)
    assert len(result) > 0


async def test_complete_calls_api_with_messages(llm_client, mock_anthropic):
    messages = [{"role": "user", "content": "¿Cómo estás?"}]
    await llm_client.complete(messages)

    mock_anthropic.assert_called_once()
    call_kwargs = mock_anthropic.call_args.kwargs
    assert call_kwargs["messages"] == messages


async def test_complete_logs_latency(llm_client, mock_anthropic, mocker):
    mock_log = mocker.patch("app.llm.client.logger.info")
    await llm_client.complete([{"role": "user", "content": "test"}])

    log_calls = [call.args[0] for call in mock_log.call_args_list]
    assert "llm.completed" in log_calls

    completed_call = next(c for c in mock_log.call_args_list if c.args[0] == "llm.completed")
    assert "latency_ms" in completed_call.kwargs


# ── ConversationManager tests ──────────────────────────────────────────────────

async def test_chat_returns_response(conversation):
    response = await conversation.chat("Hola")

    assert isinstance(response, str)
    assert len(response) > 0


async def test_chat_builds_history(conversation):
    await conversation.chat("Hola")
    await conversation.chat("¿Cómo te llamas?")

    history = conversation.history
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "Hola"}
    assert history[1]["role"] == "assistant"
    assert history[2] == {"role": "user", "content": "¿Cómo te llamas?"}
    assert history[3]["role"] == "assistant"


async def test_chat_truncates_history(llm_client):
    manager = ConversationManager(client=llm_client, max_history=4)

    await manager.chat("mensaje 1")
    await manager.chat("mensaje 2")
    await manager.chat("mensaje 3")

    assert len(manager.history) == 4


async def test_clear_resets_history(conversation):
    await conversation.chat("Hola")
    assert len(conversation.history) == 2

    conversation.clear()
    assert len(conversation.history) == 0
