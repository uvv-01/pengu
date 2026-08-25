"""
Tests for the LMStudioProvider — local model integration.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from pengu.models.base import ChatMessage, ChatResponse
from pengu.models.lmstudio import LMStudioProvider


@pytest.fixture
def provider():
    return LMStudioProvider(base_url="http://localhost:1234/v1", model="test-model")


class TestLMStudioProviderInit:
    """Test provider initialization."""

    def test_default_base_url(self):
        p = LMStudioProvider()
        assert p.base_url == "http://localhost:1234/v1"

    def test_custom_base_url(self):
        p = LMStudioProvider(base_url="http://localhost:9999/v1")
        assert p.base_url == "http://localhost:9999/v1"

    def test_trailing_slash_stripped(self):
        p = LMStudioProvider(base_url="http://localhost:1234/v1/")
        assert p.base_url == "http://localhost:1234/v1"

    def test_provider_type(self, provider):
        assert provider.provider_type.value == "local_ollama"

    def test_health_starts_unavailable(self, provider):
        assert provider.health.available is False
        assert provider.is_available() is False


class TestLMStudioProviderHealthCheck:
    """Test health check behavior."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "google/gemma-4-e4b"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.health_check()
        assert result is True
        assert provider.health.available is True

    @pytest.mark.asyncio
    async def test_health_check_no_models(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.health_check()
        assert result is False
        assert provider.health.available is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, provider):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.is_closed = False
        provider._client = mock_client

        result = await provider.health_check()
        assert result is False
        assert "not running" in provider.health.error.lower()


class TestLMStudioProviderChat:
    """Test chat completion."""

    @pytest.mark.asyncio
    async def test_chat_success(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "4",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }

        mock_models_response = MagicMock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "data": [{"id": "test-model"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_models_response)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="What is 2+2?")]
        result = await provider.chat(messages)

        assert result.content == "4"
        assert result.provider == "lmstudio"
        assert result.model == "test-model"
        assert result.usage["total_tokens"] == 11
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_chat_connection_error(self, provider):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})))
        mock_client.is_closed = False
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Hello")]
        result = await provider.chat(messages)

        assert result.error != ""
        assert "not running" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_timeout(self, provider):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timed out"))
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})))
        mock_client.is_closed = False
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Hello")]
        result = await provider.chat(messages)

        assert result.error != ""
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_chat_server_error(self, provider):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})))
        mock_client.is_closed = False
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Hello")]
        result = await provider.chat(messages)

        assert result.error != ""
        assert "500" in result.error


class TestLMStudioProviderStructuredOutput:
    """Test structured output parsing."""

    @pytest.mark.asyncio
    async def test_structured_output_json(self, provider):
        """Test that structured output correctly parses JSON responses."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": '{"category": "SYSTEM_CONTROL", "confidence": 0.95}',
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": "test"}]})))
        mock_client.is_closed = False
        provider._client = mock_client

        messages = [ChatMessage(role="user", content="Classify this")]
        schema = {"type": "object", "properties": {"category": {"type": "string"}}}
        result = await provider.structured_output(messages, schema)

        assert "category" in result
        assert result["category"] == "SYSTEM_CONTROL"
