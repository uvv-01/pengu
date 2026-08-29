"""
Tests for the Command Pipeline — end-to-end integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pengu.config import TaskCategory
from pengu.models.base import ChatMessage, ChatResponse
from pengu.pipeline import CommandPipeline, PipelineResult
from pengu.tools.deterministic import register_deterministic_tools
from pengu.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    register_deterministic_tools(registry)
    return registry


@pytest.fixture
def pipeline(tool_registry):
    return CommandPipeline(tool_registry, provider=None)


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.is_available.return_value = True
    provider.health.available = True
    provider.name = "test-provider"
    provider.provider_type = MagicMock()
    provider.provider_type.value = "test"

    async def mock_chat(messages, **kwargs):
        content = "I can help with that. Here's my response."
        return ChatResponse(
            content=content,
            provider="test-provider",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    provider.chat = mock_chat
    return provider


class TestPipelineClassification:
    """Test that the pipeline correctly classifies and routes commands."""

    @pytest.mark.asyncio
    async def test_vs_code_open(self, pipeline):
        result = await pipeline.process("open VS Code")
        assert result.intent.category == TaskCategory.SYSTEM_CONTROL
        assert result.response  # Should have a response

    @pytest.mark.asyncio
    async def test_git_status(self, pipeline):
        result = await pipeline.process("git status")
        assert result.intent.category == TaskCategory.GIT
        # Git status should execute (may fail if not in a repo, but should try)

    @pytest.mark.asyncio
    async def test_hello(self, pipeline):
        result = await pipeline.process("hello")
        assert result.intent.category == TaskCategory.CHAT
        assert result.response  # Should have a response

    @pytest.mark.asyncio
    async def test_empty_input(self, pipeline):
        result = await pipeline.process("")
        assert result.intent.category == TaskCategory.CHAT

    @pytest.mark.asyncio
    async def test_list_directory(self, pipeline):
        result = await pipeline.process("list files")
        assert result.intent.category == TaskCategory.FILE_OPERATION

    @pytest.mark.asyncio
    async def test_vision_handled(self, pipeline):
        result = await pipeline.process("look at my screen")
        assert result.intent.category == TaskCategory.VISION
        assert result.response  # Should produce a response (screenshot or vision analysis)

    @pytest.mark.asyncio
    async def test_browser_not_implemented(self, pipeline):
        result = await pipeline.process("open browser and search for something")
        # Could be SYSTEM_CONTROL or BROWSER depending on rules
        assert result.response  # Should have some response

    @pytest.mark.asyncio
    async def test_network_status(self, pipeline):
        result = await pipeline.process("wifi status")
        assert result.intent.category == TaskCategory.NETWORK


class TestPipelineWithModel:
    """Test pipeline with a mock LLM provider."""

    @pytest.mark.asyncio
    async def test_chat_with_model(self, tool_registry, mock_provider):
        pipeline = CommandPipeline(tool_registry, provider=mock_provider)
        result = await pipeline.process("hello")
        assert result.intent.category == TaskCategory.CHAT
        assert "help with that" in result.response

    @pytest.mark.asyncio
    async def test_coding_with_model(self, tool_registry, mock_provider):
        pipeline = CommandPipeline(tool_registry, provider=mock_provider)
        result = await pipeline.process("write a Python function to sort a list")
        # Should be classified as CODING or handled by model
        assert result.response  # Should have a response from the model
        assert "help with that" in result.response

    @pytest.mark.asyncio
    async def test_unavailable_model_falls_back(self, tool_registry):
        unavailable_provider = MagicMock()
        unavailable_provider.is_available.return_value = False
        unavailable_provider.health.available = False

        pipeline = CommandPipeline(tool_registry, provider=unavailable_provider)
        result = await pipeline.process("what is the meaning of life?")
        # Should still get a response (offline fallback)
        assert result.response
        assert result.error  # Should indicate model unavailable


class TestPipelineDeterministicFirst:
    """Verify that deterministic tools are called before LLM."""

    @pytest.mark.asyncio
    async def test_vscode_does_not_call_llm(self, tool_registry, mock_provider):
        pipeline = CommandPipeline(tool_registry, provider=mock_provider)
        result = await pipeline.process("open VS Code")

        # The pipeline should use the application.open tool, not the LLM
        assert result.tool_used == "application.open"
        assert result.provider == "deterministic"

    @pytest.mark.asyncio
    async def test_git_does_not_call_llm(self, tool_registry, mock_provider):
        pipeline = CommandPipeline(tool_registry, provider=mock_provider)
        result = await pipeline.process("git status")

        # Should use git tool, not LLM
        assert result.provider == "deterministic"


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_to_dict(self, pipeline):
        from pengu.router import Intent
        intent = Intent(
            category=TaskCategory.CHAT,
            confidence=0.9,
            method="rule",
            raw_text="hello",
        )
        result = PipelineResult(
            text="hello",
            intent=intent,
            response="Hi there!",
            provider="deterministic",
        )
        d = result.to_dict()
        assert d["text"] == "hello"
        assert d["category"] == "CHAT"
        assert d["response"] == "Hi there!"
        assert d["provider"] == "deterministic"


class TestPipelineHealthCheck:
    """Test pipeline health and status."""

    def test_pipeline_has_router(self, pipeline):
        assert pipeline.router is not None

    def test_pipeline_has_tools(self, pipeline):
        assert len(pipeline.tool_registry.list_tools()) > 0

    def test_pipeline_no_provider_by_default(self, pipeline):
        assert pipeline.provider is None
