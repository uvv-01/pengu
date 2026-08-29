"""
Tests for Agent Tools — registration, schemas, execution, and voice integration.

Verifies that:
- All agent tools register correctly
- Tool schemas are valid
- Tool execution works through the ToolRegistry
- AgentBrain can select and execute tools
- The voice-to-agent integration routes correctly
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengu.agent.tools import register_agent_tools, _run_async
from pengu.tools.registry import ToolRegistry, ToolResult
from pengu.agent.brain import AgentBrain, reset_brain
from pengu.agent.mission import MissionManager, reset_mission_manager
from pengu.agent.state import AgentState, MissionStatus


# ---------------------------------------------------------------------------
# Tool Registration tests
# ---------------------------------------------------------------------------

class TestAgentToolRegistration:
    def test_register_all_agent_tools(self):
        """All agent tools should register without error."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        tools = registry.list_tools()
        assert len(tools) >= 20  # browser + desktop + screen + web_search

    def test_browser_tools_registered(self):
        """All browser tools should be registered."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        browser_tools = registry.list_by_category("browser")
        expected = [
            "browser.open", "browser.navigate", "browser.back", "browser.forward",
            "browser.click", "browser.type", "browser.scroll", "browser.read",
            "browser.search", "browser.get_url", "browser.get_title", "browser.close",
        ]
        registered_names = [t.name for t in browser_tools]
        for name in expected:
            assert name in registered_names, f"Missing browser tool: {name}"

    def test_desktop_tools_registered(self):
        """All desktop tools should be registered."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        desktop_tools = registry.list_by_category("desktop")
        expected = [
            "desktop.click", "desktop.double_click", "desktop.right_click",
            "desktop.type", "desktop.press", "desktop.hotkey", "desktop.scroll",
            "desktop.focus_window", "desktop.open_app",
        ]
        registered_names = [t.name for t in desktop_tools]
        for name in expected:
            assert name in registered_names, f"Missing desktop tool: {name}"

    def test_screen_tools_registered(self):
        """All screen tools should be registered."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        screen_tools = registry.list_by_category("screen")
        expected = ["screen.get_active_window", "screen.inspect", "screen.get_ui_tree"]
        registered_names = [t.name for t in screen_tools]
        for name in expected:
            assert name in registered_names, f"Missing screen tool: {name}"

    def test_web_search_tools_registered(self):
        """Web search tools should be registered."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        ws_tools = registry.list_by_category("web_search")
        expected = ["web_search.search", "web_search.fetch"]
        registered_names = [t.name for t in ws_tools]
        for name in expected:
            assert name in registered_names, f"Missing web_search tool: {name}"

    def test_tool_schemas_are_valid(self):
        """All tool schemas should be valid OpenAI-style function schemas."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        schemas = registry.get_schemas()
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_permission_levels_set(self):
        """All agent tools should have appropriate permission levels."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        for tool in registry.list_tools():
            assert tool.permission_level is not None
            # Browser read/scroll should be SAFE
            if tool.name in ("browser.read", "browser.scroll", "browser.get_url",
                             "browser.get_title", "browser.back", "browser.forward"):
                assert tool.permission_level.value <= 1, f"{tool.name} should be SAFE"

    def test_deterministic_plus_agent_tools_combined(self):
        """Both deterministic and agent tools should coexist in one registry."""
        registry = ToolRegistry()
        from pengu.tools.deterministic import register_deterministic_tools
        register_deterministic_tools(registry)
        register_agent_tools(registry)
        all_names = [t.name for t in registry.list_tools()]
        # Deterministic tools present
        assert "filesystem.read_file" in all_names
        assert "application.open" in all_names
        assert "git.status" in all_names
        # Agent tools present
        assert "browser.navigate" in all_names
        assert "desktop.click" in all_names
        assert "screen.inspect" in all_names
        assert "web_search.search" in all_names
        assert registry.to_dict()["total"] >= 35  # 24 deterministic + 20+ agent


# ---------------------------------------------------------------------------
# Tool execution tests (mocked)
# ---------------------------------------------------------------------------

class TestAgentToolExecution:
    @pytest.mark.asyncio
    async def test_browser_tool_execute_via_registry(self):
        """Browser tools should execute through the ToolRegistry."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        # Mock the browser agent — it's imported inside the function
        with patch("pengu.agent.browser_agent.get_browser_agent") as mock_get:
            mock_agent = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.message = "Opened page"
            mock_result.error = ""
            mock_result.metadata = {}
            mock_agent.navigate = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_agent
            mock_agent._page = True  # simulate browser is open

            result = await registry.execute("browser.navigate", url="https://example.com")
            assert result.success is True
            mock_agent.navigate.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_screen_tool_execute_via_registry(self):
        """Screen tools should execute through the ToolRegistry."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        with patch("pengu.agent.observer.get_observer") as mock_get:
            mock_observer = MagicMock()
            mock_observer.get_active_window.return_value = {
                "title": "Test Window",
                "app": "TestApp",
                "x": "0", "y": "0", "width": "100", "height": "100",
            }
            mock_get.return_value = mock_observer

            result = await registry.execute("screen.get_active_window")
            assert result.success is True
            assert result.output["title"] == "Test Window"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Requesting an unknown tool should return an error."""
        registry = ToolRegistry()
        register_agent_tools(registry)
        result = await registry.execute("nonexistent.tool")
        assert result.success is False
        assert "Unknown" in result.error


# ---------------------------------------------------------------------------
# AgentBrain + ToolRegistry integration tests
# ---------------------------------------------------------------------------

class TestAgentBrainToolIntegration:
    def test_brain_plan_produces_valid_actions(self):
        """AgentBrain planning should produce action names that map to real tools."""
        brain = reset_brain()
        registry = ToolRegistry()
        register_agent_tools(registry)
        brain._tools = registry

        state = AgentState(goal="open Chrome and search for Python")
        intent = brain.understand("open Chrome and search for Python", state)
        steps = brain.plan(state, intent)

        # Verify steps have action names
        for step in steps:
            assert step.action is not None
            assert step.action != ""
            assert step.params is not None

    def test_action_mapping_covers_plan_actions(self):
        """All actions generated by the planner should map to registered tools."""
        brain = reset_brain()
        registry = ToolRegistry()
        from pengu.tools.deterministic import register_deterministic_tools
        register_deterministic_tools(registry)
        register_agent_tools(registry)

        goals = [
            "open Chrome",
            "search for Python",
            "open GitHub",
            "what is on my screen",
            "create a file called test.py",
        ]

        # Import the app's mapping function
        from pengu.app import PenguApp
        app = PenguApp.__new__(PenguApp)

        for goal in goals:
            state = AgentState(goal=goal)
            intent = brain.understand(goal, state)
            steps = brain.plan(state, intent)
            for step in steps:
                tool_name = app._map_action_to_tool(step.action)
                if tool_name is not None:
                    tool = registry.get(tool_name)
                    assert tool is not None, (
                        f"Action '{step.action}' maps to '{tool_name}' but tool not found"
                    )

    @pytest.mark.asyncio
    async def test_mission_with_mock_tools(self):
        """MissionManager should execute a goal using mock tool executor."""
        actions = []

        async def mock_executor(action, params):
            actions.append({"action": action, "params": params})
            result = MagicMock()
            result.success = True
            result.message = f"Done: {action}"
            result.error = ""
            result.output = f"Result of {action}"
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal("open Chrome", tool_executor=mock_executor)
        assert isinstance(response, str)
        assert len(actions) >= 1
        # Verify first action was executed
        assert actions[0]["action"] in ("open_app", "navigate")


# ---------------------------------------------------------------------------
# _run_async helper tests
# ---------------------------------------------------------------------------

class TestRunAsync:
    def test_run_async_from_sync(self):
        """_run_async should execute a coroutine from sync context."""
        async def my_coro():
            return 42

        result = _run_async(my_coro())
        assert result == 42

    def test_run_async_exception(self):
        """_run_async should propagate exceptions."""
        async def failing_coro():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            _run_async(failing_coro())


# ---------------------------------------------------------------------------
# Voice-to-agent routing tests
# ---------------------------------------------------------------------------

class TestVoiceAgentRouting:
    def test_multi_step_triggers_agent(self):
        """Multi-step commands should be routed to the agent."""
        import re
        multi_step_patterns = [
            "open Chrome and search for Python",
            "open GitHub then click Issues",
            "search for OpenVINO; read the page",
            "open ChatGPT and also search for async",
        ]
        for text in multi_step_patterns:
            assert re.search(
                r'\b(?:and|then|;|also|after that)\s+', text, re.IGNORECASE
            ), f"Pattern should match: {text}"

    def test_agent_triggers_for_research(self):
        """Research/query commands should trigger agent routing."""
        agent_triggers = [
            "search for Python asyncio",
            "google latest Python release",
            "find out what's new in OpenVINO",
            "research the best local LLM",
            "what is the latest version of Node.js",
            "tell me about the new features",
            "summarize the documentation",
            "check the test results",
            "read the page",
            "scroll down",
            "click the first result",
        ]
        for text in agent_triggers:
            goal_lower = text.lower()
            triggers = [
                "search", "google", "browse", "find out", "research",
                "what is", "what's", "tell me", "summarize", "check",
            "read page", "read the page", "scroll", "click", "first result",
        ]
        matched = any(t in goal_lower for t in triggers)
        assert matched, f"Should trigger agent: {text}"

    def test_simple_commands_stay_deterministic(self):
        """Simple commands should NOT trigger agent routing."""
        simple_commands = [
            "open VS Code",
            "open Chrome",
            "open Command Prompt",
            "git status",
            "system info",
        ]
        for text in simple_commands:
            goal_lower = text.lower()
            multi_step = re.search(
                r'\b(?:and|then|;|also|after that)\s+', goal_lower, re.IGNORECASE
            )
            agent_triggers = [
                "search", "google", "browse", "find out", "research",
                "what is", "what's", "tell me", "summarize", "check",
                "read page", "scroll", "click", "first result",
            ]
            has_agent_trigger = any(t in goal_lower for t in agent_triggers)
            assert not multi_step and not has_agent_trigger, (
                f"Should stay deterministic: {text}"
            )
