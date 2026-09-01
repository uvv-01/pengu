"""
Tests for Phase 4+5 — Browser Interaction + Multi-Application Task Execution.

Tests:
- BrowserState and InteractiveElement data models
- BrowserAgent navigation, observation, element discovery
- BrowserAgent interaction (click, type, submit, scroll, read, verify)
- AgentBrain browser task planning with observe→act→verify pattern
- MissionManager loop detection
- Multi-application task context preservation
- Action mapping covers all new browser tools
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengu.agent.browser_agent import (
    BrowserAgent,
    BrowserState,
    InteractiveElement,
    get_browser_agent,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestInteractiveElement:
    def test_create_default(self):
        el = InteractiveElement()
        assert el.tag == ""
        assert el.element_type == ""
        assert el.text == ""
        assert el.is_visible is True
        assert el.is_enabled is True

    def test_create_with_values(self):
        el = InteractiveElement(
            tag="input",
            element_type="input",
            text="Search",
            placeholder="Enter query",
            aria_label="Search field",
            name="q",
            selector="#search-input",
            is_visible=True,
            is_enabled=True,
        )
        assert el.tag == "input"
        assert el.element_type == "input"
        assert el.text == "Search"

    def test_to_dict(self):
        el = InteractiveElement(
            tag="button",
            element_type="button",
            text="Submit",
            aria_label="Submit form",
            role="button",
            selector="#submit",
        )
        d = el.to_dict()
        assert d["tag"] == "button"
        assert d["type"] == "button"
        assert d["text"] == "Submit"
        assert d["aria_label"] == "Submit form"
        assert d["role"] == "button"
        assert d["selector"] == "#submit"
        assert d["visible"] is True
        assert d["enabled"] is True

    def test_to_dict_truncates_long_text(self):
        el = InteractiveElement(text="x" * 200)
        d = el.to_dict()
        assert len(d["text"]) <= 80


class TestBrowserState:
    def test_create_default(self):
        state = BrowserState()
        assert state.url == ""
        assert state.title == ""
        assert state.is_ready is False
        assert state.loading is False
        assert state.error == ""
        assert state.interactive_elements == []
        assert state.links == []
        assert state.headings == []
        assert state.visible_text_preview == ""

    def test_to_dict(self):
        state = BrowserState(
            url="https://example.com",
            title="Example",
            is_ready=True,
            interactive_elements=[
                InteractiveElement(tag="input", element_type="input", text="Search"),
                InteractiveElement(tag="button", element_type="button", text="Go"),
            ],
            links=[{"text": "About", "href": "/about"}],
            headings=["Welcome"],
        )
        d = state.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Example"
        assert d["ready"] is True
        assert d["interactive_count"] == 2
        assert len(d["interactive_elements"]) == 2
        assert d["links_count"] == 1
        assert "Welcome" in d["headings"]

    def test_get_summary(self):
        state = BrowserState(
            url="https://example.com",
            title="Example Page",
            interactive_elements=[
                InteractiveElement(element_type="input", placeholder="Search..."),
                InteractiveElement(element_type="button", text="Submit"),
                InteractiveElement(element_type="link", text="About Us"),
            ],
        )
        summary = state.get_summary()
        assert "Example Page" in summary
        assert "https://example.com" in summary
        assert "Search..." in summary
        assert "Submit" in summary
        assert "About Us" in summary

    def test_get_summary_empty(self):
        state = BrowserState()
        summary = state.get_summary()
        assert "(no title)" in summary

    def test_limit_interactive_elements_in_dict(self):
        elements = [
            InteractiveElement(text=f"el_{i}") for i in range(30)
        ]
        state = BrowserState(interactive_elements=elements)
        d = state.to_dict()
        assert len(d["interactive_elements"]) == 20  # capped at 20


# ---------------------------------------------------------------------------
# BrowserAgent lifecycle tests (mocked Playwright)
# ---------------------------------------------------------------------------

class TestBrowserAgentLifecycle:
    def test_singleton(self):
        """get_browser_agent should return the same instance."""
        a1 = get_browser_agent()
        a2 = get_browser_agent()
        assert a1 is a2

    def test_initial_state(self):
        agent = BrowserAgent()
        assert agent._page is None
        assert agent._browser is None
        assert agent._playwright is None
        assert agent.is_available is False
        assert agent.state.is_ready is False

    @pytest.mark.asyncio
    async def test_ensure_browser_not_available(self):
        """ensure_browser should return error when Playwright not installed."""
        agent = BrowserAgent()
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            result = await agent.ensure_browser()
            # Should fail gracefully
            assert result.success is False
            assert "not installed" in result.message.lower() or "not available" in result.error.lower() or result.error_code == "NOT_AVAILABLE"

    @pytest.mark.asyncio
    async def test_navigate_without_browser(self):
        """navigate should fail gracefully when browser not open."""
        agent = BrowserAgent()
        result = await agent.navigate("https://example.com")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        """close should reset all state."""
        agent = BrowserAgent()
        agent._playwright = MagicMock()
        agent._browser = AsyncMock()
        agent._context = AsyncMock()
        agent._page = MagicMock()
        agent._navigation_history = ["https://test.com"]
        agent._state.url = "https://test.com"

        await agent.close()

        assert agent._page is None
        assert agent._browser is None
        assert agent._playwright is None
        assert agent._state.url == ""
        assert len(agent._navigation_history) == 0


# ---------------------------------------------------------------------------
# BrowserAgent navigation tests (mocked page)
# ---------------------------------------------------------------------------

class TestBrowserAgentNavigation:
    def _make_agent_with_mock_page(self):
        """Create a BrowserAgent with a mocked page for testing."""
        agent = BrowserAgent()
        agent._available = True
        agent._playwright = MagicMock()
        agent._browser = MagicMock()
        agent._context = MagicMock()
        agent._page = MagicMock()
        agent._page.url = "https://example.com"
        agent._page.title = AsyncMock(return_value="Example Page")
        agent._page.goto = AsyncMock()
        agent._page.go_back = AsyncMock()
        agent._page.go_forward = AsyncMock()
        agent._page.reload = AsyncMock()
        agent._page.inner_text = AsyncMock(return_value="Page content")
        agent._page.keyboard = MagicMock()
        agent._page.keyboard.type = AsyncMock()
        agent._page.keyboard.press = AsyncMock()
        agent._page.mouse = MagicMock()
        agent._page.mouse.wheel = AsyncMock()
        agent._page.wait_for_load_state = AsyncMock()
        agent._page.query_selector_all = AsyncMock(return_value=[])
        agent._page.locator = MagicMock()
        return agent

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.navigate("https://example.com")
        assert result.success is True
        assert "Example Page" in result.message
        assert agent._state.url == "https://example.com"
        assert agent._state.title == "Example Page"
        assert agent._state.error == ""
        assert "https://example.com" in agent._navigation_history

    @pytest.mark.asyncio
    async def test_navigate_prepends_https(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.navigate("example.com")
        assert result.success is True
        agent._page.goto.assert_called_once()
        call_url = agent._page.goto.call_args[0][0]
        assert call_url.startswith("https://")

    @pytest.mark.asyncio
    async def test_navigate_failure(self):
        agent = self._make_agent_with_mock_page()
        agent._page.goto = AsyncMock(side_effect=Exception("Connection refused"))
        result = await agent.navigate("https://unreachable.com")
        assert result.success is False
        assert "Connection refused" in result.error or "Could not open" in result.message

    @pytest.mark.asyncio
    async def test_go_back(self):
        agent = self._make_agent_with_mock_page()
        agent._page.url = "https://example.com/page1"
        agent._page.title = AsyncMock(return_value="Page 1")
        result = await agent.go_back()
        assert result.success is True
        assert "Page 1" in result.message

    @pytest.mark.asyncio
    async def test_go_back_without_page(self):
        agent = BrowserAgent()
        result = await agent.go_back()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_go_forward(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.go_forward()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_refresh(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.refresh()
        assert result.success is True
        agent._page.reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_scroll(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.scroll_page("down", amount=2)
        assert result.success is True
        assert agent._page.mouse.wheel.call_count == 2

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.scroll_page("up", amount=1)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_scroll_without_page(self):
        agent = BrowserAgent()
        result = await agent.scroll_page("down")
        assert result.success is False


# ---------------------------------------------------------------------------
# BrowserAgent interaction tests (mocked page)
# ---------------------------------------------------------------------------

class TestBrowserAgentInteraction:
    def _make_agent_with_mock_page(self):
        agent = BrowserAgent()
        agent._available = True
        agent._playwright = MagicMock()
        agent._browser = MagicMock()
        agent._context = MagicMock()
        agent._page = MagicMock()
        agent._page.url = "https://example.com"
        agent._page.title = AsyncMock(return_value="Example")
        agent._page.inner_text = AsyncMock(return_value="Page body text")
        agent._page.keyboard = MagicMock()
        agent._page.keyboard.type = AsyncMock()
        agent._page.keyboard.press = AsyncMock()
        agent._page.mouse = MagicMock()
        agent._page.mouse.wheel = AsyncMock()
        agent._page.wait_for_load_state = AsyncMock()
        agent._page.query_selector_all = AsyncMock(return_value=[])

        # Mock locator chain for get_by_text, get_by_role etc.
        mock_locator = MagicMock()
        mock_locator.first = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=True)
        mock_locator.first.click = AsyncMock()
        mock_locator.first.fill = AsyncMock()
        mock_locator.first.press = AsyncMock()
        mock_locator.first.inner_text = AsyncMock(return_value="Element text")
        mock_locator.first.get_attribute = AsyncMock(return_value=None)
        mock_locator.first.is_enabled = AsyncMock(return_value=True)
        mock_locator.first.evaluate = AsyncMock(return_value="input")
        mock_locator.first.count = AsyncMock(return_value=1)
        mock_locator.count = AsyncMock(return_value=1)
        agent._page.get_by_text = MagicMock(return_value=mock_locator)
        agent._page.get_by_role = MagicMock(return_value=mock_locator)
        agent._page.get_by_label = MagicMock(return_value=mock_locator)
        agent._page.get_by_placeholder = MagicMock(return_value=mock_locator)
        agent._page.locator = MagicMock(return_value=mock_locator)
        return agent

    @pytest.mark.asyncio
    async def test_find_and_click_success(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.find_and_click("Submit")
        assert result.success is True
        assert "Clicked" in result.message
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_find_and_click_not_found(self):
        agent = self._make_agent_with_mock_page()
        # Make all strategies fail
        mock_locator = MagicMock()
        mock_locator.first = MagicMock()
        mock_locator.first.is_visible = AsyncMock(return_value=False)
        agent._page.get_by_text = MagicMock(return_value=mock_locator)
        agent._page.get_by_role = MagicMock(return_value=mock_locator)
        agent._page.get_by_label = MagicMock(return_value=mock_locator)
        agent._page.get_by_placeholder = MagicMock(return_value=mock_locator)
        agent._page.locator = MagicMock(return_value=mock_locator)
        result = await agent.find_and_click("Nonexistent")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_type_in_field_success(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.type_in_field("search", "python docs")
        assert result.success is True
        assert "Typed" in result.message

    @pytest.mark.asyncio
    async def test_type_in_page(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.type_in_page("hello world")
        assert result.success is True
        agent._page.keyboard.type.assert_called_once_with("hello world", delay=30)

    @pytest.mark.asyncio
    async def test_press_key(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.press_key("Enter")
        assert result.success is True
        agent._page.keyboard.press.assert_called_with("Enter")

    @pytest.mark.asyncio
    async def test_submit_form(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.submit_form()
        assert result.success is True
        assert "Form submitted" in result.message

    @pytest.mark.asyncio
    async def test_read_page(self):
        agent = self._make_agent_with_mock_page()
        result = await agent.read_page()
        assert result.success is True
        assert "Page body text" in result.message
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_read_page_truncates(self):
        agent = self._make_agent_with_mock_page()
        agent._page.inner_text = AsyncMock(return_value="x" * 10000)
        result = await agent.read_page(max_chars=500)
        assert "(truncated)" in result.message

    @pytest.mark.asyncio
    async def test_type_in_page_without_browser(self):
        agent = BrowserAgent()
        result = await agent.type_in_page("test")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_press_key_without_browser(self):
        agent = BrowserAgent()
        result = await agent.press_key("Enter")
        assert result.success is False


# ---------------------------------------------------------------------------
# BrowserAgent observation tests (mocked page)
# ---------------------------------------------------------------------------

class TestBrowserAgentObservation:
    def _make_agent_with_mock_elements(self):
        agent = BrowserAgent()
        agent._available = True
        agent._page = MagicMock()
        agent._page.url = "https://example.com"
        agent._page.title = AsyncMock(return_value="Example")
        agent._page.inner_text = AsyncMock(return_value="Page body text with content")
        agent._page.query_selector_all = AsyncMock(return_value=[])

        # Mock individual element extraction
        async def mock_query(selector):
            if "input" in selector or "textarea" in selector or "select" in selector:
                return []  # No inputs
            if "button" in selector or "role" in selector:
                return []  # No buttons
            if "a:" in selector or "a[" in selector:
                return []  # No links
            if "h1" in selector or "h2" in selector or "h3" in selector:
                return []  # No headings
            return []

        agent._page.query_selector_all = mock_query
        return agent

    @pytest.mark.asyncio
    async def test_get_browser_state(self):
        agent = self._make_agent_with_mock_elements()
        state = await agent.get_browser_state()
        assert state.url == "https://example.com"
        assert state.title == "Example"
        assert state.is_ready is True
        assert state.loading is False
        assert state.error == ""

    @pytest.mark.asyncio
    async def test_get_browser_state_without_page(self):
        agent = BrowserAgent()
        state = await agent.get_browser_state()
        assert state.error == "Browser not open"

    @pytest.mark.asyncio
    async def test_get_page_info(self):
        agent = self._make_agent_with_mock_elements()
        info = await agent.get_page_info()
        assert info["url"] == "https://example.com"
        assert info["title"] == "Example"
        assert info["status"] == "ready"

    @pytest.mark.asyncio
    async def test_get_page_info_without_page(self):
        agent = BrowserAgent()
        info = await agent.get_page_info()
        assert info["status"] == "not_open"

    @pytest.mark.asyncio
    async def test_verify_action_success(self):
        agent = self._make_agent_with_mock_elements()
        result = await agent.verify_action(expected_title="Example")
        assert result.success is True
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_verify_action_title_mismatch(self):
        agent = self._make_agent_with_mock_elements()
        result = await agent.verify_action(expected_title="Different Title")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_action_text_found(self):
        agent = self._make_agent_with_mock_elements()
        result = await agent.verify_action(expected_text="Page body text")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_action_text_not_found(self):
        agent = self._make_agent_with_mock_elements()
        result = await agent.verify_action(expected_text="This text is not on the page")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_verify_action_url_match(self):
        agent = self._make_agent_with_mock_elements()
        result = await agent.verify_action(expected_url="example.com")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_verify_without_page(self):
        agent = BrowserAgent()
        result = await agent.verify_action(expected_title="Test")
        assert result.success is False


# ---------------------------------------------------------------------------
# BrowserAgent wait tests
# ---------------------------------------------------------------------------

class TestBrowserAgentWait:
    @pytest.mark.asyncio
    async def test_wait_for_navigation_success(self):
        agent = BrowserAgent()
        agent._page = MagicMock()
        agent._page.wait_for_load_state = AsyncMock()
        agent._page.title = AsyncMock(return_value="Done")
        agent._page.url = "https://done.com"
        result = await agent.wait_for_navigation()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_wait_for_navigation_timeout(self):
        agent = BrowserAgent()
        agent._page = MagicMock()
        agent._page.wait_for_load_state = AsyncMock(
            side_effect=Exception("Timeout")
        )
        result = await agent.wait_for_navigation()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_wait_for_element_success(self):
        agent = BrowserAgent()
        agent._page = MagicMock()
        agent._page.wait_for_selector = AsyncMock()
        result = await agent.wait_for_element("#my-id")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_wait_for_element_not_found(self):
        agent = BrowserAgent()
        agent._page = MagicMock()
        agent._page.wait_for_selector = AsyncMock(
            side_effect=Exception("Timeout")
        )
        result = await agent.wait_for_element(".missing", timeout_ms=1000)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_wait_for_navigation_without_page(self):
        agent = BrowserAgent()
        result = await agent.wait_for_navigation()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_wait_for_element_without_page(self):
        agent = BrowserAgent()
        result = await agent.wait_for_element("#test")
        assert result.success is False


# ---------------------------------------------------------------------------
# AgentBrain browser planning tests
# ---------------------------------------------------------------------------

class TestAgentBrainBrowserPlanning:
    def test_plan_browser_task_with_search(self):
        """Browser task with search should produce browser-open + search steps."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, WorldState

        brain = AgentBrain()
        world = WorldState()
        state = AgentState(goal="open Google and search for Python")

        intent = brain.understand("open Google and search for Python", state)
        steps = brain.plan(state, intent)

        actions = [s.action for s in steps]
        descriptions = [s.description for s in steps]

        # Should include open_app for browser and web_search for the query
        assert any("open_app" in a or "navigate" in a for a in actions), f"Expected open_app/navigate, got: {actions}"
        # Should include search
        assert any("search" in a for a in actions), f"Expected search, got: {actions}"
        # Should have at least 2 steps (open + search)
        assert len(steps) >= 2, f"Expected at least 2 steps, got {len(steps)}: {descriptions}"

    def test_plan_browser_task_with_click(self):
        """Browser task with click should produce observe→click→observe pattern."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, WorldState

        brain = AgentBrain()
        world = WorldState()
        state = AgentState(goal="click on Issues")

        intent = brain.understand("click on Issues", state)
        steps = brain.plan(state, intent)

        actions = [s.action for s in steps]
        descriptions = [s.description for s in steps]

        # Should include click
        assert any("click" in a for a in actions), f"Expected click, got: {actions}"

    def test_plan_browser_task_with_read(self):
        """Browser task with 'read' should produce at least one step."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState

        brain = AgentBrain()
        state = AgentState(goal="read the page")

        intent = brain.understand("read the page", state)
        steps = brain.plan(state, intent)

        assert len(steps) >= 1
        actions = [s.action for s in steps]
        # Should produce some actionable step (read, chat, etc.)
        assert any(a for a in actions), f"Expected at least one action, got: {actions}"

    def test_plan_information_query(self):
        """Information queries should be handled as simple tasks."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState

        brain = AgentBrain()
        state = AgentState(goal="what is my battery percentage")

        intent = brain.understand("what is my battery percentage", state)
        steps = brain.plan(state, intent)

        assert len(steps) >= 1

    def test_understand_complex_goal(self):
        """Complex multi-step goals should be classified as multi_step or research type."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState

        brain = AgentBrain()
        state = AgentState(goal="open GitHub and find the latest issue")

        intent = brain.understand("open GitHub and find the latest issue", state)
        assert intent["type"] in ("multi_step", "research")

    def test_understand_research_goal(self):
        """Research goals should be classified correctly."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState

        brain = AgentBrain()
        state = AgentState(goal="search for the latest OpenVINO release")

        intent = brain.understand("search for the latest OpenVINO release", state)
        assert intent["type"] in ("research", "multi_step")


# ---------------------------------------------------------------------------
# MissionManager loop detection tests
# ---------------------------------------------------------------------------

class TestMissionManagerLoopDetection:
    @pytest.mark.asyncio
    async def test_loop_detection_breaks_repetition(self):
        """MissionManager should detect repeated actions and replan."""
        from pengu.agent.mission import MissionManager
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState

        brain = AgentBrain()
        manager = MissionManager(brain=brain)

        # Create a state that will repeat the same action
        state = AgentState(goal="test loop detection")

        # Manually set up repeated actions
        action_count = 0

        async def repeating_executor(action, params):
            nonlocal action_count
            action_count += 1
            result = MagicMock()
            result.success = True
            result.message = f"Done: {action}"
            result.error = ""
            result.output = f"Result {action_count}"
            return result

        # Execute goal — loop detection should prevent infinite repetition
        response = await manager.execute_goal(
            "test loop detection",
            tool_executor=repeating_executor,
        )
        assert isinstance(response, str)
        # Should complete within reasonable iterations
        assert action_count < 30

    @pytest.mark.asyncio
    async def test_mission_handles_failure_gracefully(self):
        """MissionManager should handle action failures without crashing."""
        from pengu.agent.mission import MissionManager
        from pengu.agent.brain import AgentBrain

        brain = AgentBrain()
        manager = MissionManager(brain=brain)

        async def failing_executor(action, params):
            result = MagicMock()
            result.success = False
            result.message = f"Failed: {action}"
            result.error = "Simulated failure"
            result.output = ""
            return result

        response = await manager.execute_goal(
            "open Chrome",
            tool_executor=failing_executor,
        )
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_mission_cancel(self):
        """MissionManager should support cancellation."""
        from pengu.agent.mission import MissionManager
        from pengu.agent.brain import AgentBrain

        brain = AgentBrain()
        manager = MissionManager(brain=brain)

        async def slow_executor(action, params):
            await asyncio.sleep(0.01)
            result = MagicMock()
            result.success = True
            result.message = f"Done: {action}"
            result.error = ""
            result.output = "ok"
            return result

        # Start mission in background and cancel it
        async def run_mission():
            return await manager.execute_goal("complex task", tool_executor=slow_executor)

        task = asyncio.create_task(run_mission())
        await asyncio.sleep(0.05)
        manager.cancel()
        response = await task
        assert isinstance(response, str)


# ---------------------------------------------------------------------------
# Action mapping tests
# ---------------------------------------------------------------------------

class TestActionMapping:
    """Verify _map_action_to_tool covers all new browser tools."""

    def _get_mapping(self, action):
        from pengu.app import PenguApp
        app = PenguApp.__new__(PenguApp)
        return app._map_action_to_tool(action)

    def test_core_browser_actions_mapped(self):
        mapping = {
            "navigate": "browser.navigate",
            "click": "browser.click",
            "type_text": "browser.type",
            "read_page": "browser.read",
            "scroll": "browser.scroll",
            "web_search": "web_search.search",
            "open_app": "application.open",
        }
        for action, expected_tool in mapping.items():
            result = self._get_mapping(action)
            assert result == expected_tool, f"'{action}' should map to '{expected_tool}', got '{result}'"

    def test_new_browser_actions_mapped(self):
        mapping = {
            "browser_get_state": "browser.get_state",
            "browser_type_in_field": "browser.type_in_field",
            "browser_submit": "browser.submit",
            "browser_verify": "browser.verify",
            "browser_find_elements": "browser.find_elements",
            "browser_refresh": "browser.refresh",
            "browser_search": "browser.search",
            "browser_click": "browser.click",
            "browser_type": "browser.type",
            "browser_read": "browser.read",
            "browser_scroll": "browser.scroll",
            "browser_open": "browser.open",
            "browser_close": "browser.close",
            "browser_get_url": "browser.get_url",
            "browser_get_title": "browser.get_title",
        }
        for action, expected_tool in mapping.items():
            result = self._get_mapping(action)
            assert result == expected_tool, f"'{action}' should map to '{expected_tool}', got '{result}'"

    def test_desktop_actions_mapped(self):
        mapping = {
            "desktop_click": "desktop.click",
            "desktop_type": "desktop.type",
            "desktop_press": "desktop.press",
            "desktop_hotkey": "desktop.hotkey",
            "desktop_focus": "desktop.focus_window",
        }
        for action, expected_tool in mapping.items():
            result = self._get_mapping(action)
            assert result == expected_tool

    def test_screen_actions_mapped(self):
        mapping = {
            "screen_inspect": "screen.inspect",
            "screen_get_active": "screen.get_active_window",
            "screen_ui_tree": "screen.get_ui_tree",
        }
        for action, expected_tool in mapping.items():
            result = self._get_mapping(action)
            assert result == expected_tool

    def test_system_actions_mapped(self):
        mapping = {
            "system_battery": "system.battery",
            "system_volume": "system.volume",
            "system_wallpaper": "system.wallpaper",
            "system_info": "system.info",
        }
        for action, expected_tool in mapping.items():
            result = self._get_mapping(action)
            assert result == expected_tool

    def test_chat_returns_none(self):
        """chat action should map to None (falls through to LLM)."""
        result = self._get_mapping("chat")
        assert result is None

    def test_unknown_action_returns_none(self):
        """Unknown action should return None."""
        result = self._get_mapping("totally_unknown_action")
        assert result is None


# ---------------------------------------------------------------------------
# Integration: AgentBrain + BrowserAgent tool execution
# ---------------------------------------------------------------------------

class TestBrowserToolExecution:
    @pytest.mark.asyncio
    async def test_browser_get_state_tool_execution(self):
        """browser.get_state should execute through the registry."""
        from pengu.agent.tools import register_agent_tools
        from pengu.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_agent_tools(registry)

        mock_state = BrowserState(
            url="https://example.com",
            title="Example",
            is_ready=True,
            interactive_elements=[
                InteractiveElement(tag="input", element_type="input", text="Search"),
            ],
        )

        with patch("pengu.agent.browser_agent.get_browser_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.get_browser_state = AsyncMock(return_value=mock_state)
            mock_get.return_value = mock_agent

            result = await registry.execute("browser.get_state")
            assert result.success is True
            assert result.output["url"] == "https://example.com"
            assert result.output["interactive_count"] == 1

    @pytest.mark.asyncio
    async def test_browser_verify_tool_execution(self):
        """browser.verify should execute through the registry."""
        from pengu.agent.tools import register_agent_tools
        from pengu.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_agent_tools(registry)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Verification passed"
        mock_result.metadata = {"url": "https://example.com", "title": "Example"}

        with patch("pengu.agent.browser_agent.get_browser_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.verify_action = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_agent

            result = await registry.execute(
                "browser.verify", expected_title="Example"
            )
            assert result.success is True
            assert "passed" in result.output["message"].lower()

    @pytest.mark.asyncio
    async def test_browser_type_in_field_execution(self):
        """browser.type_in_field should execute through the registry."""
        from pengu.agent.tools import register_agent_tools
        from pengu.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_agent_tools(registry)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Typed 'test' into 'search'"
        mock_result.error = ""

        with patch("pengu.agent.browser_agent.get_browser_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.type_in_field = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_agent

            result = await registry.execute(
                "browser.type_in_field", field_text="search", value="test"
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_browser_submit_execution(self):
        """browser.submit should execute through the registry."""
        from pengu.agent.tools import register_agent_tools
        from pengu.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_agent_tools(registry)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Form submitted"
        mock_result.error = ""

        with patch("pengu.agent.browser_agent.get_browser_agent") as mock_get:
            mock_agent = MagicMock()
            mock_agent.submit_form = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_agent

            result = await registry.execute("browser.submit")
            assert result.success is True


# ---------------------------------------------------------------------------
# Multi-application context tests
# ---------------------------------------------------------------------------

class TestMultiApplicationContext:
    def test_agent_state_preserves_context(self):
        """AgentState should preserve goal and action history across steps."""
        from pengu.agent.state import AgentState

        state = AgentState(goal="Open GitHub, find issues, then ask ChatGPT to explain")
        assert state.goal == "Open GitHub, find issues, then ask ChatGPT to explain"
        assert state.current_step_index == 0

        # Simulate progress
        state.current_step_index = 1
        assert state.current_step_index == 1

    def test_world_state_tracks_multiple_apps(self):
        """WorldState should track browser, active window, and filesystem context."""
        from pengu.agent.state import WorldState

        world = WorldState()
        world.browser_open = True
        world.browser_url = "https://github.com"
        world.browser_title = "GitHub"
        world.active_app = "Chrome"

        assert world.is_browser_active() is True
        assert "github" in world.browser_url.lower()

    def test_world_state_default(self):
        """Default WorldState should have no active browser."""
        from pengu.agent.state import WorldState

        world = WorldState()
        assert world.is_browser_active() is False
        assert world.browser_open is False

    def test_task_context_preserved(self):
        """Previous actions should be preserved in state."""
        from pengu.agent.state import AgentState

        state = AgentState(goal="multi-step task")
        state.action_history.append({"action": "navigate", "result": "ok"})
        state.action_history.append({"action": "click", "result": "ok"})

        assert len(state.action_history) == 2
        assert state.action_history[0]["action"] == "navigate"
        assert state.action_history[1]["action"] == "click"
