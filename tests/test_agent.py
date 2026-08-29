"""Tests for agent modules: ActionResult, TaskPlanner, DesktopAutomation, BrowserAgent, ScreenObserver."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pengu.agent import ActionResult, ActionType, ActionStatus


# ---------------------------------------------------------------------------
# ActionResult tests
# ---------------------------------------------------------------------------

class TestActionResult:
    """Test ActionResult model."""

    def test_ok(self):
        result = ActionResult.ok("Done", action=ActionType.CLICK, target="Downloads")
        assert result.success is True
        assert result.status == ActionStatus.SUCCESS
        assert result.action == ActionType.CLICK
        assert result.target == "Downloads"
        assert result.verified is True

    def test_fail(self):
        result = ActionResult.fail("Not found", error_code="NOT_FOUND")
        assert result.success is False
        assert result.status == ActionStatus.FAILED
        assert result.error_code == "NOT_FOUND"

    def test_timeout(self):
        result = ActionResult.timeout("Timed out")
        assert result.success is False
        assert result.status == ActionStatus.TIMEOUT
        assert result.error_code == "TIMEOUT"

    def test_not_found(self):
        result = ActionResult.not_found("Button not found", target="Login")
        assert result.success is False
        assert result.status == ActionStatus.NOT_FOUND
        assert result.target == "Login"

    def test_to_dict(self):
        result = ActionResult.ok("Done")
        d = result.to_dict()
        assert d["success"] is True
        assert d["status"] == "success"
        assert d["action"] == "unknown"

    def test_metadata(self):
        result = ActionResult.ok("Done", metadata={"title": "Test"})
        assert result.metadata["title"] == "Test"


# ---------------------------------------------------------------------------
# TaskPlanner tests
# ---------------------------------------------------------------------------

class TestTaskPlanner:
    """Test TaskPlanner creates correct task plans."""

    def setup_method(self):
        from pengu.agent.planner import TaskPlanner
        self.planner = TaskPlanner()

    def test_single_step_open(self):
        plan = self.planner.create_plan("open Chrome")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.OPEN_APP
        assert "chrome" in plan.steps[0].target.lower()

    def test_single_step_click(self):
        plan = self.planner.create_plan("click the Login button")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.CLICK

    def test_single_step_search(self):
        plan = self.planner.create_plan("search for Python asyncio")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.SEARCH

    def test_multi_step(self):
        plan = self.planner.create_plan("open Chrome and search for Python")
        assert len(plan.steps) == 2
        assert plan.steps[0].action == ActionType.OPEN_APP
        assert plan.steps[1].action == ActionType.SEARCH
        assert plan.steps[1].depends_on == 0

    def test_multi_step_three(self):
        plan = self.planner.create_plan("open Chrome; search for Python; click first result")
        assert len(plan.steps) == 3
        assert plan.steps[0].action == ActionType.OPEN_APP
        assert plan.steps[1].action == ActionType.SEARCH
        assert plan.steps[2].action == ActionType.CLICK

    def test_go_to(self):
        plan = self.planner.create_plan("go to GitHub")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.NAVIGATE

    def test_type_text(self):
        plan = self.planner.create_plan("type hello world")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.TYPE_TEXT

    def test_read_page(self):
        plan = self.planner.create_plan("read the page")
        assert len(plan.steps) == 1
        assert plan.steps[0].action == ActionType.READ_PAGE

    def test_plan_to_dict(self):
        plan = self.planner.create_plan("open Chrome")
        d = plan.to_dict()
        assert "goal" in d
        assert "steps" in d
        assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# DesktopAutomation tests (mocked — no real Win32 calls)
# ---------------------------------------------------------------------------

class TestDesktopAutomation:
    """Test DesktopAutomation with mocked Win32 API."""

    @patch("pengu.agent.desktop.user32")
    def test_get_screen_size(self, mock_user32):
        mock_user32.GetSystemMetrics.side_effect = lambda idx: 1920 if idx == 0 else 1080
        from pengu.agent.desktop import DesktopAutomation
        da = DesktopAutomation()
        w, h = da.get_screen_size()
        assert w == 1920
        assert h == 1080

    @patch("pengu.agent.desktop.user32")
    def test_mouse_click(self, mock_user32):
        from pengu.agent.desktop import MouseController
        mc = MouseController()
        mc.click(100, 200)
        assert mock_user32.SendInput.called

    @patch("pengu.agent.desktop.user32")
    def test_keyboard_press_key(self, mock_user32):
        from pengu.agent.desktop import KeyboardController
        kc = KeyboardController()
        kc.press_key("enter")
        assert mock_user32.SendInput.called

    @patch("pengu.agent.desktop.user32")
    def test_window_find(self, mock_user32):
        from pengu.agent.desktop import WindowController
        wc = WindowController()
        # Mock EnumWindows to return nothing (no windows found)
        mock_user32.EnumWindows.return_value = True
        result = wc.find_window("Test Window")
        # Result depends on mock behavior, just verify it doesn't crash
        assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# BrowserAgent tests (mocked)
# ---------------------------------------------------------------------------

class TestBrowserAgent:
    """Test BrowserAgent with mocked Playwright."""

    @pytest.mark.asyncio
    async def test_ensure_browser_not_installed(self):
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            from pengu.agent.browser_agent import BrowserAgent
            agent = BrowserAgent()
            result = await agent.ensure_browser()
            assert result.success is False
            assert "Playwright not installed" in result.message

    @pytest.mark.asyncio
    async def test_navigate_not_ready(self):
        from pengu.agent.browser_agent import BrowserAgent
        agent = BrowserAgent()
        # Mock ensure_browser to fail
        agent.ensure_browser = AsyncMock(return_value=ActionResult.fail("No browser"))
        result = await agent.navigate("https://example.com")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_find_and_click_no_page(self):
        from pengu.agent.browser_agent import BrowserAgent
        agent = BrowserAgent()
        result = await agent.find_and_click("Login")
        assert result.success is False
        assert "Browser not open" in result.message

    @pytest.mark.asyncio
    async def test_read_page_no_page(self):
        from pengu.agent.browser_agent import BrowserAgent
        agent = BrowserAgent()
        result = await agent.read_page()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_close(self):
        from pengu.agent.browser_agent import BrowserAgent
        agent = BrowserAgent()
        await agent.close()  # Should not crash
        assert agent._browser is None


# ---------------------------------------------------------------------------
# ScreenObserver tests
# ---------------------------------------------------------------------------

class TestScreenObserver:
    """Test ScreenObserver."""

    @patch("pengu.agent.desktop.user32")
    def test_get_active_window(self, mock_user32):
        mock_user32.GetForegroundWindow.return_value = 12345
        mock_user32.GetWindowTextLengthW.return_value = 10
        mock_user32.IsWindowVisible.return_value = True
        mock_user32.GetSystemMetrics.side_effect = lambda idx: 1920 if idx == 0 else 1080
        mock_user32.GetWindowRect.side_effect = lambda hwnd, rect: None

        from pengu.agent.observer import ScreenObserver
        observer = ScreenObserver()
        info = observer.get_active_window()
        assert "title" in info
        assert "app" in info

    def test_extract_app_name(self):
        from pengu.agent.observer import ScreenObserver
        observer = ScreenObserver()
        assert observer._extract_app_name("GitHub - Google Chrome") == "Google Chrome"
        assert observer._extract_app_name("Downloads - File Explorer") == "File Explorer"
        assert observer._extract_app_name("SingleWindow") == "SingleWindow"
        assert observer._extract_app_name("") == ""


# ---------------------------------------------------------------------------
# Integration: context + planner
# ---------------------------------------------------------------------------

class TestContextPlannerIntegration:
    """Test that context and planner work together."""

    def test_context_resolves_followup(self):
        from pengu.context import TaskContext
        ctx = TaskContext()
        ctx.update_app("File Explorer")
        resolved = ctx.resolve_followup("open downloads")
        assert "downloads" in resolved.lower()

    def test_planner_handles_multi_step(self):
        from pengu.agent.planner import TaskPlanner
        planner = TaskPlanner()
        plan = planner.create_plan("open ChatGPT and search for quantum computing")
        assert len(plan.steps) == 2
        assert plan.steps[0].action == ActionType.OPEN_APP
        assert plan.steps[1].action == ActionType.SEARCH
