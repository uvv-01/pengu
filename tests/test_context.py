"""Tests for TaskContext — multi-turn conversation tracking."""

import time
import pytest
from pengu.context import TaskContext, ConversationTurn, get_context, reset_context


class TestTaskContext:
    """Test TaskContext functionality."""

    def setup_method(self):
        self.ctx = TaskContext()

    def test_initial_state(self):
        assert self.ctx.current_app == ""
        assert self.ctx.current_url == ""
        assert self.ctx.current_directory == ""
        assert self.ctx.history == []

    def test_update_app(self):
        self.ctx.update_app("chrome", pid=1234)
        assert self.ctx.current_app == "chrome"
        assert self.ctx.current_app_pid == 1234

    def test_update_url(self):
        self.ctx.update_url("https://github.com", "GitHub")
        assert self.ctx.current_url == "https://github.com"
        assert self.ctx.current_page_title == "GitHub"

    def test_update_directory(self):
        self.ctx.update_directory("C:\\Users\\Downloads")
        assert self.ctx.current_directory == "C:\\Users\\Downloads"
        assert self.ctx.last_opened_folder == "C:\\Users\\Downloads"

    def test_update_file(self):
        self.ctx.update_file("C:\\test.py")
        assert self.ctx.last_opened_file == "C:\\test.py"

    def test_record_action(self):
        self.ctx.record_action("open_app", "VS Code opened")
        assert self.ctx.last_action == "open_app"
        assert self.ctx.last_result == "VS Code opened"
        assert self.ctx.last_failure == ""

    def test_record_action_failure(self):
        self.ctx.record_action("open_app", "not found", success=False)
        assert self.ctx.last_failure == "open_app: not found"

    def test_add_turn(self):
        self.ctx.add_turn("open chrome", "Opening Chrome.", action_taken="open_app")
        assert len(self.ctx.history) == 1
        turn = self.ctx.history[0]
        assert turn.user_text == "open chrome"
        assert turn.response == "Opening Chrome."
        assert turn.action_taken == "open_app"

    def test_add_turn_history_limit(self):
        self.ctx.max_history = 3
        for i in range(5):
            self.ctx.add_turn(f"command {i}", f"response {i}")
        assert len(self.ctx.history) == 3
        assert self.ctx.history[0].user_text == "command 2"
        assert self.ctx.history[2].user_text == "command 4"

    def test_is_context_stale(self):
        self.ctx._context_timeout = 0.0  # instant timeout
        time.sleep(0.01)
        assert self.ctx.is_context_stale() is True

    def test_is_context_fresh(self):
        self.ctx._context_timeout = 300.0  # 5 min
        assert self.ctx.is_context_stale() is False

    def test_resolve_followup_folder_navigation(self):
        self.ctx.update_app("File Explorer")
        result = self.ctx.resolve_followup("open downloads")
        assert result == "open downloads"

    def test_resolve_followup_already_has_context(self):
        self.ctx.update_app("chrome")
        result = self.ctx.resolve_followup("open chrome settings")
        assert result == "open chrome settings"

    def test_resolve_followup_no_context(self):
        result = self.ctx.resolve_followup("open downloads")
        assert result == "open downloads"  # unchanged, no context

    def test_resolve_followup_browser_search(self):
        self.ctx.update_url("https://github.com")
        result = self.ctx.resolve_followup("search for python")
        assert result == "search for python"

    def test_get_summary(self):
        self.ctx.update_app("vscode")
        self.ctx.update_url("https://google.com")
        summary = self.ctx.get_summary()
        assert summary["current_app"] == "vscode"
        assert summary["current_url"] == "https://google.com"
        assert summary["history_length"] == 0
        assert summary["context_stale"] is False

    def test_clear(self):
        self.ctx.update_app("chrome")
        self.ctx.update_url("https://google.com")
        self.ctx.add_turn("hello", "hi")
        self.ctx.clear()
        assert self.ctx.current_app == ""
        assert self.ctx.current_url == ""
        assert self.ctx.history == []


class TestContextSingleton:
    """Test the singleton pattern."""

    def test_get_context_returns_same_instance(self):
        ctx1 = get_context()
        ctx2 = get_context()
        assert ctx1 is ctx2

    def test_reset_context(self):
        ctx1 = get_context()
        ctx1.update_app("chrome")
        ctx2 = reset_context()
        assert ctx2.current_app == ""
        assert ctx2 is not ctx1


class TestContextIntegration:
    """Test context integration with the voice loop."""

    def test_full_workflow(self):
        ctx = reset_context()

        # Step 1: Open File Explorer
        ctx.update_app("File Explorer")
        ctx.add_turn("open file explorer", "Opening File Explorer.", action_taken="open_app")

        # Step 2: Open Downloads (follow-up)
        assert ctx.current_app == "File Explorer"
        resolved = ctx.resolve_followup("open downloads")
        assert resolved == "open downloads"

        # Step 3: Open Chrome
        ctx.update_app("chrome")
        ctx.add_turn("open chrome", "Opening Chrome.", action_taken="open_app")

        # Step 4: Search (follow-up in browser context)
        ctx.update_url("https://www.google.com")
        resolved = ctx.resolve_followup("search for python")
        assert resolved == "search for python"

        # Verify history
        assert len(ctx.history) == 2
        assert ctx.history[0].user_text == "open file explorer"
        assert ctx.history[1].user_text == "open chrome"
