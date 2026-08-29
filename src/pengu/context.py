"""
Task Context — short-term session memory for multi-turn conversations.

Tracks:
  - current application / window
  - current URL / page
  - current task / goal
  - previous actions and results
  - conversation history (last N turns)
  - last failure and recovery state

This allows Pengu to understand follow-up commands like:
  "Open Downloads" (after "Open File Explorer")
  "Search for Python" (after "Open Chrome")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    timestamp: float
    user_text: str
    response: str
    action_taken: str = ""
    success: bool = True


@dataclass
class TaskContext:
    """
    Short-term session context for multi-turn interactions.

    This is NOT permanent memory — it tracks the current task session
    and resets when the user starts a completely new topic.
    """
    # Current application context
    current_app: str = ""
    current_app_pid: int = 0

    # Browser context
    current_url: str = ""
    current_page_title: str = ""

    # Filesystem context
    current_directory: str = ""
    last_opened_folder: str = ""
    last_opened_file: str = ""

    # Task state
    current_task: str = ""
    last_action: str = ""
    last_result: str = ""
    last_failure: str = ""

    # Conversation history (last N turns)
    history: list[ConversationTurn] = field(default_factory=list)
    max_history: int = 10

    # Context freshness
    _last_update: float = field(default_factory=time.time)
    _context_timeout: float = 300.0  # 5 minutes

    def update_app(self, app_name: str, pid: int = 0) -> None:
        """Track that we just opened/focused an application."""
        self.current_app = app_name
        self.current_app_pid = pid
        self._last_update = time.time()

    def update_url(self, url: str, title: str = "") -> None:
        """Track that we navigated to a URL."""
        self.current_url = url
        self.current_page_title = title
        self._last_update = time.time()

    def update_directory(self, path: str) -> None:
        """Track the current directory / folder context."""
        self.current_directory = path
        self.last_opened_folder = path
        self._last_update = time.time()

    def update_file(self, path: str) -> None:
        """Track the last opened file."""
        self.last_opened_file = path
        self._last_update = time.time()

    def record_action(self, action: str, result: str = "", success: bool = True) -> None:
        """Record an action and its result."""
        self.last_action = action
        self.last_result = result
        self._last_update = time.time()
        if not success:
            self.last_failure = f"{action}: {result}"

    def add_turn(self, user_text: str, response: str, action_taken: str = "", success: bool = True) -> None:
        """Add a conversation turn to history."""
        turn = ConversationTurn(
            timestamp=time.time(),
            user_text=user_text,
            response=response,
            action_taken=action_taken,
            success=success,
        )
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._last_update = time.time()

    def is_context_stale(self) -> bool:
        """Check if context has timed out (user moved to a new topic)."""
        return (time.time() - self._last_update) > self._context_timeout

    def resolve_followup(self, text: str) -> str:
        """
        Try to resolve a follow-up command by adding context from the session.

        For example:
          "open downloads" + current_app="explorer"
          -> "open downloads in file explorer"
        """
        text_lower = text.lower().strip()

        if self.current_app and self.current_app.lower() in text_lower:
            return text

        # Folder navigation follow-ups
        if self.current_app in ("explorer", "File Explorer"):
            folder_names = [
                "downloads", "documents", "desktop", "pictures",
                "music", "videos", "projects", "pengu",
            ]
            for folder in folder_names:
                if text_lower in (f"open {folder}", f"go to {folder}", folder):
                    return f"open {folder}"

        # Browser follow-ups
        if self.current_url:
            browser_apps = ("chrome", "edge", "firefox", "browser")
            if self.current_app in browser_apps or any(
                app in self.current_app.lower() for app in browser_apps
            ):
                if text_lower.startswith("search") or text_lower.startswith("go to"):
                    return text

        return text

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the current context."""
        return {
            "current_app": self.current_app,
            "current_url": self.current_url,
            "current_page_title": self.current_page_title,
            "current_directory": self.current_directory,
            "last_opened_folder": self.last_opened_folder,
            "last_opened_file": self.last_opened_file,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "last_failure": self.last_failure,
            "history_length": len(self.history),
            "context_stale": self.is_context_stale(),
        }

    def clear(self) -> None:
        """Clear all context (start fresh topic)."""
        self.current_app = ""
        self.current_app_pid = 0
        self.current_url = ""
        self.current_page_title = ""
        self.current_directory = ""
        self.last_opened_folder = ""
        self.last_opened_file = ""
        self.current_task = ""
        self.last_action = ""
        self.last_result = ""
        self.last_failure = ""
        self.history.clear()
        self._last_update = time.time()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_context: Optional[TaskContext] = None


def get_context() -> TaskContext:
    """Get the global task context."""
    global _context
    if _context is None:
        _context = TaskContext()
    return _context


def reset_context() -> TaskContext:
    """Reset the global context (for testing)."""
    global _context
    _context = TaskContext()
    return _context
