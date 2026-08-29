"""
Action Result — structured result model for all desktop/browser actions.

Every action in Pengu returns an ActionResult so the assistant can:
  - verify success/failure
  - recover from failures
  - report accurate status to the user
  - never silently fail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionStatus(str, Enum):
    """Status of an executed action."""
    SUCCESS = "success"
    PARTIAL = "partial"       # succeeded but with caveats
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    NOT_AVAILABLE = "not_available"
    CANCELLED = "cancelled"
    RETRY_NEEDED = "retry_needed"


class ActionType(str, Enum):
    """Types of actions Pengu can perform."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    SCROLL = "scroll"
    OPEN_APP = "open_app"
    OPEN_URL = "open_url"
    OPEN_FOLDER = "open_folder"
    NAVIGATE = "navigate"
    SEARCH = "search"
    READ_PAGE = "read_page"
    SCREENSHOT = "screenshot"
    FOCUS_WINDOW = "focus_window"
    FIND_ELEMENT = "find_element"
    WAIT = "wait"
    VERIFY = "verify"
    SPEAK = "speak"
    CHAT = "chat"
    GIT = "git"
    FILESYSTEM = "filesystem"
    UNKNOWN = "unknown"


@dataclass
class ActionResult:
    """
    Structured result from any action execution.

    Every tool/action in Pengu should return this or a compatible object.
    """
    success: bool
    status: ActionStatus = ActionStatus.SUCCESS
    action: ActionType = ActionType.UNKNOWN
    target: str = ""
    message: str = ""
    output: Any = None
    error: str = ""
    error_code: str = ""
    verified: bool = False
    duration_ms: float = 0
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "action": self.action.value,
            "target": self.target,
            "message": self.message,
            "error": self.error,
            "error_code": self.error_code,
            "verified": self.verified,
            "duration_ms": round(self.duration_ms, 1),
        }

    @staticmethod
    def ok(
        message: str,
        action: ActionType = ActionType.UNKNOWN,
        target: str = "",
        verified: bool = True,
        **kwargs: Any,
    ) -> ActionResult:
        """Create a successful result."""
        return ActionResult(
            success=True,
            status=ActionStatus.SUCCESS,
            action=action,
            target=target,
            message=message,
            verified=verified,
            **kwargs,
        )

    @staticmethod
    def fail(
        message: str,
        error_code: str = "",
        action: ActionType = ActionType.UNKNOWN,
        target: str = "",
        **kwargs: Any,
    ) -> ActionResult:
        """Create a failure result."""
        return ActionResult(
            success=False,
            status=ActionStatus.FAILED,
            action=action,
            target=target,
            message=message,
            error=message,
            error_code=error_code,
            verified=False,
            **kwargs,
        )

    @staticmethod
    def timeout(
        message: str,
        action: ActionType = ActionType.UNKNOWN,
        target: str = "",
        **kwargs: Any,
    ) -> ActionResult:
        """Create a timeout result."""
        return ActionResult(
            success=False,
            status=ActionStatus.TIMEOUT,
            action=action,
            target=target,
            message=message,
            error=message,
            error_code="TIMEOUT",
            verified=False,
            **kwargs,
        )

    @staticmethod
    def not_found(
        message: str,
        action: ActionType = ActionType.UNKNOWN,
        target: str = "",
        **kwargs: Any,
    ) -> ActionResult:
        """Create a not-found result."""
        return ActionResult(
            success=False,
            status=ActionStatus.NOT_FOUND,
            action=action,
            target=target,
            message=message,
            error=message,
            error_code="ELEMENT_NOT_FOUND",
            verified=False,
            **kwargs,
        )
