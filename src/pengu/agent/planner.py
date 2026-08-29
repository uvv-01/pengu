"""
Task Planner and Action Executor — multi-step task execution with verify.

Provides:
  - TaskPlan: ordered sequence of steps
  - TaskStep: individual step with action, target, verification
  - ActionExecutor: execute steps and verify results
  - Observe → Act → Verify loop for each step

Architecture:
  User command
    → TaskPlanner creates TaskPlan (list of steps)
    → ActionExecutor runs each step
    → Each step: act → verify → success/retry/fail
    → Final response to user
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from pengu.agent import ActionResult, ActionType, ActionStatus
from pengu.logging import get_logger

logger = get_logger("pengu.agent.planner")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TaskStep:
    """A single step in a task plan."""
    action: ActionType
    target: str = ""
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[ActionResult] = None
    retry_count: int = 0
    max_retries: int = 1
    timeout_seconds: float = 15.0
    depends_on: int = -1  # index of step this depends on (-1 = none)
    skip_if_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target": self.target,
            "description": self.description,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class TaskPlan:
    """An ordered sequence of steps to accomplish a user goal."""
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    current_step: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    response: str = ""

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    @property
    def all_succeeded(self) -> bool:
        return all(s.status == StepStatus.SUCCESS for s in self.steps)

    @property
    def any_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_failed_steps(self) -> list[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "is_complete": self.is_complete,
        }


class TaskPlanner:
    """
    Creates TaskPlan from user commands.

    Handles multi-step instructions like:
      "Open Chrome and search for Python"
      "Open ChatGPT, search for quantum computing"
      "Open Downloads, then open test.py"
    """

    def create_plan(self, text: str) -> TaskPlan:
        """Analyze text and create a task plan."""
        text_lower = text.lower().strip()

        # Multi-step: split on "and", "then", ";"
        steps_raw = self._split_steps(text)

        if len(steps_raw) > 1:
            return self._create_multi_step_plan(steps_raw)

        # Single step
        return self._create_single_step_plan(text)

    def _split_steps(self, text: str) -> list[str]:
        """Split a multi-step command into individual steps."""
        import re
        # Split on "and then", "and", "then", ";", ","
        # Be careful not to split "search for X and Y"
        parts = re.split(r'\s*;\s*|\s+then\s+|\s+and\s+(?:also\s+)?', text, flags=re.IGNORECASE)
        # Filter empty
        return [p.strip().rstrip(".") for p in parts if p.strip()]

    def _create_multi_step_plan(self, steps_raw: list[str]) -> TaskPlan:
        """Create a multi-step task plan."""
        plan = TaskPlan(goal=" + ".join(steps_raw))
        for i, step_text in enumerate(steps_raw):
            action_type, target, params = self._classify_step(step_text)
            plan.steps.append(TaskStep(
                action=action_type,
                target=target,
                description=step_text,
                params=params,
                depends_on=i - 1 if i > 0 else -1,
            ))
        return plan

    def _create_single_step_plan(self, text: str) -> TaskPlan:
        """Create a single-step task plan."""
        plan = TaskPlan(goal=text)
        action_type, target, params = self._classify_step(text)
        plan.steps.append(TaskStep(
            action=action_type,
            target=target,
            description=text,
            params=params,
        ))
        return plan

    def _classify_step(self, text: str) -> tuple[ActionType, str, dict[str, Any]]:
        """Classify a single step into action type, target, and params."""
        text_lower = text.lower().strip()

        # Navigation
        if any(text_lower.startswith(w) for w in ["open ", "launch ", "start "]):
            target = text_lower.split(None, 1)[1] if len(text_lower.split(None, 1)) > 1 else ""
            return ActionType.OPEN_APP, target, {"raw": text}

        if any(text_lower.startswith(w) for w in ["go to ", "navigate to "]):
            target = text_lower.split(None, 2)[2] if len(text_lower.split(None, 2)) > 2 else ""
            return ActionType.NAVIGATE, target, {"raw": text}

        if any(text_lower.startswith(w) for w in ["search for ", "search ", "google "]):
            query = text_lower.replace("search for ", "").replace("search ", "").replace("google ", "")
            return ActionType.SEARCH, query, {"raw": text}

        # Click
        if text_lower.startswith("click "):
            target = text_lower[6:].strip()
            return ActionType.CLICK, target, {"raw": text}

        # Type
        if text_lower.startswith("type "):
            target = text_lower[5:].strip()
            return ActionType.TYPE_TEXT, target, {"raw": text}

        # Read
        if any(text_lower.startswith(w) for w in ["read ", "what's on", "what is on"]):
            return ActionType.READ_PAGE, text, {"raw": text}

        # Default: treat as chat/unknown
        return ActionType.UNKNOWN, text, {"raw": text}


class ActionExecutor:
    """
    Executes TaskPlans step by step with observe → act → verify loop.

    Each step:
      1. OBSERVE current state
      2. ACT (perform the action)
      3. VERIFY the action succeeded
      4. If failed: retry or report failure
    """

    def __init__(self) -> None:
        self._desktop = None
        self._browser = None
        self._observer = None

    def _ensure_imports(self) -> None:
        if self._desktop is None:
            from pengu.agent.desktop import get_desktop
            self._desktop = get_desktop()
        if self._browser is None:
            from pengu.agent.browser_agent import get_browser_agent
            self._browser = get_browser_agent()
        if self._observer is None:
            from pengu.agent.observer import get_observer
            self._observer = get_observer()

    async def execute_plan(self, plan: TaskPlan) -> str:
        """Execute a task plan and return a summary response."""
        self._ensure_imports()
        plan.started_at = time.time()
        responses = []

        for i, step in enumerate(plan.steps):
            plan.current_step = i

            # Check dependency
            if step.depends_on >= 0:
                dep = plan.steps[step.depends_on]
                if dep.status != StepStatus.SUCCESS:
                    if step.skip_if_failed:
                        step.status = StepStatus.SKIPPED
                        continue
                    step.status = StepStatus.FAILED
                    step.result = ActionResult.fail(
                        f"Dependency failed: {dep.description}",
                        action=step.action,
                    )
                    responses.append(f"Skipping '{step.description}' because previous step failed.")
                    continue

            # Execute with retries
            step.status = StepStatus.RUNNING
            for attempt in range(step.max_retries + 1):
                step.retry_count = attempt
                try:
                    result = await self._execute_step(step)
                    step.result = result

                    if result.success:
                        step.status = StepStatus.SUCCESS
                        if result.message:
                            responses.append(result.message)
                        break
                    else:
                        if attempt < step.max_retries:
                            step.status = StepStatus.RETRYING
                            logger.info("step_retry", step=step.description, attempt=attempt + 1)
                            await asyncio.sleep(0.5)
                        else:
                            step.status = StepStatus.FAILED
                            responses.append(f"Failed: {result.message}")
                except Exception as e:
                    step.status = StepStatus.FAILED
                    step.result = ActionResult.fail(str(e), action=step.action)
                    responses.append(f"Error: {e}")
                    break

        plan.completed_at = time.time()
        plan.response = " ".join(responses) if responses else "Done."
        return plan.response

    async def _execute_step(self, step: TaskStep) -> ActionResult:
        """Execute a single task step."""
        action = step.action
        target = step.target

        try:
            if action == ActionType.OPEN_APP:
                return await self._exec_open_app(target)
            elif action == ActionType.NAVIGATE:
                return await self._exec_navigate(target)
            elif action == ActionType.SEARCH:
                return await self._exec_search(target)
            elif action == ActionType.CLICK:
                return await self._exec_click(target)
            elif action == ActionType.TYPE_TEXT:
                return await self._exec_type(target)
            elif action == ActionType.READ_PAGE:
                return await self._exec_read_page()
            else:
                return ActionResult.fail(
                    f"I don't know how to do: {step.description}",
                    action=action,
                    target=target,
                )
        except Exception as e:
            return ActionResult.fail(f"Error: {e}", action=action, target=target)

    async def _exec_open_app(self, target: str) -> ActionResult:
        """Open an application."""
        from pengu.os.app_launcher import get_launcher
        launcher = get_launcher()
        result = launcher.open_application(target)
        if result["success"]:
            return ActionResult.ok(
                result["message"],
                action=ActionType.OPEN_APP,
                target=target,
                verified=True,
            )
        return ActionResult.fail(result["message"], action=ActionType.OPEN_APP, target=target)

    async def _exec_navigate(self, target: str) -> ActionResult:
        """Navigate to a URL in the browser."""
        return await self._browser.navigate(target)

    async def _exec_search(self, target: str) -> ActionResult:
        """Search for something (Google or browser)."""
        # If browser is open with a page, search on that page
        if self._browser._page:
            page_info = await self._browser.get_page_info()
            if page_info.get("url"):
                # Try to find a search box and type
                return await self._browser.search_google(target)
        return await self._browser.search_google(target)

    async def _exec_click(self, target: str) -> ActionResult:
        """Click an element."""
        # Try browser first if a page is open
        if self._browser._page:
            return await self._browser.find_and_click(target)
        # Fallback to desktop automation
        return ActionResult.fail(
            f"Could not find '{target}' to click",
            action=ActionType.CLICK,
            target=target,
        )

    async def _exec_type(self, target: str) -> ActionResult:
        """Type text into the active window."""
        if self._browser._page:
            return await self._browser.type_in_page(target)
        self._desktop.keyboard.type_text(target)
        return ActionResult.ok(f"Typed '{target}'", action=ActionType.TYPE_TEXT, target=target)

    async def _exec_read_page(self) -> ActionResult:
        """Read the current page content."""
        return await self._browser.read_page()


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_planner: Optional[TaskPlanner] = None
_executor: Optional[ActionExecutor] = None


def get_planner() -> TaskPlanner:
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner


def get_executor() -> ActionExecutor:
    global _executor
    if _executor is None:
        _executor = ActionExecutor()
    return _executor
