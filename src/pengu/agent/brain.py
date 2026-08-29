"""
Agent Brain — the reasoning core of Pengu's autonomous agent.

The AgentBrain:
  1. Receives a user goal (natural language)
  2. Understands what the user wants
  3. Observes the current computer state
  4. Creates a dynamic plan
  5. Executes actions through controlled tools
  6. Observes results
  7. Verifies success
  8. Recovers from failures
  9. Re-plans when needed
  10. Reports the final result

The AgentBrain does NOT directly manipulate the OS.
It selects actions from a controlled tool set.

Architecture:
  User Goal
    → understand()
    → observe() → WorldState
    → plan() → list[PlanStep]
    → for each step:
        act(step) → ActionResult
        observe() → verify
        if failed → recover/replan
    → respond()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from pengu.agent.state import (
    AgentDecision,
    AgentState,
    MissionStatus,
    Observation,
    PlanStep,
    StepStatus,
    WorldState,
)
from pengu.logging import get_logger

logger = get_logger("pengu.agent.brain")


class AgentBrain:
    """
    The reasoning core that transforms goals into actions.

    The brain does NOT execute actions directly.
    It produces decisions that the MissionManager carries out.
    """

    def __init__(self, model_provider=None, tool_registry=None) -> None:
        self._provider = model_provider
        self._tools = tool_registry
        self._observer = None  # lazy init
        self._desktop = None   # lazy init
        self._browser = None   # lazy init

    def _ensure_observer(self):
        if self._observer is None:
            from pengu.agent.observer import get_observer
            self._observer = get_observer()
        return self._observer

    def _ensure_desktop(self):
        if self._desktop is None:
            from pengu.agent.desktop import get_desktop
            self._desktop = get_desktop()
        return self._desktop

    def _ensure_browser(self):
        if self._browser is None:
            from pengu.agent.browser_agent import get_browser_agent
            self._browser = get_browser_agent()
        return self._browser

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def understand(self, goal: str, state: AgentState) -> dict[str, Any]:
        """
        Analyze the user goal and extract structured intent.

        Returns:
          {
            "goal": str,           # cleaned goal
            "type": str,           # "simple", "multi_step", "research", "file_task", "info_query"
            "steps_hint": list,    # suggested decomposition
            "requires_browser": bool,
            "requires_desktop": bool,
            "requires_filesystem": bool,
            "risk_level": str,     # "low", "medium", "high"
          }
        """
        goal_lower = goal.lower().strip()

        result = {
            "goal": goal,
            "type": "simple",
            "steps_hint": [],
            "requires_browser": False,
            "requires_desktop": False,
            "requires_filesystem": False,
            "risk_level": "low",
        }

        # Detect multi-step
        multi_step_markers = [" and ", " then ", ";", "after that", "next"]
        if any(marker in goal_lower for marker in multi_step_markers):
            result["type"] = "multi_step"

        # Detect browser needs
        browser_keywords = [
            "search", "google", "browse", "website", "url", "http",
            "chatgpt", "github", "youtube", "open", "navigate", "click",
            "read page", "scroll", "first result", "find online",
        ]
        if any(kw in goal_lower for kw in browser_keywords):
            result["requires_browser"] = True

        # Detect desktop needs
        desktop_keywords = ["click", "press", "type", "scroll", "focus", "window"]
        if any(kw in goal_lower for kw in desktop_keywords):
            result["requires_desktop"] = True

        # Detect filesystem needs
        fs_keywords = ["file", "folder", "directory", "create", "read", "write", "list", "open folder"]
        if any(kw in goal_lower for kw in fs_keywords):
            result["requires_filesystem"] = True

        # Detect research/info queries
        research_keywords = ["find out", "research", "latest", "what is", "tell me", "summarize", "check"]
        if any(kw in goal_lower for kw in research_keywords):
            result["type"] = "research"
            result["requires_browser"] = True

        # Risk assessment
        high_risk_keywords = ["delete", "remove", "uninstall", "format", "rm", "rmdir"]
        medium_risk_keywords = ["install", "modify", "change", "update", "git push"]
        if any(kw in goal_lower for kw in high_risk_keywords):
            result["risk_level"] = "high"
        elif any(kw in goal_lower for kw in medium_risk_keywords):
            result["risk_level"] = "medium"

        logger.info(
            "goal_understood",
            goal=goal[:80],
            type=result["type"],
            risk=result["risk_level"],
            browser=result["requires_browser"],
        )
        return result

    def observe(self, state: AgentState) -> Observation:
        """
        Observe the current computer state.

        Returns an Observation with information about:
        - Active window
        - Browser state
        - Screen info
        - Available UI elements
        """
        observer = self._ensure_observer()
        desktop = self._ensure_desktop()

        obs = Observation(timestamp=time.time())

        try:
            active = observer.get_active_window()
            obs.active_window_title = active.get("title", "")
            obs.active_app = active.get("app", "")
            obs.raw_data["active_window_rect"] = (
                int(active.get("x", 0)),
                int(active.get("y", 0)),
                int(active.get("width", 0)),
                int(active.get("height", 0)),
            )
        except Exception as e:
            logger.warning("observe_active_window_failed", error=str(e))

        try:
            sw, sh = desktop.window.get_screen_size()
            obs.raw_data["screen_width"] = sw
            obs.raw_data["screen_height"] = sh
        except Exception:
            pass

        # Check browser state if agent has a browser
        try:
            browser = self._ensure_browser()
            if browser._page:
                info = asyncio.get_event_loop().run_until_complete(
                    browser.get_page_info()
                )
                obs.browser_url = info.get("url", "")
                obs.browser_title = info.get("title", "")
        except Exception:
            pass

        # Get UI elements if available
        try:
            elements = observer.get_elements()
            obs.visible_elements = [e.name for e in elements[:15] if e.name]
        except Exception:
            pass

        state.add_observation(obs)
        logger.info(
            "observation",
            app=obs.active_app[:30],
            window=obs.active_window_title[:50],
            elements=len(obs.visible_elements),
        )
        return obs

    def plan(self, state: AgentState, intent: dict[str, Any]) -> list[PlanStep]:
        """
        Generate a dynamic plan based on the goal and current state.

        The plan is NOT hardcoded — it's generated from the goal analysis
        and the current world state.
        """
        state.mark_planning()
        goal = intent["goal"].lower().strip()
        world = state.world
        steps: list[PlanStep] = []
        step_id = 0

        goal_type = intent.get("type", "simple")

        if goal_type == "multi_step":
            steps = self._plan_multi_step(goal, world, state)
        elif intent.get("requires_browser"):
            steps = self._plan_browser_task(goal, world, state)
        elif intent.get("requires_filesystem"):
            steps = self._plan_file_task(goal, world, state)
        else:
            steps = self._plan_simple(goal, world, state)

        for step in steps:
            state.add_step(step)

        logger.info(
            "plan_created",
            goal=goal[:60],
            steps=len(steps),
            step_descriptions=[s.description for s in steps],
        )
        return steps

    def decide_next(self, state: AgentState) -> AgentDecision:
        """
        Decide what to do next based on current state.

        This is the core decision loop:
        - If no plan → PLAN
        - If plan exists and current step done → advance or COMPLETE
        - If current step failed → RECOVER
        - If too many failures → REPLAN or FAIL
        - Otherwise → ACT
        """
        # No plan yet?
        if not state.plan:
            return AgentDecision.PLAN

        # Check if all done
        if state.is_plan_complete():
            if state.all_steps_succeeded():
                return AgentDecision.COMPLETE
            else:
                return AgentDecision.RECOVER

        current = state.current_step()
        if current is None:
            return AgentDecision.COMPLETE

        # Step completed successfully → advance
        if current.status == StepStatus.SUCCESS:
            state.advance_step()
            if state.is_plan_complete():
                return AgentDecision.COMPLETE
            return AgentDecision.ACT

        # Step failed → recover
        if current.status == StepStatus.FAILED:
            if current.retry_count < current.max_retries:
                return AgentDecision.RECOVER
            else:
                # Skip failed step and move on
                current.status = StepStatus.SKIPPED
                state.advance_step()
                if state.is_plan_complete():
                    return AgentDecision.COMPLETE
                return AgentDecision.ACT

        # Step pending or running → act
        if current.status in (StepStatus.PENDING, StepStatus.RUNNING):
            return AgentDecision.ACT

        # Too many consecutive failures → replan
        if state.consecutive_failures >= state.max_consecutive_failures:
            return AgentDecision.REPLAN

        return AgentDecision.ACT

    async def act(self, state: AgentState, tool_executor) -> dict[str, Any]:
        """
        Execute the current step's action through the tool executor.

        Args:
            state: current agent state
            tool_executor: callable that executes tools, e.g. (action, params) -> ActionResult

        Returns:
            Action result dict with success, message, etc.
        """
        current = state.current_step()
        if current is None:
            return {"success": False, "error": "No current step"}

        current.status = StepStatus.RUNNING
        current.started_at = time.time()

        logger.info(
            "action_executing",
            step=current.step_id,
            action=current.action,
            target=current.target[:50],
        )

        try:
            result = await tool_executor(current.action, current.params)
            current.completed_at = time.time()
            duration_ms = (current.completed_at - current.started_at) * 1000

            if hasattr(result, 'success'):
                current.result_success = result.success
                current.result_message = getattr(result, 'message', '') or getattr(result, 'output', '')
                current.error = getattr(result, 'error', '')
            elif isinstance(result, dict):
                current.result_success = result.get("success", False)
                current.result_message = result.get("message", result.get("output", ""))
                current.error = result.get("error", "")
            else:
                current.result_success = bool(result)
                current.result_message = str(result)

            if current.result_success:
                current.status = StepStatus.SUCCESS
                state.record_action(
                    current.action, current.target, True,
                    current.result_message,
                )
            else:
                current.status = StepStatus.FAILED
                current.retry_count += 1
                state.record_action(
                    current.action, current.target, False,
                    error=current.error,
                )

            logger.info(
                "action_completed",
                step=current.step_id,
                action=current.action,
                success=current.result_success,
                message=current.result_message[:80] if current.result_message else "",
                duration_ms=round(duration_ms),
            )

            return {
                "success": current.result_success,
                "message": current.result_message,
                "error": current.error,
                "duration_ms": round(duration_ms),
            }

        except Exception as e:
            current.completed_at = time.time()
            current.status = StepStatus.FAILED
            current.error = str(e)
            current.retry_count += 1
            state.record_action(current.action, current.target, False, error=str(e))
            state.add_error(f"Action failed: {current.action} — {e}")
            logger.error("action_error", step=current.step_id, error=str(e))
            return {"success": False, "error": str(e)}

    def recover(self, state: AgentState) -> Optional[PlanStep]:
        """
        Attempt to recover from a failed step.

        Strategy:
        1. Retry the same step (if retries remaining)
        2. Try an alternative approach
        3. Skip and continue
        """
        current = state.current_step()
        if current is None:
            return None

        if current.retry_count < current.max_retries:
            current.status = StepStatus.PENDING
            current.error = ""
            logger.info(
                "recovery_retry",
                step=current.step_id,
                attempt=current.retry_count + 1,
            )
            return current

        # Max retries exceeded — mark as skipped
        current.status = StepStatus.SKIPPED
        state.record_action(
            current.action, current.target, False,
            error="Max retries exceeded, skipping",
        )
        logger.info("recovery_skip", step=current.step_id)
        return None

    def replan(self, state: AgentState, reason: str = "") -> list[PlanStep]:
        """
        Generate a new plan after failures.

        Keeps completed steps, generates new steps for remaining work.
        """
        logger.info("replan", reason=reason[:100], completed=state.current_step_index)

        # Get remaining goal from uncompleted steps
        remaining_steps = [
            s for s in state.plan[state.current_step_index:]
            if s.status != StepStatus.SUCCESS
        ]
        remaining_goal = " ".join(s.description for s in remaining_steps) if remaining_steps else state.goal

        # Reset plan to only remaining work
        state.plan = state.plan[:state.current_step_index]
        state.current_step_index = 0
        state.consecutive_failures = 0

        # Generate new plan for remaining work
        intent = self.understand(remaining_goal, state)
        new_steps = self.plan(state, intent)
        return new_steps

    def generate_response(self, state: AgentState) -> str:
        """
        Generate a human-readable response summarizing what happened.
        """
        if state.status == MissionStatus.COMPLETED:
            if state.all_steps_succeeded():
                messages = [
                    s.result_message for s in state.plan
                    if s.result_message
                ]
                if messages:
                    return " ".join(messages[-3:])  # last 3 messages
                return "Done."
            else:
                succeeded = sum(1 for s in state.plan if s.status == StepStatus.SUCCESS)
                total = len(state.plan)
                return f"Completed {succeeded} of {total} steps."

        elif state.status == MissionStatus.FAILED:
            failed = state.failed_steps()
            if failed:
                return f"I couldn't complete: {failed[0].description}. {failed[0].error}"
            return "I wasn't able to complete that task."

        elif state.status == MissionStatus.CANCELLED:
            return "Task cancelled."

        return "Task finished."

    # ------------------------------------------------------------------
    # Planning methods
    # ------------------------------------------------------------------

    def _plan_simple(self, goal: str, world: WorldState, state: AgentState) -> list[PlanStep]:
        """Plan a simple single-action task."""
        goal_lower = goal.lower().strip()
        steps: list[PlanStep] = []

        # Application opening
        if any(goal_lower.startswith(w) for w in ["open ", "launch ", "start "]):
            target = goal_lower.split(None, 1)[1] if len(goal_lower.split(None, 1)) > 1 else ""
            steps.append(PlanStep(
                step_id=0,
                action="open_app",
                target=target,
                description=f"Open {target}",
                params={"app": target},
            ))
        # URL/website opening
        elif any(goal_lower.startswith(w) for w in ["go to ", "navigate to ", "open "]):
            target = goal_lower.split(None, 2)[-1] if len(goal_lower.split(None, 2)) > 2 else goal_lower.split(None, 1)[-1]
            steps.append(PlanStep(
                step_id=0,
                action="navigate",
                target=target,
                description=f"Navigate to {target}",
                params={"url": target},
            ))
        # Search
        elif any(goal_lower.startswith(w) for w in ["search ", "google ", "look up ", "find "]):
            query = goal_lower
            for prefix in ["search for ", "search ", "google for ", "google ", "look up ", "find ", "find me "]:
                if goal_lower.startswith(prefix):
                    query = goal_lower[len(prefix):]
                    break
            steps.append(PlanStep(
                step_id=0,
                action="web_search",
                target=query,
                description=f"Search for {query}",
                params={"query": query},
            ))
        # Click
        elif goal_lower.startswith("click "):
            target = goal_lower[6:].strip()
            steps.append(PlanStep(
                step_id=0,
                action="click",
                target=target,
                description=f"Click {target}",
                params={"text": target},
            ))
        # Default: send to LLM as chat
        else:
            steps.append(PlanStep(
                step_id=0,
                action="chat",
                target=goal,
                description=goal,
                params={"text": goal},
            ))

        return steps

    def _plan_multi_step(self, goal: str, world: WorldState, state: AgentState) -> list[PlanStep]:
        """Plan a multi-step task by splitting on connectors."""
        import re
        # Split on "and", "then", ";"
        parts = re.split(r'\s+and\s+(?:also\s+)?|\s+then\s+|\s*;\s*', goal, flags=re.IGNORECASE)
        parts = [p.strip().rstrip(".") for p in parts if p.strip()]

        steps: list[PlanStep] = []
        for i, part in enumerate(parts):
            sub_steps = self._plan_simple(part, world, state)
            for j, step in enumerate(sub_steps):
                step.step_id = len(steps)
                step.depends_on = len(steps) - 1 if len(steps) > 0 else -1
                steps.append(step)

        return steps

    def _plan_browser_task(self, goal: str, world: WorldState, state: AgentState) -> list[PlanStep]:
        """Plan a task that requires browser interaction."""
        goal_lower = goal.lower().strip()
        steps: list[PlanStep] = []
        step_id = 0

        # Check if browser is already open
        browser_already_open = world.is_browser_active() or world.browser_open

        # Detect if the goal involves opening a specific site
        site_open_patterns = [
            (r"open\s+(google|chrome)", "google", "https://www.google.com"),
            (r"open\s+(github)", "github", "https://github.com"),
            (r"open\s+(youtube)", "youtube", "https://www.youtube.com"),
            (r"open\s+(chatgpt|chat\s*gpt)", "chatgpt", "https://chatgpt.com"),
            (r"open\s+(stackoverflow|stack\s*overflow)", "stackoverflow", "https://stackoverflow.com"),
        ]

        import re
        for pattern, name, url in site_open_patterns:
            if re.search(pattern, goal_lower):
                if not (world.browser_url and name in world.browser_url.lower()):
                    steps.append(PlanStep(
                        step_id=step_id, action="navigate", target=url,
                        description=f"Open {name}",
                        params={"url": url},
                    ))
                    step_id += 1
                break

        # Detect search query
        search_match = re.search(
            r"(?:search|google|look\s+up|find)\s+(?:for\s+|on\s+)?(.+)",
            goal_lower,
        )
        if search_match:
            query = search_match.group(1).strip().rstrip(".")
            if query:
                steps.append(PlanStep(
                    step_id=step_id, action="web_search", target=query,
                    description=f"Search for {query}",
                    params={"query": query},
                ))
                step_id += 1

        # Detect click
        click_match = re.search(r"click\s+(?:the\s+)?(.+)", goal_lower)
        if click_match:
            target = click_match.group(1).strip()
            steps.append(PlanStep(
                step_id=step_id, action="click", target=target,
                description=f"Click {target}",
                params={"text": target},
            ))
            step_id += 1

        # Detect read
        if any(w in goal_lower for w in ["read", "what's on", "what is on", "tell me about"]):
            steps.append(PlanStep(
                step_id=step_id, action="read_page", target="",
                description="Read page content",
                params={},
            ))
            step_id += 1

        # If no specific steps were generated, fall back to simple plan
        if not steps:
            steps = self._plan_simple(goal, world, state)

        return steps

    def _plan_file_task(self, goal: str, world: WorldState, state: AgentState) -> list[PlanStep]:
        """Plan a filesystem task."""
        goal_lower = goal.lower().strip()
        steps: list[PlanStep] = []

        # Common folder opening
        shell_folders = {
            "downloads": "downloads", "documents": "documents",
            "desktop": "desktop", "pictures": "pictures",
            "music": "music", "videos": "videos",
        }

        import re
        for folder_name, _ in shell_folders.items():
            if re.search(rf"open\s+{folder_name}", goal_lower):
                steps.append(PlanStep(
                    step_id=0, action="open_folder", target=folder_name,
                    description=f"Open {folder_name}",
                    params={"folder": folder_name},
                ))
                return steps

        # File/folder creation
        create_match = re.search(
            r"create\s+(?:a\s+)?(?:file|folder|directory)\s+(?:called?|named?)?\s*(.+?)(?:\s+in\s+|$)",
            goal_lower,
        )
        if create_match:
            name = create_match.group(1).strip()
            is_folder = "folder" in goal_lower or "directory" in goal_lower
            steps.append(PlanStep(
                step_id=0,
                action="create_file" if not is_folder else "create_folder",
                target=name,
                description=f"Create {name}",
                params={"name": name, "is_folder": is_folder},
            ))
            return steps

        # List files
        if any(w in goal_lower for w in ["list", "show files", "what's in"]):
            steps.append(PlanStep(
                step_id=0, action="list_files", target="",
                description="List files", params={},
            ))
            return steps

        # Fallback to simple plan
        return self._plan_simple(goal, world, state)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_brain: Optional[AgentBrain] = None


def get_brain(model_provider=None, tool_registry=None) -> AgentBrain:
    """Get or create the global agent brain."""
    global _brain
    if _brain is None:
        _brain = AgentBrain(model_provider=model_provider, tool_registry=tool_registry)
    return _brain


def reset_brain() -> AgentBrain:
    """Reset the global agent brain (for testing)."""
    global _brain
    _brain = AgentBrain()
    return _brain
