"""
Mission Manager — lifecycle management for autonomous agent missions.

A Mission is a user goal being worked on by the AgentBrain.
The MissionManager:
  - creates missions from user goals
  - runs the observe → think → act → verify loop
  - tracks mission progress
  - supports pause/resume/cancel
  - stores mission history
  - runs missions in the background when needed

Architecture:
  User says goal
    → MissionManager.create_mission(goal)
    → AgentBrain.understand(goal)
    → AgentBrain.observe() → WorldState
    → AgentBrain.plan() → Plan
    → for each step:
        AgentBrain.decide_next()
        → ACT: AgentBrain.act()
        → VERIFY: AgentBrain.observe()
        → RECOVER: AgentBrain.recover()
    → AgentBrain.generate_response()
    → response to user
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from pengu.agent.brain import AgentBrain
from pengu.agent.state import (
    AgentState,
    MissionStatus,
    PlanStep,
    StepStatus,
)
from pengu.logging import get_logger

logger = get_logger("pengu.agent.mission")


class MissionManager:
    """
    Manages the lifecycle of agent missions.

    Coordinates between:
    - AgentBrain (reasoning)
    - Observer (perception)
    - Tool executor (action)
    - Voice engine (user communication)
    """

    def __init__(self, brain: Optional[AgentBrain] = None) -> None:
        self._brain = brain or AgentBrain()
        self._missions: dict[str, AgentState] = {}
        self._current_mission: Optional[AgentState] = None
        self._running = False
        self._paused = False
        self._tool_executor: Optional[Callable] = None
        self._on_status_change: Optional[Callable] = None
        self._on_action_complete: Optional[Callable] = None

    def set_tool_executor(self, executor: Callable) -> None:
        """Set the function that executes tool actions."""
        self._tool_executor = executor

    def set_status_callback(self, callback: Callable) -> None:
        """Set callback for mission status changes."""
        self._on_status_change = callback

    def set_action_callback(self, callback: Callable) -> None:
        """Set callback for action completion."""
        self._on_action_complete = callback

    @property
    def current_mission(self) -> Optional[AgentState]:
        return self._current_mission

    @property
    def is_busy(self) -> bool:
        return self._running and not self._paused

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    async def execute_goal(self, goal: str, tool_executor: Optional[Callable] = None) -> str:
        """
        Execute a user goal end-to-end.

        This is the main entry point. It:
        1. Creates a mission
        2. Runs the observe → think → act → verify loop
        3. Returns the final response

        Args:
            goal: natural language goal from the user
            tool_executor: function(action, params) -> ActionResult

        Returns:
            Human-readable response string
        """
        executor = tool_executor or self._tool_executor
        if executor is None:
            logger.error("no_tool_executor")
            return "I don't have access to any tools right now."

        mission = self._create_mission(goal)
        self._current_mission = mission
        self._running = True
        self._paused = False

        try:
            response = await self._run_mission_loop(mission, executor)
            return response
        except Exception as e:
            mission.mark_failed(str(e))
            mission.add_error(f"Mission crashed: {e}")
            logger.error("mission_crashed", goal=goal[:60], error=str(e))
            return f"Something went wrong: {e}"
        finally:
            self._running = False
            self._paused = False

    def pause(self) -> None:
        """Pause the current mission."""
        if self._running and self._current_mission:
            self._paused = True
            self._current_mission.status = MissionStatus.WAITING
            logger.info("mission_paused", mission_id=self._current_mission.mission_id)

    def resume(self) -> None:
        """Resume a paused mission."""
        if self._paused and self._current_mission:
            self._paused = False
            self._current_mission.status = MissionStatus.RUNNING
            logger.info("mission_resumed", mission_id=self._current_mission.mission_id)

    def cancel(self) -> str:
        """Cancel the current mission."""
        if self._current_mission and self._running:
            self._current_mission.mark_cancelled()
            self._running = False
            self._paused = False
            response = "Task cancelled."
            logger.info("mission_cancelled", mission_id=self._current_mission.mission_id)
            return response
        return "No mission running."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_mission(self, goal: str) -> AgentState:
        """Create a new mission from a user goal."""
        state = AgentState(goal=goal)
        self._missions[state.mission_id] = state
        logger.info("mission_created", mission_id=state.mission_id, goal=goal[:80])
        self._notify_status(state, "created")
        return state

    async def _run_mission_loop(
        self, state: AgentState, executor: Callable,
    ) -> str:
        """
        Run the core observe → think → act → verify loop.

        This is the heart of the autonomous agent.
        Includes loop detection to prevent infinite repeated actions.
        """
        max_iterations = 30  # safety limit
        iteration = 0

        # Loop detection: track recent actions to detect stuck patterns
        _action_window: list[str] = []  # last N action signatures
        _MAX_REPEAT_THRESHOLD = 3  # if same action signature appears 3x in a row, break loop
        _consecutive_same_count = 0
        _last_action_sig = ""

        # Step 1: Understand the goal
        intent = self._brain.understand(state.goal, state)
        logger.info("mission_intent", type=intent["type"], risk=intent["risk_level"])

        # Step 2: Observe the world
        self._brain.observe(state)

        # Step 3: Create the plan
        steps = self._brain.plan(state, intent)
        if not steps:
            state.mark_completed("I'm not sure what to do for that.")
            return "I'm not sure what to do for that."

        state.mark_running()
        self._notify_status(state, "running")

        # Step 4: Execute the plan loop
        while iteration < max_iterations:
            if self._paused:
                await asyncio.sleep(0.5)
                continue

            if not self._running:
                state.mark_cancelled()
                return "Task cancelled."

            decision = self._brain.decide_next(state)
            logger.info("brain_decision", decision=decision.value, step=state.current_step_index)

            if decision.value == "plan":
                # Shouldn't happen if plan was created, but handle it
                intent = self._brain.understand(state.goal, state)
                self._brain.plan(state, intent)

            elif decision.value == "act":
                # Build action signature for loop detection
                current = state.current_step()
                action_sig = f"{current.action}:{current.target}" if current else f"unknown:{iteration}"

                # Check for repeated action pattern
                if action_sig == _last_action_sig:
                    _consecutive_same_count += 1
                else:
                    _consecutive_same_count = 1
                _last_action_sig = action_sig
                _action_window.append(action_sig)
                if len(_action_window) > 10:
                    _action_window.pop(0)

                if _consecutive_same_count >= _MAX_REPEAT_THRESHOLD:
                    logger.warning(
                        "loop_detected",
                        action=action_sig,
                        repeats=_consecutive_same_count,
                    )
                    # Try replanning instead of repeating
                    self._brain.replan(state, reason=f"loop detected: {action_sig} repeated {_consecutive_same_count} times")
                    _consecutive_same_count = 0
                    _last_action_sig = ""
                    if not state.plan:
                        state.mark_failed("Detected a loop — could not find an alternative approach.")
                        break
                    continue

                result = await self._brain.act(state, executor)
                if self._on_action_complete:
                    self._on_action_complete(state, result)

                # Observe after action to verify
                self._brain.observe(state)

                # Reset consecutive count on successful action
                if result.get("success"):
                    _consecutive_same_count = 0

            elif decision.value == "recover":
                recovered = self._brain.recover(state)
                if recovered is None:
                    # Skip failed step
                    state.advance_step()

            elif decision.value == "replan":
                self._brain.replan(state, reason="too many failures")
                _consecutive_same_count = 0
                _last_action_sig = ""
                if not state.plan:
                    state.mark_failed("Could not complete the task after multiple attempts.")
                    break

            elif decision.value == "complete":
                break

            elif decision.value == "clarify":
                state.mark_completed("I need more information to complete this task.")
                break

            elif decision.value == "wait":
                await asyncio.sleep(0.5)

            iteration += 1

        if iteration >= max_iterations:
            state.mark_failed("Task took too many steps.")
            logger.warning("mission_max_iterations", goal=state.goal[:60])

        # Generate final response
        response = self._brain.generate_response(state)
        if state.status not in (MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED):
            state.mark_completed(response)

        self._notify_status(state, state.status.value)
        logger.info(
            "mission_complete",
            mission_id=state.mission_id,
            status=state.status.value,
            steps=len(state.plan),
            response=response[:100],
        )
        return response

    def _notify_status(self, state: AgentState, event: str) -> None:
        """Notify listeners of status change."""
        if self._on_status_change:
            try:
                self._on_status_change(state, event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_mission_history(self) -> list[dict[str, Any]]:
        """Get summary of all missions."""
        return [
            {
                "id": m.mission_id,
                "goal": m.goal[:80],
                "status": m.status.value,
                "steps": len(m.plan),
                "response": m.final_response[:100],
            }
            for m in self._missions.values()
        ]

    def get_mission(self, mission_id: str) -> Optional[AgentState]:
        """Get a specific mission by ID."""
        return self._missions.get(mission_id)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_mission_manager: Optional[MissionManager] = None


def get_mission_manager(brain: Optional[AgentBrain] = None) -> MissionManager:
    """Get or create the global mission manager."""
    global _mission_manager
    if _mission_manager is None:
        _mission_manager = MissionManager(brain=brain)
    return _mission_manager


def reset_mission_manager() -> MissionManager:
    """Reset the global mission manager (for testing)."""
    global _mission_manager
    _mission_manager = MissionManager()
    return _mission_manager
