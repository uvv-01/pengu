"""
Agent State and World State — structured models for autonomous agent operation.

AgentState tracks the current mission, goal, plan, and execution history.
WorldState represents what Pengu currently knows about the computer.

The AgentBrain uses both to reason about what to do next.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MissionStatus(str, Enum):
    """Status of a mission."""
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentDecision(str, Enum):
    """What the agent brain decided to do next."""
    PLAN = "plan"               # Generate a plan
    ACT = "act"                 # Execute the next action
    VERIFY = "verify"           # Verify the last action
    RECOVER = "recover"         # Recover from a failure
    REPLAN = "replan"           # Re-plan based on new observations
    COMPLETE = "complete"       # Mission is complete
    CLARIFY = "clarify"         # Need user clarification
    WAIT = "wait"               # Waiting for external state


class StepStatus(str, Enum):
    """Status of a single plan step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Plan Step
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in the agent's plan."""
    step_id: int
    action: str                  # tool/action name
    target: str = ""             # what to act on
    description: str = ""        # human-readable description
    params: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result_message: str = ""
    result_success: bool = False
    error: str = ""
    retry_count: int = 0
    max_retries: int = 1
    started_at: float = 0.0
    completed_at: float = 0.0
    depends_on: int = -1  # index of step this depends on (-1 = none)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "target": self.target,
            "description": self.description,
            "status": self.status.value,
            "success": self.result_success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """What the agent observed about the world state."""
    timestamp: float = 0.0
    active_window_title: str = ""
    active_app: str = ""
    browser_url: str = ""
    browser_title: str = ""
    visible_elements: list[str] = field(default_factory=list)
    screen_summary: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "active_app": self.active_app,
            "active_window": self.active_window_title[:80],
            "browser_url": self.browser_url,
            "visible_elements": self.visible_elements[:10],
            "screen_summary": self.screen_summary[:200],
        }


# ---------------------------------------------------------------------------
# World State
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """
    What Pengu currently knows about the computer.

    Updated after each observation. Used by the AgentBrain to reason
    about what to do next.
    """
    # Active application
    active_app: str = ""
    active_window_title: str = ""
    active_window_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    # Browser state
    browser_open: bool = False
    browser_url: str = ""
    browser_title: str = ""

    # Desktop state
    screen_width: int = 0
    screen_height: int = 0

    # Filesystem context
    current_directory: str = ""
    last_opened_folder: str = ""
    last_opened_file: str = ""

    # Available tools (cached from registry)
    available_tools: list[str] = field(default_factory=list)

    # Timestamps
    last_observed: float = 0.0

    def update_from_observation(self, obs: Observation) -> None:
        """Update world state from a new observation."""
        self.active_app = obs.active_app
        self.active_window_title = obs.active_window_title
        if obs.raw_data:
            rect = obs.raw_data.get("active_window_rect")
            if rect and len(rect) == 4:
                self.active_window_rect = tuple(rect)
            self.screen_width = int(obs.raw_data.get("screen_width", 0))
            self.screen_height = int(obs.raw_data.get("screen_height", 0))
        if obs.browser_url:
            self.browser_url = obs.browser_url
            self.browser_open = True
        if obs.browser_title:
            self.browser_title = obs.browser_title
        self.last_observed = obs.timestamp

    def is_browser_active(self) -> bool:
        """Check if a browser is the active application."""
        browser_names = ("chrome", "edge", "firefox", "browser", "opera", "vivaldi")
        return any(name in self.active_app.lower() for name in browser_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_app": self.active_app,
            "active_window": self.active_window_title[:80],
            "browser_open": self.browser_open,
            "browser_url": self.browser_url,
            "browser_title": self.browser_title[:60],
            "screen": f"{self.screen_width}x{self.screen_height}",
            "current_directory": self.current_directory,
            "last_observed": self.last_observed,
        }


# ---------------------------------------------------------------------------
# Agent State (mission-level)
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    Structured state for the current agent mission.

    Contains everything the AgentBrain needs to reason about:
    - the goal
    - the current plan
    - execution progress
    - observations
    - history
    """
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    status: MissionStatus = MissionStatus.PENDING
    created_at: float = field(default_factory=time.time)

    # Plan
    plan: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0

    # World state (updated by observations)
    world: WorldState = field(default_factory=WorldState)

    # Observations history
    observations: list[Observation] = field(default_factory=list)
    max_observations: int = 20

    # Execution history
    action_history: list[dict[str, Any]] = field(default_factory=list)
    max_action_history: int = 50

    # Error tracking
    errors: list[str] = field(default_factory=list)
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5

    # Response
    final_response: str = ""

    # --- Plan helpers ---

    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.plan.append(step)

    def current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    def advance_step(self) -> bool:
        """Move to the next step. Returns True if there are more steps."""
        self.current_step_index += 1
        return self.current_step_index < len(self.plan)

    def is_plan_complete(self) -> bool:
        """Check if all plan steps have been executed."""
        return self.current_step_index >= len(self.plan)

    def all_steps_succeeded(self) -> bool:
        """Check if every step succeeded."""
        return all(s.status == StepStatus.SUCCESS for s in self.plan)

    def has_failures(self) -> bool:
        """Check if any step failed."""
        return any(s.status == StepStatus.FAILED for s in self.plan)

    def failed_steps(self) -> list[PlanStep]:
        """Get list of failed steps."""
        return [s for s in self.plan if s.status == StepStatus.FAILED]

    # --- Observation helpers ---

    def add_observation(self, obs: Observation) -> None:
        """Record an observation and update world state."""
        self.observations.append(obs)
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]
        self.world.update_from_observation(obs)

    def latest_observation(self) -> Optional[Observation]:
        """Get the most recent observation."""
        return self.observations[-1] if self.observations else None

    # --- Action history ---

    def record_action(self, action: str, target: str, success: bool,
                      message: str = "", error: str = "") -> None:
        """Record an executed action."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "target": target,
            "success": success,
            "message": message[:200],
            "error": error[:200],
        }
        self.action_history.append(entry)
        if len(self.action_history) > self.max_action_history:
            self.action_history = self.action_history[-self.max_action_history:]
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    # --- Error tracking ---

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"[{time.strftime('%H:%M:%S')}] {error}")
        if len(self.errors) > 20:
            self.errors = self.errors[-20:]

    # --- State transitions ---

    def mark_planning(self) -> None:
        self.status = MissionStatus.PLANNING

    def mark_running(self) -> None:
        self.status = MissionStatus.RUNNING

    def mark_completed(self, response: str = "") -> None:
        self.status = MissionStatus.COMPLETED
        self.final_response = response

    def mark_failed(self, reason: str = "") -> None:
        self.status = MissionStatus.FAILED
        self.final_response = reason

    def mark_cancelled(self) -> None:
        self.status = MissionStatus.CANCELLED

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "status": self.status.value,
            "plan_steps": len(self.plan),
            "current_step": self.current_step_index,
            "world": self.world.to_dict(),
            "observations": len(self.observations),
            "action_history": len(self.action_history),
            "errors": len(self.errors),
            "consecutive_failures": self.consecutive_failures,
            "final_response": self.final_response[:200],
        }
