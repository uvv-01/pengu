"""
Core state machine for Pengu.

Central interaction model:

  STANDBY → WAKE_DETECTED → ACTIVE → LISTENING → THINKING →
  PLANNING → EXECUTING → SPEAKING → COMPLETE → STANDBY

Supports:
  - State transitions with validation
  - Current task context
  - Interrupt support (Ctrl+Shift+P)
  - Error recovery
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from pengu.config import AssistantState
from pengu.logging import AuditLogger, get_logger, get_task_id, new_task_id

logger = get_logger("pengu.state")
audit = AuditLogger()

# Valid state transitions
VALID_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.STANDBY: {AssistantState.WAKE_DETECTED, AssistantState.ACTIVE},
    AssistantState.WAKE_DETECTED: {AssistantState.ACTIVE, AssistantState.STANDBY},
    AssistantState.ACTIVE: {AssistantState.LISTENING, AssistantState.STANDBY, AssistantState.ERROR},
    AssistantState.LISTENING: {AssistantState.THINKING, AssistantState.STANDBY, AssistantState.ERROR},
    AssistantState.THINKING: {AssistantState.PLANNING, AssistantState.EXECUTING, AssistantState.ERROR, AssistantState.STANDBY},
    AssistantState.PLANNING: {AssistantState.EXECUTING, AssistantState.ERROR, AssistantState.STANDBY},
    AssistantState.EXECUTING: {
        AssistantState.THINKING,   # need more reasoning
        AssistantState.WAITING_CONFIRMATION,
        AssistantState.SPEAKING,
        AssistantState.COMPLETE,
        AssistantState.ERROR,
        AssistantState.STANDBY,
    },
    AssistantState.WAITING_CONFIRMATION: {AssistantState.EXECUTING, AssistantState.STANDBY, AssistantState.ERROR},
    AssistantState.SPEAKING: {AssistantState.COMPLETE, AssistantState.STANDBY, AssistantState.ERROR},
    AssistantState.ERROR: {AssistantState.STANDBY},
    AssistantState.COMPLETE: {AssistantState.STANDBY},
}


class StateError(Exception):
    """Invalid state transition."""


class AssistantStateMachine:
    """
    Manages the lifecycle of a single Pengu interaction.

    Usage:
        sm = StateMachine()
        await sm.transition(AssistantState.WAKE_DETECTED)
        await sm.transition(AssistantState.ACTIVE)
        ...
        await sm.complete()
    """

    def __init__(self) -> None:
        self._state: AssistantState = AssistantState.STANDBY
        self._task_id: str = ""
        self._started_at: float = 0
        self._transition_history: list[tuple[float, AssistantState, AssistantState]] = []
        self._cancelled: bool = False
        self._context: dict[str, Any] = {}

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def is_active(self) -> bool:
        return self._state not in (
            AssistantState.STANDBY,
            AssistantState.COMPLETE,
            AssistantState.ERROR,
        )

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)

    async def transition(self, new_state: AssistantState) -> None:
        """Attempt a state transition."""
        old_state = self._state

        if new_state not in VALID_TRANSITIONS.get(old_state, set()):
            raise StateError(
                f"Invalid transition: {old_state.value} → {new_state.value}"
            )

        self._state = new_state
        now = time.time()
        self._transition_history.append((now, old_state, new_state))

        # Generate task_id when entering ACTIVE
        if new_state == AssistantState.ACTIVE and not self._task_id:
            self._task_id = new_task_id()
            self._started_at = now

        audit.log_state_transition(old_state.value, new_state.value)
        logger.info(
            "state_transition",
            from_state=old_state.value,
            to_state=new_state.value,
            task_id=self._task_id,
        )

    async def activate(self) -> None:
        """Full activation sequence: STANDBY → WAKE_DETECTED → ACTIVE."""
        if self._state == AssistantState.STANDBY:
            await self.transition(AssistantState.WAKE_DETECTED)
            await self.transition(AssistantState.ACTIVE)

    async def start_listening(self) -> None:
        await self.transition(AssistantState.LISTENING)

    async def think(self) -> None:
        await self.transition(AssistantState.THINKING)

    async def plan(self) -> None:
        await self.transition(AssistantState.PLANNING)

    async def execute(self) -> None:
        await self.transition(AssistantState.EXECUTING)

    async def speak(self) -> None:
        await self.transition(AssistantState.SPEAKING)

    async def complete(self) -> None:
        """Return to STANDBY after task completion."""
        if self._state in (AssistantState.SPEAKING, AssistantState.COMPLETE, AssistantState.EXECUTING):
            await self.transition(AssistantState.COMPLETE)
            await self.transition(AssistantState.STANDBY)
        self._cancelled = False
        self._task_id = ""
        self._context.clear()

    async def error(self, error_msg: str = "") -> None:
        """Transition to error state and schedule recovery."""
        if self._state != AssistantState.STANDBY:
            self._context["last_error"] = error_msg
            try:
                await self.transition(AssistantState.ERROR)
            except StateError:
                # Force to ERROR if current state doesn't support it
                self._state = AssistantState.ERROR
                self._transition_history.append((time.time(), self._state, AssistantState.ERROR))
            await self.transition(AssistantState.STANDBY)

    async def cancel(self) -> None:
        """Emergency cancel — Ctrl+Shift+P."""
        self._cancelled = True
        old = self._state
        self._state = AssistantState.STANDBY
        self._transition_history.append((time.time(), old, AssistantState.STANDBY))
        logger.warning("emergency_cancel", from_state=old.value, task_id=self._task_id)

    def is_cancelled(self) -> bool:
        return self._cancelled

    def get_transition_log(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": ts,
                "from": from_s.value,
                "to": to_s.value,
            }
            for ts, from_s, to_s in self._transition_history
        ]

    def reset(self) -> None:
        """Hard reset — use only for testing or unrecoverable errors."""
        self._state = AssistantState.STANDBY
        self._task_id = ""
        self._started_at = 0
        self._cancelled = False
        self._context.clear()
        self._transition_history.clear()
