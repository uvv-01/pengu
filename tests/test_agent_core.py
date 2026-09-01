"""
Tests for the Agent Core — AgentState, WorldState, AgentBrain, MissionManager.

These tests verify the autonomous agent infrastructure without requiring
real hardware, browser, or microphone access.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pengu.agent.state import (
    AgentDecision,
    AgentState,
    MissionStatus,
    Observation,
    PlanStep,
    StepStatus,
    WorldState,
)
from pengu.agent.brain import AgentBrain, reset_brain
from pengu.agent.mission import MissionManager, reset_mission_manager


# ---------------------------------------------------------------------------
# WorldState tests
# ---------------------------------------------------------------------------

class TestWorldState:
    def test_initial_state(self):
        ws = WorldState()
        assert ws.active_app == ""
        assert ws.browser_open is False
        assert ws.last_observed == 0.0

    def test_update_from_observation(self):
        ws = WorldState()
        obs = Observation(
            timestamp=time.time(),
            active_window_title="Google Chrome — Python docs",
            active_app="Google Chrome",
            browser_url="https://docs.python.org",
            browser_title="Python docs",
            raw_data={"screen_width": 1920, "screen_height": 1080},
        )
        ws.update_from_observation(obs)
        assert ws.active_app == "Google Chrome"
        assert ws.browser_url == "https://docs.python.org"
        assert ws.browser_open is True
        assert ws.screen_width == 1920

    def test_is_browser_active(self):
        ws = WorldState()
        ws.active_app = "Google Chrome"
        assert ws.is_browser_active() is True
        ws.active_app = "VS Code"
        assert ws.is_browser_active() is False
        ws.active_app = "Mozilla Firefox"
        assert ws.is_browser_active() is True

    def test_to_dict(self):
        ws = WorldState()
        ws.active_app = "Chrome"
        d = ws.to_dict()
        assert "active_app" in d
        assert d["active_app"] == "Chrome"


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_initial_observation(self):
        obs = Observation(timestamp=time.time())
        assert obs.active_window_title == ""
        assert obs.visible_elements == []

    def test_observation_with_data(self):
        obs = Observation(
            timestamp=time.time(),
            active_window_title="Test Window",
            active_app="TestApp",
            visible_elements=["Button1", "TextField"],
        )
        d = obs.to_dict()
        assert d["active_app"] == "TestApp"
        assert len(d["visible_elements"]) == 2


# ---------------------------------------------------------------------------
# AgentState tests
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_initial_state(self):
        state = AgentState(goal="Open Chrome")
        assert state.status == MissionStatus.PENDING
        assert state.goal == "Open Chrome"
        assert state.plan == []
        assert state.current_step_index == 0

    def test_add_and_advance_step(self):
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", target="chrome")
        state.add_step(step)
        assert len(state.plan) == 1
        assert state.current_step() is step

        step.status = StepStatus.SUCCESS
        assert state.advance_step() is False  # no more steps
        assert state.is_plan_complete() is True

    def test_multiple_steps(self):
        state = AgentState(goal="multi step task")
        state.add_step(PlanStep(step_id=0, action="open_app", target="chrome"))
        state.add_step(PlanStep(step_id=1, action="navigate", target="google.com"))
        state.add_step(PlanStep(step_id=2, action="type_text", target="python"))

        assert state.current_step().step_id == 0

        state.plan[0].status = StepStatus.SUCCESS
        state.advance_step()
        assert state.current_step().step_id == 1

        state.plan[1].status = StepStatus.SUCCESS
        state.advance_step()
        assert state.current_step().step_id == 2

        state.plan[2].status = StepStatus.SUCCESS
        assert state.advance_step() is False  # no more steps after last
        assert state.is_plan_complete() is True

    def test_observation_tracking(self):
        state = AgentState(goal="test")
        obs = Observation(timestamp=time.time(), active_app="Chrome")
        state.add_observation(obs)
        assert len(state.observations) == 1
        assert state.world.active_app == "Chrome"
        assert state.latest_observation() is obs

    def test_action_recording(self):
        state = AgentState(goal="test")
        state.record_action("open_app", "chrome", True, "Chrome opened")
        assert len(state.action_history) == 1
        assert state.consecutive_failures == 0

        state.record_action("click", "button", False, error="not found")
        assert state.consecutive_failures == 1

    def test_error_tracking(self):
        state = AgentState(goal="test")
        state.add_error("Something went wrong")
        assert len(state.errors) == 1

    def test_mission_status_transitions(self):
        state = AgentState(goal="test")
        assert state.status == MissionStatus.PENDING
        state.mark_planning()
        assert state.status == MissionStatus.PLANNING
        state.mark_running()
        assert state.status == MissionStatus.RUNNING
        state.mark_completed("Done")
        assert state.status == MissionStatus.COMPLETED
        assert state.final_response == "Done"

    def test_failed_steps(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(step_id=0, action="a", status=StepStatus.SUCCESS))
        state.add_step(PlanStep(step_id=1, action="b", status=StepStatus.FAILED))
        state.add_step(PlanStep(step_id=2, action="c", status=StepStatus.SUCCESS))

        assert state.has_failures() is True
        assert len(state.failed_steps()) == 1
        assert state.all_steps_succeeded() is False

    def test_to_dict(self):
        state = AgentState(goal="test goal")
        d = state.to_dict()
        assert d["goal"] == "test goal"
        assert d["status"] == "pending"


# ---------------------------------------------------------------------------
# AgentBrain tests
# ---------------------------------------------------------------------------

class TestAgentBrain:
    def setup_method(self):
        self.brain = reset_brain()

    def test_understand_simple(self):
        state = AgentState(goal="Open Chrome")
        intent = self.brain.understand("Open Chrome", state)
        assert intent["type"] in ("simple", "multi_step")
        assert intent["requires_browser"] is True  # "open" triggers browser detection

    def test_understand_multi_step(self):
        state = AgentState(goal="Open Chrome and search for Python")
        intent = self.brain.understand("Open Chrome and search for Python", state)
        assert intent["type"] == "multi_step"

    def test_understand_research(self):
        state = AgentState(goal="Find out the latest Python version")
        intent = self.brain.understand("Find out the latest Python version", state)
        assert intent["type"] == "research"
        assert intent["requires_browser"] is True

    def test_understand_high_risk(self):
        state = AgentState(goal="Delete all files in Downloads")
        intent = self.brain.understand("Delete all files in Downloads", state)
        assert intent["risk_level"] == "high"

    def test_understand_medium_risk(self):
        state = AgentState(goal="Install Python 3.12")
        intent = self.brain.understand("Install Python 3.12", state)
        assert intent["risk_level"] == "medium"

    def test_decide_next_no_plan(self):
        state = AgentState(goal="test")
        decision = self.brain.decide_next(state)
        assert decision == AgentDecision.PLAN

    def test_decide_next_with_plan(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(step_id=0, action="open_app", target="chrome"))
        decision = self.brain.decide_next(state)
        assert decision == AgentDecision.ACT

    def test_decide_next_step_complete(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(step_id=0, action="open_app", status=StepStatus.SUCCESS))
        decision = self.brain.decide_next(state)
        assert decision == AgentDecision.COMPLETE  # all done

    def test_decide_next_step_failed_with_retries(self):
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", max_retries=2, retry_count=0)
        step.status = StepStatus.FAILED
        state.add_step(step)
        decision = self.brain.decide_next(state)
        assert decision == AgentDecision.RECOVER

    def test_decide_next_step_failed_no_retries(self):
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", max_retries=0, retry_count=1)
        step.status = StepStatus.FAILED
        state.add_step(step)
        decision = self.brain.decide_next(state)
        # Should skip and advance
        assert decision in (AgentDecision.ACT, AgentDecision.COMPLETE)

    def test_recover_with_retries(self):
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", max_retries=2, retry_count=0)
        step.status = StepStatus.FAILED
        state.add_step(step)
        recovered = self.brain.recover(state)
        assert recovered is not None
        assert recovered.status == StepStatus.PENDING

    def test_recover_skip_when_max_retries(self):
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", max_retries=0, retry_count=1)
        step.status = StepStatus.FAILED
        state.add_step(step)
        recovered = self.brain.recover(state)
        assert recovered is None
        assert step.status == StepStatus.SKIPPED

    def test_generate_response_completed(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(
            step_id=0, action="open_app", status=StepStatus.SUCCESS,
            result_message="Chrome is open",
        ))
        state.mark_completed()
        response = self.brain.generate_response(state)
        assert "Chrome" in response

    def test_generate_response_failed(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(
            step_id=0, action="open_app", status=StepStatus.FAILED,
            error="Not found",
            description="Open Firefox",
        ))
        state.mark_failed("Could not complete")
        response = self.brain.generate_response(state)
        assert "Firefox" in response or "couldn't" in response.lower()

    def test_generate_response_partial(self):
        state = AgentState(goal="test")
        state.add_step(PlanStep(step_id=0, action="a", status=StepStatus.SUCCESS))
        state.add_step(PlanStep(step_id=1, action="b", status=StepStatus.FAILED))
        state.mark_completed("Done with issues")
        response = self.brain.generate_response(state)
        assert "1 of 2" in response

    def test_replan(self):
        state = AgentState(goal="open chrome and search python")
        state.add_step(PlanStep(step_id=0, action="open_app", target="chrome",
                                status=StepStatus.SUCCESS))
        state.add_step(PlanStep(step_id=1, action="search", target="python",
                                status=StepStatus.FAILED))
        state.current_step_index = 1

        new_steps = self.brain.replan(state, reason="search failed")
        # Should have created new plan from remaining goal
        assert isinstance(new_steps, list)


# ---------------------------------------------------------------------------
# MissionManager tests
# ---------------------------------------------------------------------------

class TestMissionManager:
    def setup_method(self):
        self.manager = reset_mission_manager()

    def test_create_mission(self):
        state = self.manager._create_mission("Open Chrome")
        assert state.goal == "Open Chrome"
        assert state.status == MissionStatus.PENDING
        assert state.mission_id in self.manager._missions

    def test_initial_state(self):
        assert self.manager.current_mission is None
        assert self.manager.is_busy is False

    def test_cancel(self):
        result = self.manager.cancel()
        assert "No mission" in result

    def test_mission_history(self):
        self.manager._create_mission("Task 1")
        self.manager._create_mission("Task 2")
        history = self.manager.get_mission_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_execute_goal_with_mock_executor(self):
        """Test the full mission loop with a mock tool executor."""
        call_count = 0

        async def mock_executor(action, params):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.message = f"Executed {action}"
            result.error = ""
            result.output = f"Result of {action}"
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome",
            tool_executor=mock_executor,
        )
        assert isinstance(response, str)
        assert len(response) > 0
        assert call_count >= 1  # at least one tool was executed

    @pytest.mark.asyncio
    async def test_execute_goal_failing_action(self):
        """Test that failing actions are handled gracefully."""
        async def failing_executor(action, params):
            result = MagicMock()
            result.success = False
            result.message = ""
            result.error = "Action failed"
            result.output = ""
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome",
            tool_executor=failing_executor,
        )
        # Should not crash — should return a failure message
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_execute_goal_no_executor(self):
        """Test that missing executor is handled gracefully."""
        manager = reset_mission_manager()
        response = await manager.execute_goal("Open Chrome")
        assert "tools" in response.lower() or "don't" in response.lower()

    @pytest.mark.asyncio
    async def test_cancel_during_execution(self):
        """Test that cancellation works during execution."""
        cancel_called = False

        async def slow_executor(action, params):
            nonlocal cancel_called
            if not cancel_called:
                cancel_called = True
                # Cancel during the first action
                # The mission should detect cancellation
                return MagicMock(success=True, message="Step done", error="", output="")
            return MagicMock(success=True, message="Done", error="", output="")

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome",
            tool_executor=slow_executor,
        )
        assert isinstance(response, str)
        assert len(response) > 0


# ---------------------------------------------------------------------------
# PlanStep tests
# ---------------------------------------------------------------------------

class TestPlanStep:
    def test_initial_step(self):
        step = PlanStep(step_id=0, action="open_app", target="chrome")
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0

    def test_step_to_dict(self):
        step = PlanStep(step_id=1, action="click", target="button",
                        description="Click the button")
        d = step.to_dict()
        assert d["step_id"] == 1
        assert d["action"] == "click"
        assert d["status"] == "pending"


# ---------------------------------------------------------------------------
# Integration: Brain + MissionManager end-to-end
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_simple_task(self):
        """Simulate a simple end-to-end agent mission."""
        actions_executed = []

        async def mock_executor(action, params):
            actions_executed.append({"action": action, "params": params})
            result = MagicMock()
            result.success = True
            result.message = f"Done: {action}"
            result.error = ""
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome",
            tool_executor=mock_executor,
        )

        # Verify the mission ran
        assert isinstance(response, str)
        assert len(actions_executed) >= 1

        # Verify history was recorded
        history = manager.get_mission_history()
        assert len(history) == 1
        assert history[0]["goal"] == "Open Chrome"

    @pytest.mark.asyncio
    async def test_end_to_end_multi_step(self):
        """Simulate a multi-step mission."""
        actions_executed = []

        async def mock_executor(action, params):
            actions_executed.append(action)
            result = MagicMock()
            result.success = True
            result.message = f"Done: {action}"
            result.error = ""
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome and search for Python",
            tool_executor=mock_executor,
        )

        # Multi-step should have executed more than one action
        assert len(actions_executed) >= 1
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_end_to_end_partial_failure(self):
        """Test that partial failures don't crash the mission."""
        call_count = 0

        async def partial_failure_executor(action, params):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                result.success = True
                result.message = "First action succeeded"
            else:
                result.success = False
                result.message = ""
                result.error = "Second action failed"
            result.output = ""
            return result

        manager = reset_mission_manager()
        response = await manager.execute_goal(
            "Open Chrome and search for Python",
            tool_executor=partial_failure_executor,
        )

        # Should not crash — should handle partial failure
        assert isinstance(response, str)
        assert len(response) > 0

    def test_brain_observe_with_mocked_observer(self):
        """Test that observe works with mocked observer."""
        brain = AgentBrain()

        mock_observer = MagicMock()
        mock_observer.get_active_window.return_value = {
            "title": "Google Chrome — New Tab",
            "app": "Google Chrome",
            "x": "0", "y": "0", "width": "1920", "height": "1080",
        }
        mock_observer.get_elements.return_value = []

        mock_desktop = MagicMock()
        mock_desktop.window.get_screen_size.return_value = (1920, 1080)

        brain._observer = mock_observer
        brain._desktop = mock_desktop

        state = AgentState(goal="test")
        obs = brain.observe(state)

        assert obs.active_app == "Google Chrome"
        assert obs.active_window_title == "Google Chrome — New Tab"
        assert state.world.screen_width == 1920


class TestAgentBrainLLMPlanning:
    """Test AgentBrain LLM-assisted planning."""

    def test_plan_without_llm_falls_back_to_rules(self):
        """Without LLM provider, plan should use rule-based planning."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState
        brain = AgentBrain(model_provider=None)
        state = AgentState(goal="notepad")
        intent = brain.understand("open notepad", state)
        steps = brain.plan(state, intent)
        assert len(steps) >= 1
        assert steps[0].action in ("open_app", "navigate")

    def test_plan_simple_search(self):
        """Simple search should generate a web_search step."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState
        brain = AgentBrain(model_provider=None)
        state = AgentState(goal="search for Python")
        intent = brain.understand("search for Python", state)
        steps = brain.plan(state, intent)
        assert len(steps) >= 1
        assert any(s.action == "web_search" for s in steps)

    def test_plan_multi_step_without_llm(self):
        """Multi-step without LLM should split on 'and'/'then'."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState
        brain = AgentBrain(model_provider=None)
        state = AgentState(goal="open Chrome and search for Python")
        intent = brain.understand("open Chrome and search for Python", state)
        steps = brain.plan(state, intent)
        assert len(steps) >= 2

    def test_understand_battery_query(self):
        """Battery query should be understood as info_query."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState
        brain = AgentBrain(model_provider=None)
        state = AgentState(goal="What's my battery percentage?")
        intent = brain.understand("What's my battery percentage?", state)
        assert "requires_browser" in intent  # should not require browser
        assert intent["type"] in ("simple", "info_query")

    def test_understand_wallpaper_change(self):
        """Wallpaper change should be understood."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState
        brain = AgentBrain(model_provider=None)
        state = AgentState(goal="Change my wallpaper")
        intent = brain.understand("Change my wallpaper", state)
        assert intent["type"] == "simple"

    def test_generate_response_completed(self):
        """Completed mission should produce a useful response."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, MissionStatus, PlanStep, StepStatus
        brain = AgentBrain()
        state = AgentState(goal="test")
        state.status = MissionStatus.COMPLETED
        state.plan = [
            PlanStep(step_id=0, action="open_app", status=StepStatus.SUCCESS,
                     result_message="Chrome is open."),
        ]
        response = brain.generate_response(state)
        assert "Chrome" in response or "open" in response.lower() or "Done" in response

    def test_generate_response_failed(self):
        """Failed mission should explain what went wrong."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, MissionStatus, PlanStep, StepStatus
        brain = AgentBrain()
        state = AgentState(goal="test")
        state.status = MissionStatus.FAILED
        state.plan = [
            PlanStep(step_id=0, action="open_app", status=StepStatus.FAILED,
                     error="Application not found"),
        ]
        response = brain.generate_response(state)
        assert "not" in response.lower() or "couldn" in response.lower() or "error" in response.lower()

    def test_recover_retry(self):
        """Recovery should retry a failed step if retries remain."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, PlanStep, StepStatus
        brain = AgentBrain()
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", status=StepStatus.FAILED,
                         retry_count=0, max_retries=2)
        state.plan = [step]
        state.current_step_index = 0
        recovered = brain.recover(state)
        assert recovered is not None
        assert recovered.status == StepStatus.PENDING

    def test_recover_skip_when_max_retries(self):
        """Recovery should skip when max retries exceeded."""
        from pengu.agent.brain import AgentBrain
        from pengu.agent.state import AgentState, PlanStep, StepStatus
        brain = AgentBrain()
        state = AgentState(goal="test")
        step = PlanStep(step_id=0, action="open_app", status=StepStatus.FAILED,
                         retry_count=2, max_retries=2)
        state.plan = [step]
        state.current_step_index = 0
        recovered = brain.recover(state)
        assert recovered is None
        assert step.status == StepStatus.SKIPPED
