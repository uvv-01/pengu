"""Tests for the state machine."""

import pytest

from pengu.config import AssistantState
from pengu.state import AssistantStateMachine, StateError


class TestStateMachine:
    @pytest.fixture
    def sm(self) -> AssistantStateMachine:
        return AssistantStateMachine()

    def test_initial_state(self, sm: AssistantStateMachine):
        assert sm.state == AssistantState.STANDBY
        assert sm.is_active is False

    @pytest.mark.asyncio
    async def test_valid_transition(self, sm: AssistantStateMachine):
        await sm.transition(AssistantState.WAKE_DETECTED)
        assert sm.state == AssistantState.WAKE_DETECTED

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, sm: AssistantStateMachine):
        with pytest.raises(StateError):
            await sm.transition(AssistantState.EXECUTING)

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, sm: AssistantStateMachine):
        await sm.activate()
        assert sm.state == AssistantState.ACTIVE
        assert sm.task_id.startswith("task_")

        await sm.start_listening()
        assert sm.state == AssistantState.LISTENING

        await sm.think()
        assert sm.state == AssistantState.THINKING

        await sm.plan()
        assert sm.state == AssistantState.PLANNING

        await sm.execute()
        assert sm.state == AssistantState.EXECUTING

        await sm.speak()
        assert sm.state == AssistantState.SPEAKING

        await sm.complete()
        assert sm.state == AssistantState.STANDBY
        assert sm.task_id == ""

    @pytest.mark.asyncio
    async def test_error_recovery(self, sm: AssistantStateMachine):
        await sm.activate()
        await sm.start_listening()
        await sm.error("something broke")
        assert sm.state == AssistantState.STANDBY

    @pytest.mark.asyncio
    async def test_cancel(self, sm: AssistantStateMachine):
        await sm.activate()
        await sm.cancel()
        assert sm.state == AssistantState.STANDBY
        assert sm.is_cancelled() is True

    @pytest.mark.asyncio
    async def test_transition_log(self, sm: AssistantStateMachine):
        await sm.activate()
        log = sm.get_transition_log()
        assert len(log) >= 2  # STANDBY → WAKE_DETECTED → ACTIVE
        assert log[0]["from"] == "STANDBY"
