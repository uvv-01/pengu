"""
Tests for Phase 6 (Memory Integration), Phase 7 (Scheduler), Phase 8 (Safety).
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =====================================================================
# Phase 8: Safety / Risk Classification
# =====================================================================

class TestRiskClassifier:
    def test_safe_actions(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        for action in ["system.info", "system.battery", "web_search", "chat", "list_files", "git.status"]:
            result = c.classify(action, "")
            assert result.risk_level == RiskLevel.SAFE, f"{action} should be SAFE"

    def test_low_risk_default(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("open_app", "chrome")
        assert result.risk_level == RiskLevel.LOW_RISK
        assert result.needs_confirmation is False

    def test_medium_risk_create_file(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("create", "file test.txt")
        assert result.risk_level == RiskLevel.MEDIUM_RISK
        assert result.needs_confirmation is True
        assert result.reversible is True

    def test_medium_risk_install(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("install", "some software")
        assert result.risk_level == RiskLevel.MEDIUM_RISK
        assert result.needs_confirmation is True

    def test_medium_risk_send(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("send", "email message")
        assert result.risk_level == RiskLevel.MEDIUM_RISK
        assert result.needs_confirmation is True

    def test_medium_risk_git_push(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("git", "push to origin")
        assert result.risk_level == RiskLevel.MEDIUM_RISK

    def test_high_risk_delete_files(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("delete", "important folder")
        assert result.risk_level == RiskLevel.HIGH_RISK
        assert result.needs_confirmation is True
        assert result.reversible is False

    def test_high_risk_uninstall(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("uninstall", "chrome")
        assert result.risk_level == RiskLevel.HIGH_RISK

    def test_high_risk_git_force_push(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("git", "push --force")
        assert result.risk_level == RiskLevel.HIGH_RISK

    def test_blocked_format_disk(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("format", "c:")
        assert result.risk_level == RiskLevel.BLOCKED
        assert result.needs_confirmation is False

    def test_blocked_regedit(self):
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        result = c.classify("open", "regedit")
        assert result.risk_level == RiskLevel.BLOCKED

    def test_explanation_generated(self):
        from pengu.safety import RiskClassifier
        c = RiskClassifier()
        result = c.classify("delete", "my folder")
        assert len(result.explanation) > 0
        assert "delete" in result.explanation.lower() or "permanently" in result.explanation.lower()

    def test_to_dict(self):
        from pengu.safety import RiskClassifier
        c = RiskClassifier()
        result = c.classify("open", "chrome")
        d = result.to_dict()
        assert "action" in d
        assert "risk_level" in d
        assert "needs_confirmation" in d


class TestConfirmationManager:
    def test_request_confirmation(self):
        from pengu.safety import ConfirmationManager, RiskClassifier, RiskLevel, ActionClassification
        cm = ConfirmationManager()
        cls = ActionClassification("delete", "folder", RiskLevel.HIGH_RISK, "test", needs_confirmation=True)
        msg = cm.request_confirmation(cls)
        assert "HIGH RISK" in msg

    def test_medium_risk_confirmation(self):
        from pengu.safety import ConfirmationManager, RiskLevel, ActionClassification
        cm = ConfirmationManager()
        cls = ActionClassification("install", "app", RiskLevel.MEDIUM_RISK, "test", needs_confirmation=True)
        msg = cm.request_confirmation(cls)
        assert "confirmation" in msg.lower()

    def test_session_permission(self):
        from pengu.safety import ConfirmationManager
        cm = ConfirmationManager()
        assert cm.check_session_permission("delete", "folder") is False
        cm.grant_session_permission("delete", "folder")
        assert cm.check_session_permission("delete", "folder") is True
        assert cm.check_session_permission("delete", "other") is False

    def test_revoke_permission(self):
        from pengu.safety import ConfirmationManager
        cm = ConfirmationManager()
        cm.grant_session_permission("action", "target")
        assert cm.check_session_permission("action", "target") is True
        cm.revoke_session_permission("action", "target")
        assert cm.check_session_permission("action", "target") is False

    def test_resolve(self):
        from pengu.safety import ConfirmationManager, RiskLevel, ActionClassification
        cm = ConfirmationManager()
        cls = ActionClassification("test", "", RiskLevel.MEDIUM_RISK, "test", needs_confirmation=True)
        msg = cm.request_confirmation(cls)
        conf_id = list(cm._pending.keys())[0]
        result = cm.resolve(conf_id, True)
        assert result is not None
        assert cm.resolve(conf_id, True) is None  # already resolved


class TestSafetyPolicy:
    def test_check_safe(self):
        from pengu.safety import SafetyPolicy, RiskLevel
        sp = SafetyPolicy()
        result = sp.check("system.battery")
        assert result.risk_level == RiskLevel.SAFE
        assert result.needs_confirmation is False

    def test_check_blocked(self):
        from pengu.safety import SafetyPolicy, RiskLevel
        sp = SafetyPolicy()
        result = sp.check("format", "c:")
        assert result.risk_level == RiskLevel.BLOCKED

    def test_check_medium_risk(self):
        from pengu.safety import SafetyPolicy, RiskLevel
        sp = SafetyPolicy()
        result = sp.check("create", "file test.py")
        assert result.risk_level == RiskLevel.MEDIUM_RISK
        assert result.needs_confirmation is True

    def test_permission_bypasses_confirmation(self):
        from pengu.safety import SafetyPolicy, RiskLevel
        sp = SafetyPolicy()
        sp.grant_permission("create", "file test.py")
        result = sp.check("create", "file test.py")
        assert result.needs_confirmation is False

    def test_confirm_action(self):
        from pengu.safety import SafetyPolicy, RiskLevel
        sp = SafetyPolicy()
        result = sp.check("delete", "folder")
        msg = sp.confirm_action(result)
        assert len(msg) > 0
        assert "HIGH RISK" in msg

    def test_singleton(self):
        from pengu.safety import get_safety_policy, reset_safety_policy
        reset_safety_policy()
        sp1 = get_safety_policy()
        sp2 = get_safety_policy()
        assert sp1 is sp2


# =====================================================================
# Phase 7: Scheduler
# =====================================================================

class TestSchedule:
    def test_delayed_schedule(self):
        from pengu.scheduler import Schedule, ScheduleType
        s = Schedule(ScheduleType.DELAYED, delay_seconds=1800)
        next_run = s.get_next_run(time.time())
        assert next_run > time.time()
        assert next_run <= time.time() + 1801

    def test_once_schedule(self):
        from pengu.scheduler import Schedule, ScheduleType
        future = time.time() + 3600
        s = Schedule(ScheduleType.ONCE, run_at=future)
        assert s.get_next_run() == future

    def test_recurring_schedule(self):
        from pengu.scheduler import Schedule, ScheduleType
        s = Schedule(ScheduleType.RECURRING, interval_seconds=3600)
        next_run = s.get_next_run(time.time())
        assert next_run > time.time()

    def test_schedule_to_dict_from_dict(self):
        from pengu.scheduler import Schedule, ScheduleType
        s = Schedule(ScheduleType.DELAYED, delay_seconds=60, description="in 1 minute")
        d = s.to_dict()
        s2 = Schedule.from_dict(d)
        assert s2.schedule_type == ScheduleType.DELAYED
        assert s2.delay_seconds == 60
        assert s2.description == "in 1 minute"


class TestScheduledMission:
    def test_mission_to_dict_from_dict(self):
        from pengu.scheduler import ScheduledMission, Schedule, ScheduleType, MissionState
        s = Schedule(ScheduleType.ONCE, run_at=time.time() + 3600)
        m = ScheduledMission(id="test1", name="test", description="desc", task="task", schedule=s, state=MissionState.QUEUED)
        d = m.to_dict()
        m2 = ScheduledMission.from_dict(d)
        assert m2.id == "test1"
        assert m2.name == "test"
        assert m2.state == MissionState.QUEUED


class TestParseSchedule:
    def test_parse_delayed_minutes(self):
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("in 30 minutes")
        assert s is not None
        assert s.schedule_type == ScheduleType.DELAYED
        assert s.delay_seconds == 1800

    def test_parse_delayed_hours(self):
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("in 2 hours")
        assert s is not None
        assert s.schedule_type == ScheduleType.DELAYED
        assert s.delay_seconds == 7200

    def test_parse_recurring(self):
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("every day")
        assert s is not None
        assert s.schedule_type == ScheduleType.RECURRING
        assert s.interval_seconds == 86400

    def test_parse_recurring_weekly(self):
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("every 2 weeks")
        assert s is not None
        assert s.interval_seconds == 2 * 604800

    def test_parse_at_time(self):
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("at 9am")
        assert s is not None
        assert s.schedule_type == ScheduleType.ONCE
        assert "9:00" in s.description

    def test_parse_no_schedule(self):
        from pengu.scheduler import parse_schedule
        s = parse_schedule("open chrome")
        assert s is None


class TestScheduler:
    @pytest.mark.asyncio
    async def test_create_mission(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        s = Scheduler(":memory:")
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() + 3600)
        m = await s.create_mission("test", "do something", sched)
        assert m.state == MissionState.QUEUED
        assert m.id in s._missions
        await s.close()

    @pytest.mark.asyncio
    async def test_cancel_mission(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        s = Scheduler(":memory:")
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() + 3600)
        m = await s.create_mission("test", "task", sched)
        result = await s.cancel_mission(m.id)
        assert "cancelled" in result.lower()
        assert m.state == MissionState.CANCELLED
        await s.close()

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        s = Scheduler(":memory:")
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() + 3600)
        m = await s.create_mission("test", "task", sched)
        await s.pause_mission(m.id)
        assert m.state == MissionState.PAUSED
        await s.resume_mission(m.id)
        assert m.state == MissionState.QUEUED
        await s.close()

    @pytest.mark.asyncio
    async def test_list_missions(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        s = Scheduler(":memory:")
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() + 3600)
        await s.create_mission("m1", "task1", sched)
        await s.create_mission("m2", "task2", sched)
        all_missions = s.list_missions()
        assert len(all_missions) == 2
        queued = s.list_missions(state=MissionState.QUEUED)
        assert len(queued) == 2
        await s.close()

    @pytest.mark.asyncio
    async def test_execute_mission(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        executor_called = []
        async def executor(task):
            executor_called.append(task)
            return "done"
        s = Scheduler(":memory:")
        s.set_task_executor(executor)
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() - 1)  # already due
        m = await s.create_mission("test", "my task", sched)
        await s._execute_mission(m)
        assert len(executor_called) == 1
        assert executor_called[0] == "my task"
        assert m.state == MissionState.COMPLETED
        assert m.last_result == "done"
        await s.close()

    @pytest.mark.asyncio
    async def test_mission_failure_retry(self):
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        call_count = [0]
        async def failing_executor(task):
            call_count[0] += 1
            raise RuntimeError("test error")
        s = Scheduler(":memory:")
        s.set_task_executor(failing_executor)
        await s.initialize()
        sched = Schedule(ScheduleType.ONCE, run_at=time.time() - 1)
        m = await s.create_mission("test", "task", sched)
        m.max_retries = 2
        await s._execute_mission(m)
        assert call_count[0] == 1
        assert m.failure_count == 1
        assert m.state == MissionState.QUEUED  # will retry
        await s.close()

    @pytest.mark.asyncio
    async def test_singleton(self):
        from pengu.scheduler import get_scheduler, reset_scheduler
        reset_scheduler()
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2


# =====================================================================
# Phase 6: Context Enhancement
# =====================================================================

class TestContextEnhancement:
    def test_context_summary(self):
        from pengu.context import get_context, reset_context
        ctx = reset_context()
        ctx.update_app("chrome")
        ctx.update_url("https://example.com", "Example")
        summary = ctx.get_summary()
        assert summary["current_app"] == "chrome"
        assert summary["current_url"] == "https://example.com"

    def test_context_followup_browser(self):
        from pengu.context import get_context, reset_context
        ctx = reset_context()
        ctx.update_app("chrome")
        ctx.update_url("https://github.com", "GitHub")
        result = ctx.resolve_followup("search for python")
        assert "python" in result

    def test_context_followup_explorer(self):
        from pengu.context import get_context, reset_context
        ctx = reset_context()
        ctx.update_app("explorer")
        result = ctx.resolve_followup("downloads")
        assert "downloads" in result.lower()

    def test_context_clear(self):
        from pengu.context import get_context, reset_context
        ctx = reset_context()
        ctx.update_app("chrome")
        ctx.update_url("https://example.com")
        ctx.clear()
        assert ctx.current_app == ""
        assert ctx.current_url == ""

    def test_conversation_history(self):
        from pengu.context import get_context, reset_context
        ctx = reset_context()
        ctx.add_turn("hello", "Hi there!")
        ctx.add_turn("open chrome", "Chrome is open.")
        assert len(ctx.history) == 2
        assert ctx.history[0].user_text == "hello"
        assert ctx.history[1].response == "Chrome is open."


# =====================================================================
# Integration: Router recognizes MISSIONS category
# =====================================================================

class TestRouterIntegration:
    def test_reminder_classified_as_missions(self):
        from pengu.router import IntentRouter
        router = IntentRouter()
        intent = router.classify("remind me in 30 minutes")
        from pengu.config import TaskCategory
        assert intent.category == TaskCategory.MISSIONS

    def test_schedule_classified_as_missions(self):
        from pengu.router import IntentRouter
        router = IntentRouter()
        intent = router.classify("what tasks are scheduled")
        from pengu.config import TaskCategory
        assert intent.category == TaskCategory.MISSIONS

    def test_memory_still_classified(self):
        from pengu.router import IntentRouter
        router = IntentRouter()
        intent = router.classify("remember that I use Chrome")
        from pengu.config import TaskCategory
        assert intent.category == TaskCategory.MEMORY
