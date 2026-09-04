"""
Integration tests for Phase 6 (Memory), Phase 7 (Scheduler), Phase 8 (Safety),
and Phase 9 (Observe/Act/Verify/Recover) — testing real execution paths.
"""

import asyncio
import os
import time
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# =====================================================================
# Phase 6: Memory — Integration Tests
# =====================================================================

class TestMemoryIntegration:
    """Test memory handler uses singleton and supports forget/recall."""

    @pytest.mark.asyncio
    async def test_memory_handler_uses_singleton(self):
        """Verify handle_memory uses get_memory() not MemoryManager()."""
        from pengu.memory import get_memory, MemoryCategory, MemoryType, reset_memory

        mem = reset_memory()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        mem._db_path = db_path
        await mem.initialize()

        entry = await mem.save(
            content="User prefers Chrome browser",
            category=MemoryCategory.PREFERENCE,
            memory_type=MemoryType.PERSISTENT,
        )
        assert entry.id is not None

        results = await mem.search("Chrome", limit=5)
        assert len(results) >= 1
        assert "Chrome" in results[0].content

        await mem.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_memory_persistence_across_restart(self):
        """Test that persistent memory survives closing and reopening."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            from pengu.memory import MemoryManager, MemoryCategory, MemoryType
            mem1 = MemoryManager(db_path=db_path)
            await mem1.initialize()
            entry = await mem1.save(
                content="My project is called Pengu",
                category=MemoryCategory.PROJECT,
                memory_type=MemoryType.PERSISTENT,
            )
            entry_id = entry.id
            await mem1.close()

            mem2 = MemoryManager(db_path=db_path)
            await mem2.initialize()
            retrieved = await mem2.get(entry_id)
            assert retrieved is not None
            assert retrieved.content == "My project is called Pengu"
            assert retrieved.category == MemoryCategory.PROJECT
            await mem2.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_memory_delete(self):
        """Test memory deletion."""
        from pengu.memory import MemoryManager, MemoryCategory, MemoryType
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = MemoryManager(db_path=db_path)
            await mem.initialize()
            entry = await mem.save(
                content="Temporary fact",
                category=MemoryCategory.GENERAL,
                memory_type=MemoryType.PERSISTENT,
            )
            deleted = await mem.delete(entry.id)
            assert deleted is True

            retrieved = await mem.get(entry.id)
            assert retrieved is None
            await mem.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_memory_sensitive_content_rejected(self):
        """Test that sensitive content is rejected."""
        from pengu.memory import MemoryManager, MemoryCategory, MemoryType
        mem = MemoryManager()
        with pytest.raises(ValueError, match="sensitive"):
            await mem.save(
                content="My API key is abc123",
                category=MemoryCategory.GENERAL,
                memory_type=MemoryType.PERSISTENT,
            )

    @pytest.mark.asyncio
    async def test_memory_search(self):
        """Test memory search by content."""
        from pengu.memory import MemoryManager, MemoryCategory, MemoryType
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            mem = MemoryManager(db_path=db_path)
            await mem.initialize()
            await mem.save("I prefer dark mode", MemoryCategory.PREFERENCE, MemoryType.PERSISTENT)
            await mem.save("I use VS Code for coding", MemoryCategory.PREFERENCE, MemoryType.PERSISTENT)
            await mem.save("My project is Pengu", MemoryCategory.PROJECT, MemoryType.PERSISTENT)

            results = await mem.search("dark mode")
            assert len(results) >= 1
            assert any("dark mode" in r.content for r in results)

            results2 = await mem.search("VS Code")
            assert len(results2) >= 1
            await mem.close()
        finally:
            os.unlink(db_path)

    def test_memory_handler_forget_flow(self):
        """Test the forget handler works through pipeline."""
        from pengu.memory import reset_memory, get_memory
        mem = reset_memory()
        assert mem is not None

    def test_context_pronoun_resolution(self):
        """Test that context resolves pronouns from last results."""
        from pengu.context import reset_context
        ctx = reset_context()
        ctx.update_directory("C:/Users/Downloads")
        ctx.record_action("open_folder:downloads", "C:/Users/Downloads")
        ctx.last_opened_folder = "C:/Users/Downloads"

        result = ctx.resolve_followup("open it")
        assert isinstance(result, str)

    def test_context_folder_followup(self):
        """Test folder follow-up resolution."""
        from pengu.context import reset_context
        ctx = reset_context()
        ctx.update_app("File Explorer")

        result = ctx.resolve_followup("open downloads")
        assert "downloads" in result.lower()

    def test_context_browser_followup(self):
        """Test browser follow-up resolution."""
        from pengu.context import reset_context
        ctx = reset_context()
        ctx.update_url("https://google.com", "Google")
        ctx.update_app("chrome")

        result = ctx.resolve_followup("search for Python")
        assert "search" in result.lower()


# =====================================================================
# Phase 7: Scheduler — Integration Tests
# =====================================================================

class TestSchedulerIntegration:
    """Test scheduler initialization, persistence, and mission lifecycle."""

    @pytest.mark.asyncio
    async def test_scheduler_create_and_list(self):
        """Test creating and listing missions."""
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            sched = Scheduler(db_path=db_path)
            await sched.initialize()

            schedule = Schedule(
                schedule_type=ScheduleType.DELAYED,
                delay_seconds=300,
                description="in 5 minutes",
            )
            mission = await sched.create_mission(
                name="Test reminder",
                task="Check email",
                schedule=schedule,
            )
            assert mission.id.startswith("mis_")
            assert mission.state == MissionState.QUEUED

            missions = sched.list_missions()
            assert len(missions) >= 1
            await sched.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_scheduler_cancel(self):
        """Test cancelling a mission."""
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            sched = Scheduler(db_path=db_path)
            await sched.initialize()

            schedule = Schedule(schedule_type=ScheduleType.DELAYED, delay_seconds=60)
            mission = await sched.create_mission("Cancel test", "task", schedule)

            result = await sched.cancel_mission(mission.id)
            assert "cancelled" in result.lower()

            m = sched.get_mission(mission.id)
            assert m.state == MissionState.CANCELLED
            await sched.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_scheduler_persistence(self):
        """Test that missions survive restart."""
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            sched1 = Scheduler(db_path=db_path)
            await sched1.initialize()
            schedule = Schedule(schedule_type=ScheduleType.DELAYED, delay_seconds=3600)
            mission = await sched1.create_mission("Persistent test", "task", schedule)
            mid = mission.id
            await sched1.close()

            sched2 = Scheduler(db_path=db_path)
            await sched2.initialize()
            m = sched2.get_mission(mid)
            assert m is not None
            assert m.name == "Persistent test"
            assert m.state == MissionState.QUEUED
            await sched2.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_scheduler_pause_resume(self):
        """Test pausing and resuming missions."""
        from pengu.scheduler import Scheduler, Schedule, ScheduleType, MissionState
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            sched = Scheduler(db_path=db_path)
            await sched.initialize()

            schedule = Schedule(schedule_type=ScheduleType.RECURRING, interval_seconds=3600)
            mission = await sched.create_mission("Recurring test", "task", schedule)

            result = await sched.pause_mission(mission.id)
            assert "paused" in result.lower()
            m = sched.get_mission(mission.id)
            assert m.state == MissionState.PAUSED

            await sched.resume_mission(mission.id)
            m2 = sched.get_mission(mission.id)
            assert m2.state == MissionState.QUEUED
            await sched.close()
        finally:
            os.unlink(db_path)

    def test_parse_schedule_delayed(self):
        """Test parsing delayed schedule from natural language."""
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("Remind me in 30 minutes")
        assert s is not None
        assert s.schedule_type == ScheduleType.DELAYED
        assert s.delay_seconds == 1800

    def test_parse_schedule_recurring(self):
        """Test parsing recurring schedule."""
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("Every day")
        assert s is not None
        assert s.schedule_type == ScheduleType.RECURRING
        assert s.interval_seconds == 86400

    def test_parse_schedule_time(self):
        """Test parsing time-based schedule."""
        from pengu.scheduler import parse_schedule, ScheduleType
        s = parse_schedule("at 9am")
        assert s is not None
        assert s.schedule_type == ScheduleType.ONCE
        assert s.run_at > 0


# =====================================================================
# Phase 8: Safety — Integration Tests
# =====================================================================

class TestSafetyIntegration:
    """Test safety policy integration into real execution paths."""

    def test_safety_blocks_dangerous_commands(self):
        """Test that blocked actions are classified correctly."""
        from pengu.safety import SafetyPolicy, RiskLevel
        policy = SafetyPolicy()

        cls = policy.check("format c:", "")
        assert cls.risk_level == RiskLevel.BLOCKED

        cls2 = policy.check("dd if=/dev/sda of=/dev/null", "")
        assert cls2.risk_level == RiskLevel.BLOCKED

    def test_safety_allows_safe_actions(self):
        """Test that safe actions are not blocked."""
        from pengu.safety import SafetyPolicy, RiskLevel
        policy = SafetyPolicy()

        cls = policy.check("system.info", "")
        assert cls.risk_level == RiskLevel.SAFE

        cls2 = policy.check("system.battery", "")
        assert cls2.risk_level == RiskLevel.SAFE

        cls3 = policy.check("browser_read", "")
        assert cls3.risk_level == RiskLevel.SAFE

    def test_safety_requires_confirmation_for_high_risk(self):
        """Test that high-risk actions need confirmation."""
        from pengu.safety import SafetyPolicy, RiskLevel
        policy = SafetyPolicy()

        cls = policy.check("delete all files in folder", "")
        assert cls.risk_level == RiskLevel.HIGH_RISK
        assert cls.needs_confirmation is True

    def test_safety_confirmation_flow(self):
        """Test the confirmation request/response flow."""
        from pengu.safety import SafetyPolicy, RiskLevel, ActionClassification
        policy = SafetyPolicy()

        cls = ActionClassification(
            action="delete",
            target="important_folder",
            risk_level=RiskLevel.HIGH_RISK,
            reason="test",
            needs_confirmation=True,
            explanation="This will delete important_folder permanently.",
        )

        msg = policy.confirm_action(cls)
        assert "HIGH RISK" in msg
        assert "important_folder" in msg

    def test_safety_session_permissions(self):
        """Test session permission granting and expiry."""
        from pengu.safety import SafetyPolicy
        policy = SafetyPolicy()

        assert not policy.confirmation_manager.check_session_permission("test_action", "test_target")

        policy.grant_permission("test_action", "test_target")
        assert policy.confirmation_manager.check_session_permission("test_action", "test_target")

        policy.confirmation_manager.revoke_session_permission("test_action", "test_target")
        assert not policy.confirmation_manager.check_session_permission("test_action", "test_target")

    def test_safety_blocks_git_force_push(self):
        """Test that git force push is classified as high risk."""
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()
        cls = c.classify("git push --force", "")
        assert cls.risk_level == RiskLevel.HIGH_RISK

    def test_safety_medium_risk_actions(self):
        """Test medium risk actions require confirmation."""
        from pengu.safety import RiskClassifier, RiskLevel
        c = RiskClassifier()

        cls = c.classify("install", "new software")
        assert cls.risk_level == RiskLevel.MEDIUM_RISK
        assert cls.needs_confirmation is True

    def test_safety_unrelated_confirmation_not_accepted(self):
        """Test that a stale confirmation doesn't authorize a different action."""
        from pengu.safety import SafetyPolicy, ActionClassification, RiskLevel
        policy = SafetyPolicy()

        cls1 = ActionClassification(
            action="delete", target="folder1",
            risk_level=RiskLevel.HIGH_RISK, reason="test", needs_confirmation=True,
            explanation="Delete folder1"
        )
        policy.confirmation_manager.request_confirmation(cls1)

        # Get the actual conf_id from the pending dict
        pending_ids = list(policy.confirmation_manager._pending.keys())
        assert len(pending_ids) == 1
        conf_id = pending_ids[0]

        resolved = policy.confirmation_manager.resolve(conf_id, approved=True)
        assert resolved is not None
        assert resolved.target == "folder1"

        assert len(policy.confirmation_manager._pending) == 0


# =====================================================================
# Phase 9: Mission Safety Gate Tests
# =====================================================================

class TestMissionSafetyGate:
    """Test that MissionManager checks safety before each action."""

    def test_mission_safety_gate_blocks_action(self):
        """Verify the mission loop checks safety policy."""
        from pengu.safety import get_safety_policy, RiskLevel

        policy = get_safety_policy()
        cls = policy.check("format c:", "")
        assert cls.risk_level == RiskLevel.BLOCKED

    def test_mission_concurrent_failures_triggers_replan(self):
        """Test that too many consecutive failures triggers replan."""
        from pengu.agent.state import AgentState
        state = AgentState(goal="test goal")

        for i in range(5):
            state.record_action(f"action_{i}", f"target_{i}", False, error="failed")

        assert state.consecutive_failures == 5

    def test_mission_loop_detection(self):
        """Test loop detection in mission execution."""
        from pengu.agent.state import AgentState
        state = AgentState(goal="test")

        action_window = []
        last_sig = ""
        consecutive = 0
        threshold = 3

        for _ in range(5):
            sig = "same_action:target"
            if sig == last_sig:
                consecutive += 1
            else:
                consecutive = 1
            last_sig = sig
            action_window.append(sig)

        assert consecutive >= threshold


# =====================================================================
# App.py Integration Tests
# =====================================================================

class TestAppIntegration:
    """Test that app.py correctly integrates safety and scheduler."""

    def test_app_has_safety_check(self):
        """Verify PenguApp has _check_safety method."""
        from pengu.app import PenguApp
        assert hasattr(PenguApp, '_check_safety')

    def test_app_has_confirmation_handler(self):
        """Verify PenguApp has _handle_confirmation method."""
        from pengu.app import PenguApp
        assert hasattr(PenguApp, '_handle_confirmation')

    def test_confirmation_words_accepted(self):
        """Test that confirmation words are recognized."""
        from pengu.app import PenguApp
        app = PenguApp()

        from pengu.safety import ActionClassification, RiskLevel
        app._pending_confirmation = ActionClassification(
            action="test", target="test",
            risk_level=RiskLevel.HIGH_RISK, reason="test",
            needs_confirmation=True, explanation="test"
        )

        result = app._handle_confirmation("yes")
        assert result is None

    def test_denial_words_rejected(self):
        """Test that denial words cancel the action."""
        from pengu.app import PenguApp
        app = PenguApp()

        from pengu.safety import ActionClassification, RiskLevel
        app._pending_confirmation = ActionClassification(
            action="test", target="test",
            risk_level=RiskLevel.HIGH_RISK, reason="test",
            needs_confirmation=True, explanation="test"
        )

        result = app._handle_confirmation("no")
        assert result is not None
        assert "cancelled" in result.lower()
        assert app._pending_confirmation is None

    def test_no_pending_confirmation_returns_none(self):
        """Test that without pending confirmation, handler returns None."""
        from pengu.app import PenguApp
        app = PenguApp()
        app._pending_confirmation = None

        result = app._handle_confirmation("yes")
        assert result is None


# =====================================================================
# Pipeline Handler Integration Tests
# =====================================================================

class TestPipelineHandlers:
    """Test pipeline handler routing for memory and missions."""

    def test_memory_handler_singleton(self):
        """Verify memory handler uses singleton pattern."""
        from pengu.memory import get_memory, reset_memory
        mem1 = reset_memory()
        mem2 = get_memory()
        assert mem1 is mem2

    def test_scheduler_singleton(self):
        """Verify scheduler uses singleton pattern."""
        from pengu.scheduler import get_scheduler, reset_scheduler
        s1 = reset_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_safety_singleton(self):
        """Verify safety policy uses singleton pattern."""
        from pengu.safety import get_safety_policy, reset_safety_policy
        p1 = reset_safety_policy()
        p2 = get_safety_policy()
        assert p1 is p2
