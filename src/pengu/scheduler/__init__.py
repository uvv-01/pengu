"""
Scheduler - background task scheduling and mission persistence.
"""

from __future__ import annotations
import asyncio, json, re, time, uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from pengu.logging import get_logger

logger = get_logger("pengu.scheduler")


class MissionState(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduleType(str, Enum):
    ONCE = "once"
    DELAYED = "delayed"
    RECURRING = "recurring"


@dataclass
class Schedule:
    schedule_type: ScheduleType
    run_at: float = 0.0
    delay_seconds: float = 0.0
    interval_seconds: float = 0.0
    end_at: float = 0.0
    description: str = ""

    def get_next_run(self, after=0.0):
        if self.schedule_type == ScheduleType.ONCE:
            return self.run_at
        elif self.schedule_type == ScheduleType.DELAYED:
            return after + self.delay_seconds
        elif self.schedule_type == ScheduleType.RECURRING:
            if after == 0.0:
                return time.time() + self.interval_seconds
            nxt = after + self.interval_seconds
            return 0.0 if (self.end_at and nxt > self.end_at) else nxt
        return 0.0

    def to_dict(self):
        return {"type": self.schedule_type.value, "run_at": self.run_at,
                "delay_seconds": self.delay_seconds, "interval_seconds": self.interval_seconds,
                "end_at": self.end_at, "description": self.description}

    @classmethod
    def from_dict(cls, d):
        return cls(schedule_type=ScheduleType(d.get("type", "once")),
                   run_at=d.get("run_at", 0.0), delay_seconds=d.get("delay_seconds", 0.0),
                   interval_seconds=d.get("interval_seconds", 0.0),
                   end_at=d.get("end_at", 0.0), description=d.get("description", ""))


@dataclass
class ScheduledMission:
    id: str
    name: str
    description: str
    task: str
    schedule: Schedule
    state: MissionState = MissionState.CREATED
    created_at: float = field(default_factory=time.time)
    next_run: float = 0.0
    last_run: float = 0.0
    last_result: str = ""
    failure_count: int = 0
    max_retries: int = 3
    retry_delay: float = 60.0

    def to_dict(self):
        return {"id": self.id, "name": self.name, "description": self.description,
                "task": self.task, "schedule": self.schedule.to_dict(),
                "state": self.state.value, "created_at": self.created_at,
                "next_run": self.next_run, "last_run": self.last_run,
                "last_result": self.last_result[:200], "failure_count": self.failure_count}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d.get("id", uuid.uuid4().hex[:12]), name=d.get("name", ""),
                   description=d.get("description", ""), task=d.get("task", ""),
                   schedule=Schedule.from_dict(d.get("schedule", {})),
                   state=MissionState(d.get("state", "created")),
                   created_at=d.get("created_at", time.time()),
                   next_run=d.get("next_run", 0.0), last_run=d.get("last_run", 0.0),
                   last_result=d.get("last_result", ""), failure_count=d.get("failure_count", 0))


class Scheduler:
    def __init__(self, db_path=None):
        self._db_path = db_path or "data/pengu_missions.db"
        self._missions = {}
        self._task_executor = None
        self._running = False
        self._loop_task = None
        self._initialized = False

    def set_task_executor(self, executor):
        self._task_executor = executor

    async def initialize(self):
        import aiosqlite, os
        os.makedirs("data", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("CREATE TABLE IF NOT EXISTS scheduled_missions (id TEXT PRIMARY KEY, data TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
        await self._db.commit()
        cursor = await self._db.execute("SELECT data FROM scheduled_missions")
        for row in await cursor.fetchall():
            try:
                m = ScheduledMission.from_dict(json.loads(row[0]))
                if m.state not in (MissionState.CANCELLED, MissionState.COMPLETED):
                    self._missions[m.id] = m
            except Exception:
                pass
        self._initialized = True

    async def close(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        if hasattr(self, "_db") and self._db:
            await self._db.close()

    async def create_mission(self, name, task, schedule, description=""):
        m = ScheduledMission(id=f"mis_{uuid.uuid4().hex[:12]}", name=name,
            description=description or name, task=task, schedule=schedule,
            state=MissionState.QUEUED)
        m.next_run = schedule.get_next_run(time.time())
        self._missions[m.id] = m
        await self._persist(m)
        return m

    async def cancel_mission(self, mid):
        m = self._missions.get(mid)
        if not m:
            return f"Mission {mid} not found."
        m.state = MissionState.CANCELLED
        await self._persist(m)
        return f"Mission '{m.name}' cancelled."

    async def pause_mission(self, mid):
        m = self._missions.get(mid)
        if not m:
            return f"Mission {mid} not found."
        m.state = MissionState.PAUSED
        await self._persist(m)
        return f"Mission '{m.name}' paused."

    async def resume_mission(self, mid):
        m = self._missions.get(mid)
        if not m:
            return f"Mission {mid} not found."
        m.state = MissionState.QUEUED
        m.next_run = m.schedule.get_next_run(time.time())
        await self._persist(m)
        return f"Mission '{m.name}' resumed."

    def get_mission(self, mid):
        return self._missions.get(mid)

    def list_missions(self, state=None):
        ms = list(self._missions.values())
        if state:
            ms = [m for m in ms if m.state == state]
        ms.sort(key=lambda m: m.next_run or m.created_at)
        return ms

    async def start(self):
        if self._running:
            return
        if not self._initialized:
            await self.initialize()
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()

    async def _run_loop(self):
        while self._running:
            try:
                now = time.time()
                for m in list(self._missions.values()):
                    if m.state in (MissionState.CANCELLED, MissionState.COMPLETED,
                                   MissionState.PAUSED, MissionState.RUNNING):
                        continue
                    if now >= m.next_run and m.next_run > 0:
                        await self._execute_mission(m)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduler_error", error=str(e))
                await asyncio.sleep(10)

    async def _execute_mission(self, mission):
        mission.state = MissionState.RUNNING
        mission.last_run = time.time()
        try:
            if self._task_executor:
                result = await self._task_executor(mission.task)
                mission.last_result = str(result)
                mission.failure_count = 0
                mission.state = (MissionState.COMPLETED
                    if mission.schedule.schedule_type == ScheduleType.ONCE
                    else MissionState.QUEUED)
                if mission.schedule.schedule_type == ScheduleType.RECURRING:
                    mission.next_run = mission.schedule.get_next_run(time.time())
                    if mission.next_run == 0.0:
                        mission.state = MissionState.COMPLETED
                await self._persist(mission)
            else:
                mission.last_result = "No task executor configured"
                mission.state = MissionState.FAILED
                await self._persist(mission)
        except Exception as e:
            mission.failure_count += 1
            mission.last_result = f"Error: {e}"
            if mission.failure_count >= mission.max_retries:
                mission.state = MissionState.FAILED
            else:
                mission.state = MissionState.QUEUED
                mission.next_run = time.time() + mission.retry_delay
            await self._persist(mission)

    async def _persist(self, mission):
        if not self._initialized:
            return
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO scheduled_missions (id,data,created_at,updated_at) VALUES (?,?,?,?)",
                (mission.id, json.dumps(mission.to_dict()), mission.created_at, time.time()))
            await self._db.commit()
        except Exception:
            pass


def parse_schedule(text):
    tl = text.lower().strip()
    m = re.search(r"in (\d+) (minute|hour|second)s?", tl)
    if m:
        a, u = int(m.group(1)), m.group(2)
        d = a * ({"minute": 60, "hour": 3600}.get(u, 1))
        return Schedule(ScheduleType.DELAYED, delay_seconds=d, description=f"in {a} {u}")
    m = re.search(r"every (day|week|month)", tl)
    if m:
        unit = m.group(1)
        interval = {"day": 86400, "week": 604800, "month": 2592000}.get(unit, 86400)
        return Schedule(ScheduleType.RECURRING, interval_seconds=interval, description=f"every {unit}")
    m = re.search(r"every (\d+) (minute|hour|day|week)s?", tl)
    if m:
        a, u = int(m.group(1)), m.group(2)
        i = a * ({"minute": 60, "hour": 3600, "day": 86400, "week": 604800}.get(u, 3600))
        return Schedule(ScheduleType.RECURRING, interval_seconds=i, description=f"every {a} {u}")
    m = re.search(r"at (\d{1,2})(?::(\d{2}))?\s*(am|pm)", tl)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        now = datetime.now()
        t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if t <= now:
            t += timedelta(days=1)
        return Schedule(ScheduleType.ONCE, run_at=t.timestamp(), description=f"at {h:02d}:{mi:02d} {ap}")
    return None


_scheduler = None

def get_scheduler(db_path=None):
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler(db_path)
    return _scheduler

def reset_scheduler():
    global _scheduler
    _scheduler = Scheduler()
    return _scheduler
