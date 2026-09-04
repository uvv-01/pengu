"""
Pipeline handlers for Day 5 features: memory, web search, browser.
"""

from __future__ import annotations

import re
from typing import Any

from pengu.config import TaskCategory
from pengu.logging import get_logger
from pengu.router import Intent

logger = get_logger("pengu.pipeline.handlers")


async def handle_memory(
    text: str,
    intent: Intent,
    tool_registry: Any,
    steps: list[dict[str, Any]],
    PipelineResult: type,
) -> Any:
    """Handle memory operations."""
    steps.append({"step": "memory", "status": "running"})

    from pengu.memory import get_memory, MemoryCategory, MemoryType

    memory = get_memory()
    if not memory._initialized:
        await memory.initialize()
    text_lower = text.lower()

    # Save/remember
    if any(w in text_lower for w in ["save", "remember", "store"]):
        content_match = None
        for pattern in [
            r"remember\s+(?:that\s+)?(.+)",
            r"save\s+(?:that\s+)?(.+)",
            r"store\s+(?:that\s+)?(.+)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                content_match = m.group(1).strip()
                break

        if content_match:
            try:
                entry = await memory.save(
                    content=content_match,
                    category=MemoryCategory.GENERAL,
                    memory_type=MemoryType.PERSISTENT,
                )
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Remembered: {content_match}",
                    provider="deterministic", tool_used="memory.save",
                )
            except ValueError as e:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Could not save memory: {e}",
                    provider="deterministic", error=str(e),
                )
        else:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text, intent=intent,
                response="What would you like me to remember?",
                provider="deterministic",
            )

    # Search
    elif any(w in text_lower for w in ["search", "find", "look up"]):
        query_match = None
        for pattern in [
            r"search\s+(?:for\s+)?(.+)",
            r"find\s+(?:.*\s+)?(.+)",
            r"look\s+up\s+(.+)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                query_match = m.group(1).strip()
                break

        if query_match:
            results = await memory.search(query_match, limit=5)
            if results:
                lines = [f"- {r.content} ({r.category.value})" for r in results]
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Found {len(results)} memories:\n" + "\n".join(lines),
                    provider="deterministic", tool_used="memory.search",
                )
            else:
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"No memories found for '{query_match}'.",
                    provider="deterministic", tool_used="memory.search",
                )

    # List
    elif any(w in text_lower for w in ["list", "show all"]):
        memories = await memory.list_all(limit=10)
        if memories:
            lines = [f"- {r.content} ({r.category.value})" for r in memories]
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response="Your memories:\n" + "\n".join(lines),
                provider="deterministic", tool_used="memory.list",
            )
        else:
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response="No memories stored yet.",
                provider="deterministic", tool_used="memory.list",
            )

    # Clear
    elif any(w in text_lower for w in ["clear", "delete all"]):
        count = await memory.clear()
        steps[-1]["status"] = "complete"
        return PipelineResult(
            text=text, intent=intent,
            response=f"Cleared {count} memories.",
            provider="deterministic", tool_used="memory.clear",
        )

    # Forget/delete
    elif any(w in text_lower for w in ["forget", "delete", "remove"]):
        # Try to delete by searching for the topic
        query_match = None
        for pattern in [
            r"forget\s+(?:about\s+)?(.+)",
            r"delete\s+(?:memory\s+(?:about\s+)?)(.+)",
            r"remove\s+(?:memory\s+(?:about\s+)?)(.+)",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                query_match = m.group(1).strip()
                break

        if query_match:
            results = await memory.search(query_match, limit=10)
            if results:
                deleted = 0
                for entry in results:
                    if await memory.delete(entry.id):
                        deleted += 1
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Forgot {deleted} memor{'y' if deleted == 1 else 'ies'} about '{query_match}'.",
                    provider="deterministic", tool_used="memory.delete",
                )
            else:
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"I don't have any memories about '{query_match}' to forget.",
                    provider="deterministic", tool_used="memory.delete",
                )
        else:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text, intent=intent,
                response="What would you like me to forget?",
                provider="deterministic",
            )

    # Recall/what do you know
    elif any(w in text_lower for w in ["what do you know", "recall", "what did i tell you"]):
        memories = await memory.list_all(limit=10)
        if memories:
            lines = [f"- {r.content} ({r.category.value})" for r in memories]
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response="Here is what I remember:\n" + "\n".join(lines),
                provider="deterministic", tool_used="memory.list",
            )
        else:
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response="I don't have any memories stored yet.",
                provider="deterministic", tool_used="memory.list",
            )

    steps[-1]["status"] = "error"
    return PipelineResult(
        text=text, intent=intent,
        response="I can help you save, search, list, forget, or clear memories. What would you like to do?",
        provider="deterministic",
    )


async def handle_web_search(
    text: str,
    intent: Intent,
    tool_registry: Any,
    steps: list[dict[str, Any]],
    PipelineResult: type,
) -> Any:
    """Handle web search operations."""
    steps.append({"step": "web_search", "status": "running"})

    query = intent.extracted_target
    if not query:
        match = re.search(
            r"(?:search|google|look\s+up|find)\s+(?:for\s+)?(.+)",
            text, re.IGNORECASE,
        )
        if match:
            query = match.group(1).strip()

    if not query:
        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text, intent=intent,
            response="What would you like me to search for?",
            provider="deterministic",
        )

    try:
        from pengu.web.search import get_search_provider
        provider = get_search_provider()
        results = await provider.search(query, max_results=5)

        if results:
            lines = [
                f"{i+1}. **{r.title}**\n   {r.url}\n   {r.snippet}"
                for i, r in enumerate(results)
            ]
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response=f"Search results for '{query}':\n\n" + "\n\n".join(lines),
                provider="deterministic", tool_used="web_search",
            )
        else:
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text, intent=intent,
                response=f"No results found for '{query}'.",
                provider="deterministic", tool_used="web_search",
            )
    except Exception as e:
        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text, intent=intent,
            response=f"Web search failed: {e}",
            provider="deterministic", error=str(e),
        )


async def handle_browser(
    text: str,
    intent: Intent,
    tool_registry: Any,
    steps: list[dict[str, Any]],
    PipelineResult: type,
) -> Any:
    """Handle browser operations."""
    steps.append({"step": "browser", "status": "running"})

    url_match = re.search(r"(https?://\S+)", text)
    if not url_match:
        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text, intent=intent,
            response="What URL would you like me to open?",
            provider="deterministic",
        )

    url = url_match.group(1)

    try:
        from pengu.web.browser import get_browser
        browser = get_browser()
        page = await browser.open(url)
        steps[-1]["status"] = "complete"
        return PipelineResult(
            text=text, intent=intent,
            response=f"Opened {page.title} ({url})",
            provider="deterministic", tool_used="browser.open",
        )
    except Exception as e:
        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text, intent=intent,
            response=f"Browser error: {e}",
            provider="deterministic", error=str(e),
        )


async def handle_missions(
    text: str,
    intent: Intent,
    tool_registry: Any,
    steps: list[dict[str, Any]],
    PipelineResult: type,
) -> Any:
    """Handle mission/scheduler commands: reminders, background tasks, scheduling."""
    steps.append({"step": "missions", "status": "running"})
    text_lower = text.lower().strip()

    from pengu.scheduler import (
        Scheduler, Schedule, ScheduleType, MissionState,
        parse_schedule, get_scheduler,
    )

    scheduler = get_scheduler()

    # --- Create reminder/mission ---
    if any(w in text_lower for w in ["remind", "reminder", "schedule", "check every", "watch"]):
        # Extract the task and schedule
        task = text
        schedule = None

        # Try to extract schedule from text
        for phrase in [
            r"remind me\s+(.+)",
            r"schedule\s+(.+)",
            r"check every\s+(.+)",
            r"watch\s+(.+)",
        ]:
            m = re.search(phrase, text, re.IGNORECASE)
            if m:
                rest = m.group(1).strip()
                schedule = parse_schedule(rest)
                if schedule:
                    task = rest
                    break

        if not schedule:
            # Default: try the full text
            schedule = parse_schedule(text)

        if schedule:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                mission = loop.run_until_complete(
                    scheduler.create_mission(
                        name=text[:60],
                        task=task,
                        schedule=schedule,
                        description=schedule.description,
                    )
                )
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Scheduled: {mission.name} ({schedule.description})",
                    provider="deterministic", tool_used="scheduler.create",
                )
            finally:
                loop.close()
        else:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text, intent=intent,
                response="When should I schedule that? Try 'in 30 minutes' or 'every day at 9am'.",
                provider="deterministic",
            )

    # --- List missions ---
    if any(w in text_lower for w in ["what are you doing", "what tasks", "what is scheduled", "list tasks", "list missions"]):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            missions = loop.run_until_complete(scheduler.initialize()) or []
            active = scheduler.list_missions()
            if active:
                lines = []
                for m in active[:5]:
                    status = m.state.value
                    schedule_desc = m.schedule.description or m.schedule.schedule_type.value
                    lines.append(f"  - {m.name} [{status}] ({schedule_desc})")
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=f"Scheduled tasks:\n" + "\n".join(lines),
                    provider="deterministic", tool_used="scheduler.list",
                )
            else:
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response="No scheduled tasks.",
                    provider="deterministic", tool_used="scheduler.list",
                )
        finally:
            loop.close()

    # --- Cancel mission ---
    cancel_match = re.search(r"cancel\s+(?:the\s+)?(?:task|mission|reminder|schedule)", text_lower)
    if cancel_match:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            active = scheduler.list_missions(state=MissionState.QUEUED)
            if active:
                mission = active[-1]  # cancel the most recent
                result = loop.run_until_complete(scheduler.cancel_mission(mission.id))
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response=result,
                    provider="deterministic", tool_used="scheduler.cancel",
                )
            else:
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text, intent=intent,
                    response="No active tasks to cancel.",
                    provider="deterministic",
                )
        finally:
            loop.close()

    steps[-1]["status"] = "error"
    return PipelineResult(
        text=text, intent=intent,
        response="What would you like me to schedule? Try 'remind me in 30 minutes' or 'check every day at 9am'.",
        provider="deterministic",
    )
