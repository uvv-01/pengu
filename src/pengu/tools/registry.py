"""
Tool Registry — typed tools with permission levels and execution.

Every OS capability is a Tool:
  - filesystem.read_file
  - terminal.execute
  - application.open
  - git.status
  - etc.

Each tool has:
  - name, description, parameter schema
  - permission level (SAFE, LOW_RISK, HIGH_RISK, CRITICAL)
  - execution function
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pengu.config import PermissionLevel
from pengu.logging import AuditLogger, get_logger

logger = get_logger("pengu.tools")
audit = AuditLogger()


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class Tool:
    """A single tool definition."""

    name: str
    description: str
    category: str  # "filesystem", "terminal", "application", etc.
    permission_level: PermissionLevel
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema
    handler: Optional[Callable] = None
    requires_confirmation: bool = False
    enabled: bool = True

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        if not self.enabled:
            return ToolResult(success=False, error=f"Tool '{self.name}' is disabled")

        if self.handler is None:
            return ToolResult(success=False, error=f"Tool '{self.name}' has no handler")

        import time
        start = time.perf_counter()

        try:
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**kwargs)
            else:
                result = self.handler(**kwargs)

            duration = (time.perf_counter() - start) * 1000

            if isinstance(result, ToolResult):
                result.duration_ms = duration
                return result

            return ToolResult(success=True, output=result, duration_ms=duration)

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.error("tool_execution_error", tool=self.name, error=str(e))
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=duration,
            )

    def to_schema(self) -> dict[str, Any]:
        """Return OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning("tool_overwrite", name=tool.name)
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name, category=tool.category)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_enabled(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.enabled]

    def list_by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category and t.enabled]

    def list_by_permission(self, max_level: PermissionLevel) -> list[Tool]:
        return [t for t in self._tools.values() if t.permission_level.value <= max_level.value]

    def get_schemas(self, tools: list[Tool] | None = None) -> list[dict[str, Any]]:
        """Get OpenAI-style function schemas for LLM tool calling."""
        target = tools or self.list_enabled()
        return [t.to_schema() for t in target]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with audit logging."""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        result = await tool.execute(**kwargs)

        audit.log_tool_execution(
            tool_name=name,
            params=kwargs,
            permission_level=tool.permission_level.value,
            granted=result.success,
            result=str(result.output)[:200] if result.output else result.error[:200],
        )

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": len(self._tools),
            "enabled": len(self.list_enabled()),
            "tools": [
                {
                    "name": t.name,
                    "category": t.category,
                    "permission": t.permission_level.name,
                    "enabled": t.enabled,
                }
                for t in self._tools.values()
            ],
        }
