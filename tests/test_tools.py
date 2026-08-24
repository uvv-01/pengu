"""Tests for the tool registry."""

import pytest

from pengu.config import PermissionLevel
from pengu.tools.registry import Tool, ToolRegistry, ToolResult


class TestToolResult:
    def test_success(self):
        result = ToolResult(success=True, output="ok")
        assert result.success is True
        d = result.to_dict()
        assert d["success"] is True

    def test_failure(self):
        result = ToolResult(success=False, error="something broke")
        assert result.success is False


class TestTool:
    @pytest.mark.asyncio
    async def test_sync_handler(self):
        tool = Tool(
            name="test.sync",
            description="A sync test tool",
            category="test",
            permission_level=PermissionLevel.SAFE,
            handler=lambda x=1: x + 1,
        )
        result = await tool.execute()
        assert result.success is True
        assert result.output == 2

    @pytest.mark.asyncio
    async def test_async_handler(self):
        async def handler(name: str = "world") -> str:
            return f"hello {name}"

        tool = Tool(
            name="test.async",
            description="An async test tool",
            category="test",
            permission_level=PermissionLevel.SAFE,
            handler=handler,
        )
        result = await tool.execute(name="pengu")
        assert result.success is True
        assert result.output == "hello pengu"

    @pytest.mark.asyncio
    async def test_disabled_tool(self):
        tool = Tool(
            name="test.disabled",
            description="Disabled",
            category="test",
            permission_level=PermissionLevel.SAFE,
            handler=lambda: "ok",
            enabled=False,
        )
        result = await tool.execute()
        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self):
        def bad_handler():
            raise ValueError("boom")

        tool = Tool(
            name="test.bad",
            description="Bad handler",
            category="test",
            permission_level=PermissionLevel.SAFE,
            handler=bad_handler,
        )
        result = await tool.execute()
        assert result.success is False
        assert "ValueError" in result.error

    def test_to_schema(self):
        tool = Tool(
            name="test.schema",
            description="Schema test",
            category="test",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        schema = tool.to_schema()
        assert schema["function"]["name"] == "test.schema"


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(
            name="reg.test",
            description="Test",
            category="test",
            permission_level=PermissionLevel.SAFE,
        )
        reg.register(tool)
        assert reg.get("reg.test") is tool

    def test_list_by_category(self):
        reg = ToolRegistry()
        reg.register(Tool("fs.read", "Read", "filesystem", PermissionLevel.SAFE))
        reg.register(Tool("fs.write", "Write", "filesystem", PermissionLevel.LOW_RISK))
        reg.register(Tool("term.exec", "Exec", "terminal", PermissionLevel.HIGH_RISK))

        fs_tools = reg.list_by_category("filesystem")
        assert len(fs_tools) == 2

    def test_list_by_permission(self):
        reg = ToolRegistry()
        reg.register(Tool("safe", "Safe", "test", PermissionLevel.SAFE))
        reg.register(Tool("low", "Low", "test", PermissionLevel.LOW_RISK))
        reg.register(Tool("high", "High", "test", PermissionLevel.HIGH_RISK))

        safe_tools = reg.list_by_permission(PermissionLevel.SAFE)
        assert len(safe_tools) == 1

        low_tools = reg.list_by_permission(PermissionLevel.LOW_RISK)
        assert len(low_tools) == 2

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="calc.add",
            description="Add",
            category="calc",
            permission_level=PermissionLevel.SAFE,
            handler=lambda a=1, b=2: a + b,
        ))
        result = await reg.execute("calc.add", a=3, b=4)
        assert result.success is True
        assert result.output == 7

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent")
        assert result.success is False

    def test_get_schemas(self):
        reg = ToolRegistry()
        reg.register(Tool("t1", "T1", "cat", PermissionLevel.SAFE))
        reg.register(Tool("t2", "T2", "cat", PermissionLevel.SAFE, enabled=False))
        schemas = reg.get_schemas()
        assert len(schemas) == 1  # only enabled

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register(Tool("del.me", "Delete", "test", PermissionLevel.SAFE))
        reg.unregister("del.me")
        assert reg.get("del.me") is None

    def test_to_dict(self):
        reg = ToolRegistry()
        reg.register(Tool("d.test", "Test", "test", PermissionLevel.SAFE))
        d = reg.to_dict()
        assert d["total"] == 1
