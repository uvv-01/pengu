"""
Tests for Day 3 OS control tools.
"""

import asyncio
import os
import tempfile
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Application Manager tests
# ---------------------------------------------------------------------------


class TestAppManager:
    """Tests for application discovery, open, close, is_running."""

    def test_resolve_known_app(self):
        from pengu.os.app_manager import get_app_manager, AppManager
        manager = AppManager()
        app = manager.resolve("code")
        assert app is not None
        assert app.name == "code"
        assert app.display_name == "VS Code"

    def test_resolve_alias(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        app = manager.resolve("vs code")
        assert app is not None
        assert app.name == "code"

    def test_resolve_alias_visual_studio_code(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        app = manager.resolve("visual studio code")
        assert app is not None
        assert app.name == "code"

    def test_resolve_unknown_app(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        app = manager.resolve("nonexistent_app_xyz_123")
        assert app is None

    def test_resolve_case_insensitive(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        app = manager.resolve("CHROME")
        assert app is not None
        assert app.name == "chrome"

    def test_discover_installs(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        manager.discover()
        # At least some apps should be installed
        installed = manager.list_installed()
        assert isinstance(installed, list)

    def test_list_all(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        all_apps = manager.list_all()
        assert len(all_apps) > 10
        assert all(isinstance(a, dict) for a in all_apps)
        assert all("name" in a for a in all_apps)

    def test_list_installed_format(self):
        from pengu.os.app_manager import AppManager
        manager = AppManager()
        installed = manager.list_installed()
        for app in installed:
            assert "name" in app
            assert "display_name" in app
            assert "installed" in app
            assert app["installed"] is True


class TestAppOpen:
    """Tests for application opening."""

    @pytest.mark.asyncio
    async def test_open_unknown_app(self):
        from pengu.tools.deterministic import app_open
        result = await app_open("nonexistent_app_xyz_123")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_open_list_installed(self):
        from pengu.tools.deterministic import app_list_installed
        result = await app_list_installed()
        assert result.success is True
        assert "applications" in result.output

    @pytest.mark.asyncio
    async def test_app_is_running_unknown(self):
        from pengu.tools.deterministic import app_is_running
        result = await app_is_running("nonexistent_app_xyz_123")
        assert result.success is False
        assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# Process Manager tests
# ---------------------------------------------------------------------------


class TestProcessManager:
    """Tests for process listing and inspection."""

    def test_list_processes(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        procs = pm.list_processes(max_results=5)
        assert len(procs) > 0
        assert all(hasattr(p, "pid") for p in procs)
        assert all(hasattr(p, "name") for p in procs)

    def test_list_processes_with_filter(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        # Filter for Python processes (we know Python is running)
        procs = pm.list_processes(name_filter="python", max_results=5)
        assert len(procs) > 0
        assert all("python" in p.name.lower() for p in procs)

    def test_get_process_info(self):
        from pengu.os.process_manager import ProcessManager
        import os
        pm = ProcessManager()
        # Get our own PID
        proc = pm.get_process(os.getpid())
        assert proc is not None
        assert proc.pid == os.getpid()

    def test_get_process_nonexistent(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        proc = pm.get_process(99999999)
        assert proc is None

    def test_is_running(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        # Python should be running
        assert pm.is_running("python") is True

    def test_is_not_running(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        assert pm.is_running("nonexistent_app_xyz_12345") is False

    def test_terminate_protected(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        # Try to terminate a protected process by PID 0 (system idle on Windows)
        result = pm.terminate(0)
        # Should refuse
        assert result["success"] is False

    def test_top_memory(self):
        from pengu.os.process_manager import ProcessManager
        pm = ProcessManager()
        top = pm.get_top_memory(n=5)
        assert len(top) <= 5
        # Should be sorted by memory descending
        for i in range(len(top) - 1):
            assert top[i].memory_mb >= top[i + 1].memory_mb

    def test_process_info_dict(self):
        from pengu.os.process_manager import ProcessManager
        import os
        pm = ProcessManager()
        proc = pm.get_process(os.getpid())
        d = proc.to_dict()
        assert "pid" in d
        assert "name" in d
        assert "memory_mb" in d
        assert d["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# Secure Filesystem tests
# ---------------------------------------------------------------------------


class TestSecureFilesystem:
    """Tests for path validation and sensitive file blocking."""

    def test_is_sensitive_env(self):
        from pengu.os.filesystem import is_sensitive_path
        assert is_sensitive_path("/home/user/.env") is True
        assert is_sensitive_path("/home/user/.env.local") is True
        assert is_sensitive_path("/home/user/config.py") is False

    def test_is_sensitive_ssh(self):
        from pengu.os.filesystem import is_sensitive_path
        assert is_sensitive_path("/home/user/.ssh/id_rsa") is True
        assert is_sensitive_path("/home/user/.ssh/authorized_keys") is True

    def test_is_sensitive_pem(self):
        from pengu.os.filesystem import is_sensitive_path
        assert is_sensitive_path("/home/user/cert.pem") is True
        assert is_sensitive_path("/home/user/key.pem") is True

    def test_is_not_sensitive_normal(self):
        from pengu.os.filesystem import is_sensitive_path
        assert is_sensitive_path("/home/user/README.md") is False
        assert is_sensitive_path("/home/user/src/main.py") is False
        assert is_sensitive_path("/home/user/project/config.json") is False

    def test_validate_path_normal(self):
        from pengu.os.filesystem import validate_path
        valid, error = validate_path("/tmp/test_file.txt", "read")
        # Path may not exist, that's ok for this test
        assert isinstance(valid, bool)

    def test_validate_path_sensitive(self):
        from pengu.os.filesystem import validate_path
        valid, error = validate_path("/home/user/.env", "read")
        assert valid is False
        assert "sensitive" in error.lower()

    @pytest.mark.asyncio
    async def test_safe_list_directory(self):
        from pengu.os.filesystem import safe_list_directory
        result = await safe_list_directory(".")
        assert result["success"] is True
        assert "entries" in result
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_safe_list_nonexistent(self):
        from pengu.os.filesystem import safe_list_directory
        result = await safe_list_directory("/nonexistent/path/xyz_123")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_safe_read_file(self):
        import tempfile
        from pengu.os.filesystem import safe_read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, Pengu!")
            tmp_path = f.name
        try:
            result = await safe_read_file(tmp_path)
            assert result["success"] is True
            assert result["content"] == "Hello, Pengu!"
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_safe_read_sensitive_blocked(self):
        from pengu.os.filesystem import safe_read_file
        result = await safe_read_file("/home/user/.env")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_safe_write_file(self):
        import tempfile
        from pengu.os.filesystem import safe_write_file
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "test_write.txt")
        try:
            result = await safe_write_file(tmp_path, "test content", confirm_overwrite=False)
            assert result["success"] is True
            assert os.path.exists(tmp_path)
            with open(tmp_path) as f:
                assert f.read() == "test content"
        finally:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)

    @pytest.mark.asyncio
    async def test_safe_write_sensitive_blocked(self):
        from pengu.os.filesystem import safe_write_file
        result = await safe_write_file("/home/user/.env", "MALICIOUS")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_safe_grep(self):
        from pengu.os.filesystem import safe_grep
        result = await safe_grep("def ", path=".", file_pattern="*.py", max_results=5)
        assert result["success"] is True
        assert "results" in result

    @pytest.mark.asyncio
    async def test_safe_search_files(self):
        from pengu.os.filesystem import safe_search_files
        result = await safe_search_files("*.py", path=".", max_results=5)
        assert result["success"] is True
        assert result["count"] > 0


# ---------------------------------------------------------------------------
# Safe Terminal tests
# ---------------------------------------------------------------------------


class TestSafeTerminal:
    """Tests for terminal safety validation."""

    def test_validate_safe_command(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("git status")
        assert safe is True

    def test_validate_blocked_format(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("format C:")
        assert safe is False
        assert "blocked" in reason.lower()

    def test_validate_blocked_del(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("del /s /q C:\\*")
        assert safe is False

    def test_validate_blocked_shutdown(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("shutdown /s /t 0")
        assert safe is False

    def test_validate_blocked_encoded_ps(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("powershell -enc ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
        assert safe is False

    def test_validate_empty(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        safe, reason = ts.validate("")
        assert safe is False

    def test_get_command_family(self):
        from pengu.os.terminal import TerminalSecurity
        ts = TerminalSecurity()
        assert ts.get_command_family("git status") == "git"
        assert ts.get_command_family("python --version") == "python"
        assert ts.get_command_family("echo hello") == "echo"

    @pytest.mark.asyncio
    async def test_execute_safe_command(self):
        from pengu.os.terminal import SafeTerminal
        terminal = SafeTerminal()
        result = await terminal.execute("echo hello")
        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_execute_blocked_command(self):
        from pengu.os.terminal import SafeTerminal
        terminal = SafeTerminal()
        result = await terminal.execute("format C:")
        assert result["success"] is False
        assert result.get("blocked") is True




# ---------------------------------------------------------------------------
# System Info tests
# ---------------------------------------------------------------------------


class TestSystemInfo:
    """Tests for system information tools."""

    def test_get_system_info(self):
        from pengu.os.system_info import get_system_info
        info = get_system_info()
        assert "os" in info
        assert "cpu" in info
        assert "ram" in info
        assert "storage" in info
        assert "tier" in info

    def test_system_info_has_values(self):
        from pengu.os.system_info import get_system_info
        info = get_system_info()
        assert info["os"]["name"] != ""
        assert info["cpu"]["cores_logical"] > 0
        assert info["ram"]["total_gb"] > 0
        assert info["tier"] in ("LOW", "MEDIUM", "HIGH")

    def test_system_info_summary(self):
        from pengu.os.system_info import get_system_info_summary
        summary = get_system_info_summary()
        assert "OS:" in summary
        assert "CPU:" in summary
        assert "RAM:" in summary
        assert "Tier:" in summary

    @pytest.mark.asyncio
    async def test_system_info_tool(self):
        from pengu.tools.deterministic import system_info
        result = await system_info()
        assert result.success is True
        assert "summary" in result.output


# ---------------------------------------------------------------------------
# VS Code Integration tests
# ---------------------------------------------------------------------------


class TestVSCode:
    """Tests for VS Code integration."""

    def test_vscode_available_check(self):
        from pengu.os.vscode import _is_vscode_available
        # Just verify the function exists and returns a bool
        result = _is_vscode_available()
        assert isinstance(result, bool)

    def test_resolve_path_nonexistent(self):
        from pengu.os.vscode import _resolve_path
        result = _resolve_path("nonexistent_folder_xyz_123")
        assert result is None

    def test_resolve_path_current_dir(self):
        from pengu.os.vscode import _resolve_path
        result = _resolve_path(".")
        assert result is not None
        assert result.exists()

    @pytest.mark.asyncio
    async def test_vscode_open_folder_not_found(self):
        from pengu.tools.deterministic import vscode_open_folder
        result = await vscode_open_folder("nonexistent_folder_xyz_123")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_vscode_open_file_not_found(self):
        from pengu.tools.deterministic import vscode_open_file
        result = await vscode_open_file("nonexistent_file_xyz_123.py")
        assert result.success is False


# ---------------------------------------------------------------------------
# Tool Registry integration tests
# ---------------------------------------------------------------------------


class TestToolRegistryDay3:
    """Tests that Day 3 tools are properly registered."""

    def test_all_new_tools_registered(self):
        from pengu.tools.registry import ToolRegistry
        from pengu.tools.deterministic import register_deterministic_tools

        registry = ToolRegistry()
        register_deterministic_tools(registry)

        tool_names = [t.name for t in registry.list_tools()]

        # Day 3 tools
        assert "system.info" in tool_names
        assert "process.list" in tool_names
        assert "process.info" in tool_names
        assert "process.terminate" in tool_names
        assert "application.close" in tool_names
        assert "application.is_running" in tool_names
        assert "application.list_installed" in tool_names
        assert "vscode.open_folder" in tool_names
        assert "vscode.open_file" in tool_names
        assert "vscode.focus" in tool_names

        # Day 1+2 tools still present
        assert "filesystem.read_file" in tool_names
        assert "filesystem.write_file" in tool_names
        assert "filesystem.list_directory" in tool_names
        assert "terminal.execute" in tool_names
        assert "application.open" in tool_names
        assert "git.status" in tool_names
        assert "git.log" in tool_names
        assert "git.diff" in tool_names

    def test_total_tool_count(self):
        from pengu.tools.registry import ToolRegistry
        from pengu.tools.deterministic import register_deterministic_tools

        registry = ToolRegistry()
        register_deterministic_tools(registry)
        # Should have 24+ tools now
        assert len(registry.list_tools()) >= 24

    def test_tool_permission_levels(self):
        from pengu.tools.registry import ToolRegistry
        from pengu.tools.deterministic import register_deterministic_tools
        from pengu.config import PermissionLevel

        registry = ToolRegistry()
        register_deterministic_tools(registry)

        # SAFE tools
        safe_tools = registry.list_by_permission(PermissionLevel.SAFE)
        safe_names = [t.name for t in safe_tools]
        assert "system.info" in safe_names
        assert "process.list" in safe_names
        assert "application.is_running" in safe_names

        # HIGH_RISK tools
        high_risk = [t for t in registry.list_tools() if t.permission_level == PermissionLevel.HIGH_RISK]
        high_risk_names = [t.name for t in high_risk]
        assert "terminal.execute" in high_risk_names
        assert "process.terminate" in high_risk_names
