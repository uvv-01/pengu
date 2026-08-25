"""
Deterministic tools for Pengu — real OS control.

These tools execute actual operations on the user's machine.
No LLM needed for deterministic operations.

Categories:
  - filesystem: read, write, list, search, create, delete
  - terminal: execute shell commands
  - application: open, close, focus applications
  - git: status, diff, log, branch, checkout, add, commit
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pengu.config import PermissionLevel
from pengu.logging import get_logger
from pengu.tools.registry import Tool, ToolRegistry, ToolResult

logger = get_logger("pengu.tools.deterministic")


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------


async def fs_read_file(path: str, encoding: str = "utf-8") -> ToolResult:
    """Read a file's contents."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        if not p.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")
        if p.stat().st_size > 1_000_000:  # 1MB limit
            return ToolResult(
                success=False,
                error=f"File too large: {p.stat().st_size / 1024:.0f}KB (limit 1MB)",
            )
        content = p.read_text(encoding=encoding)
        return ToolResult(
            success=True,
            output={
                "path": str(p),
                "content": content,
                "size": p.stat().st_size,
                "lines": content.count("\n") + 1,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def fs_write_file(path: str, content: str, encoding: str = "utf-8") -> ToolResult:
    """Write content to a file."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return ToolResult(
            success=True,
            output={"path": str(p), "bytes_written": len(content.encode(encoding))},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def fs_list_directory(path: str = ".") -> ToolResult:
    """List directory contents."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return ToolResult(success=False, error=f"Directory not found: {path}")
        if not p.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}")

        entries = []
        for entry in sorted(p.iterdir()):
            stat = entry.stat()
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else 0,
                    "modified": stat.st_mtime,
                }
            )

        return ToolResult(
            success=True,
            output={"path": str(p), "entries": entries, "count": len(entries)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def fs_search_files(
    pattern: str, path: str = ".", max_results: int = 50
) -> ToolResult:
    """Search for files matching a glob pattern."""
    try:
        search_path = os.path.join(path, "**", pattern)
        matches = glob.glob(search_path, recursive=True)
        results = []
        for match in matches[:max_results]:
            p = Path(match)
            results.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "type": "dir" if p.is_dir() else "file",
                }
            )
        return ToolResult(
            success=True,
            output={"pattern": pattern, "results": results, "count": len(results)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def fs_grep(
    query: str, path: str = ".", file_pattern: str = "*", max_results: int = 50
) -> ToolResult:
    """Search file contents for a pattern (like grep)."""
    try:
        results = []
        search_path = os.path.join(path, "**", file_pattern)
        for file_path in glob.glob(search_path, recursive=True):
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if query.lower() in line.lower():
                            results.append(
                                {
                                    "file": file_path,
                                    "line": i,
                                    "content": line.strip()[:200],
                                }
                            )
                            if len(results) >= max_results:
                                return ToolResult(
                                    success=True,
                                    output={
                                        "query": query,
                                        "results": results,
                                        "count": len(results),
                                        "truncated": True,
                                    },
                                )
            except (PermissionError, OSError):
                continue

        return ToolResult(
            success=True,
            output={"query": query, "results": results, "count": len(results)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Terminal tools
# ---------------------------------------------------------------------------


async def terminal_execute(
    command: str,
    cwd: str | None = None,
    timeout: int = 30,
    shell: str = "powershell",
) -> ToolResult:
    """Execute a shell command."""
    try:
        if shell == "powershell":
            cmd_list = ["powershell", "-NoProfile", "-Command", command]
        elif shell == "cmd":
            cmd_list = ["cmd", "/c", command]
        else:
            cmd_list = ["bash", "-c", command]

        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
        )

        return ToolResult(
            success=result.returncode == 0,
            output={
                "command": command,
                "stdout": result.stdout[:5000] if result.stdout else "",
                "stderr": result.stderr[:2000] if result.stderr else "",
                "exit_code": result.returncode,
                "shell": shell,
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            error=f"Command timed out after {timeout}s: {command[:100]}",
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Application tools
# ---------------------------------------------------------------------------

# Known application commands (Windows)
APP_COMMANDS: dict[str, dict[str, str]] = {
    "code": {"command": "code", "description": "VS Code"},
    "chrome": {"command": "start chrome", "description": "Google Chrome"},
    "firefox": {"command": "start firefox", "description": "Mozilla Firefox"},
    "msedge": {"command": "start msedge", "description": "Microsoft Edge"},
    "wt": {"command": "wt", "description": "Windows Terminal"},
    "explorer": {"command": "explorer", "description": "File Explorer"},
    "notepad": {"command": "notepad", "description": "Notepad"},
    "pwsh": {"command": "pwsh", "description": "PowerShell"},
    "cmd": {"command": "cmd", "description": "Command Prompt"},
    "cursor": {"command": "cursor", "description": "Cursor"},
    "intellij": {"command": "idea64", "description": "IntelliJ IDEA"},
    "pycharm": {"command": "pycharm64", "description": "PyCharm"},
    "webstorm": {"command": "webstorm64", "description": "WebStorm"},
    "git-bash": {"command": "start git-bash", "description": "Git Bash"},
}


async def app_open(
    application: str, arguments: str = "", cwd: str | None = None
) -> ToolResult:
    """Open an application."""
    try:
        # Normalize application name
        app_lower = application.lower().strip()

        # Check known apps
        if app_lower in APP_COMMANDS:
            cmd = APP_COMMANDS[app_lower]["command"]
            if arguments:
                cmd = f"{cmd} {arguments}"
            elif cwd:
                cmd = f"{cmd} {cwd}"

            subprocess.Popen(
                cmd,
                shell=True,
                cwd=None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return ToolResult(
                success=True,
                output={
                    "application": APP_COMMANDS[app_lower]["description"],
                    "command": cmd,
                    "status": "launched",
                },
            )

        # Try to find the application on PATH
        path = shutil.which(app_lower)
        if path:
            cmd_list = [path]
            if arguments:
                cmd_list.extend(arguments.split())

            subprocess.Popen(
                cmd_list,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            return ToolResult(
                success=True,
                output={
                    "application": application,
                    "path": path,
                    "status": "launched",
                },
            )

        # Try Windows 'start' command as last resort
        if platform.system() == "Windows":
            cmd = f"start {application}"
            if arguments:
                cmd += f" {arguments}"
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ToolResult(
                success=True,
                output={
                    "application": application,
                    "command": cmd,
                    "status": "launched via start",
                },
            )

        return ToolResult(
            success=False,
            error=f"Application not found: {application}. Known apps: {', '.join(APP_COMMANDS.keys())}",
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def app_list_running() -> ToolResult:
    """List running applications."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object Name, MainWindowTitle | ConvertTo-Json"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )
            import json
            try:
                processes = json.loads(result.stdout)
                if isinstance(processes, dict):
                    processes = [processes]
                apps = [
                    {"name": p.get("Name", ""), "title": p.get("MainWindowTitle", "")}
                    for p in processes
                    if p.get("MainWindowTitle")
                ]
                return ToolResult(
                    success=True,
                    output={"applications": apps, "count": len(apps)},
                )
            except json.JSONDecodeError:
                return ToolResult(
                    success=True,
                    output={"raw": result.stdout[:2000]},
                )
        else:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return ToolResult(
                success=True,
                output={"raw": result.stdout[:3000]},
            )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------


async def git_execute(args: list[str], cwd: str | None = None) -> ToolResult:
    """Execute a git command."""
    try:
        cmd_list = ["git"] + args
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
        )
        return ToolResult(
            success=result.returncode == 0,
            output={
                "command": "git " + " ".join(args),
                "stdout": result.stdout[:5000] if result.stdout else "",
                "stderr": result.stderr[:2000] if result.stderr else "",
                "exit_code": result.returncode,
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="Git command timed out")
    except FileNotFoundError:
        return ToolResult(success=False, error="Git is not installed or not on PATH")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def git_status(cwd: str | None = None) -> ToolResult:
    """Get git status."""
    return await git_execute(["status", "--short"], cwd=cwd)


async def git_log(cwd: str | None = None, n: int = 10) -> ToolResult:
    """Get recent git log."""
    return await git_execute(["log", f"--oneline", f"-{n}"], cwd=cwd)


async def git_diff(cwd: str | None = None) -> ToolResult:
    """Get git diff."""
    return await git_execute(["diff"], cwd=cwd)


# ---------------------------------------------------------------------------
# Network tools
# ---------------------------------------------------------------------------


async def network_list_wifi() -> ToolResult:
    """List available Wi-Fi networks."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        return ToolResult(
            success=result.returncode == 0,
            output={"raw": result.stdout[:5000]},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def network_wifi_status() -> ToolResult:
    """Get current Wi-Fi status."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        return ToolResult(
            success=result.returncode == 0,
            output={"raw": result.stdout[:3000]},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_deterministic_tools(registry: ToolRegistry) -> None:
    """Register all deterministic tools in the registry."""
    tools = [
        # Filesystem
        Tool(
            name="filesystem.read_file",
            description="Read a file's contents",
            category="filesystem",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path"],
            },
            handler=fs_read_file,
        ),
        Tool(
            name="filesystem.write_file",
            description="Write content to a file",
            category="filesystem",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {"type": "string", "default": "utf-8"},
                },
                "required": ["path", "content"],
            },
            handler=fs_write_file,
        ),
        Tool(
            name="filesystem.list_directory",
            description="List directory contents",
            category="filesystem",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                },
            },
            handler=fs_list_directory,
        ),
        Tool(
            name="filesystem.search_files",
            description="Search for files matching a glob pattern",
            category="filesystem",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["pattern"],
            },
            handler=fs_search_files,
        ),
        Tool(
            name="filesystem.grep",
            description="Search file contents for a text pattern",
            category="filesystem",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                    "path": {"type": "string", "default": "."},
                    "file_pattern": {"type": "string", "default": "*"},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["query"],
            },
            handler=fs_grep,
        ),
        # Terminal
        Tool(
            name="terminal.execute",
            description="Execute a shell command",
            category="terminal",
            permission_level=PermissionLevel.HIGH_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "cwd": {"type": "string", "description": "Working directory"},
                    "timeout": {"type": "integer", "default": 30},
                    "shell": {"type": "string", "enum": ["powershell", "cmd", "bash"], "default": "powershell"},
                },
                "required": ["command"],
            },
            handler=terminal_execute,
        ),
        # Application
        Tool(
            name="application.open",
            description="Open an application",
            category="application",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "application": {"type": "string", "description": "Application name or command"},
                    "arguments": {"type": "string", "default": ""},
                    "cwd": {"type": "string", "description": "Working directory"},
                },
                "required": ["application"],
            },
            handler=app_open,
        ),
        Tool(
            name="application.list_running",
            description="List currently running applications with visible windows",
            category="application",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=app_list_running,
        ),
        # Git
        Tool(
            name="git.status",
            description="Get git repository status",
            category="git",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Repository path"},
                },
            },
            handler=git_status,
        ),
        Tool(
            name="git.log",
            description="Get recent git log",
            category="git",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Repository path"},
                    "n": {"type": "integer", "default": 10, "description": "Number of commits"},
                },
            },
            handler=git_log,
        ),
        Tool(
            name="git.diff",
            description="Get git diff",
            category="git",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "cwd": {"type": "string", "description": "Repository path"},
                },
            },
            handler=git_diff,
        ),
        Tool(
            name="git.execute",
            description="Execute an arbitrary git command",
            category="git",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Git arguments"},
                    "cwd": {"type": "string", "description": "Repository path"},
                },
                "required": ["args"],
            },
            handler=git_execute,
        ),
        # Network
        Tool(
            name="network.list_wifi",
            description="List available Wi-Fi networks",
            category="network",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=network_list_wifi,
        ),
        Tool(
            name="network.wifi_status",
            description="Get current Wi-Fi connection status",
            category="network",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=network_wifi_status,
        ),
    ]

    for tool in tools:
        registry.register(tool)

    logger.info("deterministic_tools_registered", count=len(tools))
