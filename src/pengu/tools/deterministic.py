"""
Deterministic tools for Pengu — real OS control.

These tools execute actual operations on the user's machine.
No LLM needed for deterministic operations.

Day 1+2: filesystem, terminal, application, git, network basics
Day 3:   enhanced app management, process management, secure filesystem,
         safe terminal, system info, VS Code integration
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
# Filesystem tools (Day 1 — kept for backward compatibility)
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
        return ToolResult(success=False, error=f"Command timed out after {timeout}s: {command[:100]}")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Application tools (Day 1 base)
# ---------------------------------------------------------------------------

async def app_open(
    application: str, arguments: str = "", cwd: str | None = None
) -> ToolResult:
    """Open an application via the AppManager (registry-only, no arbitrary launch)."""
    from pengu.os.app_manager import get_app_manager
    manager = get_app_manager()
    result = manager.open(application, arguments=arguments, cwd=cwd)
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


async def app_close(application: str) -> ToolResult:
    """Close an application by name using the AppManager."""
    from pengu.os.app_manager import get_app_manager
    manager = get_app_manager()
    result = manager.close(application)
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


async def app_is_running(application: str) -> ToolResult:
    """Check if an application is running."""
    from pengu.os.app_manager import get_app_manager
    manager = get_app_manager()
    result = manager.is_running(application)
    return ToolResult(
        success=result["success"],
        output=result,
        error=result.get("error", ""),
    )


async def app_list_installed() -> ToolResult:
    """List all discovered installed applications."""
    from pengu.os.app_manager import get_app_manager
    manager = get_app_manager()
    apps = manager.list_installed()
    return ToolResult(
        success=True,
        output={"applications": apps, "count": len(apps)},
    )


async def app_list_running() -> ToolResult:
    """List running applications with visible windows."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Process | Where-Object {$_.MainWindowTitle} | "
                 "Select-Object Name, MainWindowTitle, Id | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            import json
            try:
                processes = json.loads(result.stdout)
                if isinstance(processes, dict):
                    processes = [processes]
                apps = [
                    {"name": p.get("Name", ""), "title": p.get("MainWindowTitle", ""), "pid": p.get("Id", 0)}
                    for p in processes
                    if p.get("MainWindowTitle")
                ]
                return ToolResult(
                    success=True,
                    output={"applications": apps, "count": len(apps)},
                )
            except json.JSONDecodeError:
                return ToolResult(success=True, output={"raw": result.stdout[:2000]})
        else:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10)
            return ToolResult(success=True, output={"raw": result.stdout[:3000]})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Process management tools (Day 3)
# ---------------------------------------------------------------------------


async def process_list(
    name_filter: str = "", max_results: int = 20
) -> ToolResult:
    """List running processes with optional filtering."""
    from pengu.os.process_manager import ProcessManager
    pm = ProcessManager()
    processes = pm.list_processes(name_filter=name_filter, max_results=max_results)
    return ToolResult(
        success=True,
        output={
            "processes": [p.to_dict() for p in processes],
            "count": len(processes),
        },
    )


async def process_info(pid: int) -> ToolResult:
    """Get detailed info for a specific process."""
    from pengu.os.process_manager import ProcessManager
    pm = ProcessManager()
    proc = pm.get_process(pid)
    if proc is None:
        return ToolResult(success=False, error=f"Process with PID {pid} not found")
    return ToolResult(success=True, output=proc.to_dict())


async def process_terminate(pid: int, force: bool = False) -> ToolResult:
    """Safely terminate a process by PID."""
    from pengu.os.process_manager import ProcessManager
    pm = ProcessManager()
    result = pm.terminate(pid, force=force)
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------


async def git_execute(args: list[str], cwd: str | None = None) -> ToolResult:
    """Execute a git command."""
    try:
        cmd_list = ["git"] + args
        result = subprocess.run(
            cmd_list, capture_output=True, text=True, timeout=15, cwd=cwd,
            encoding="utf-8", errors="replace",
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
    return await git_execute(["log", "--oneline", f"-{n}"], cwd=cwd)


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
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return ToolResult(success=result.returncode == 0, output={"raw": result.stdout[:5000]})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def network_wifi_status() -> ToolResult:
    """Get current Wi-Fi status."""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return ToolResult(success=result.returncode == 0, output={"raw": result.stdout[:3000]})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# System info tools (Day 3)
# ---------------------------------------------------------------------------


async def system_info() -> ToolResult:
    """Get human-readable system info summary."""
    from pengu.os.system_info import get_system_info_summary
    summary = get_system_info_summary()
    return ToolResult(success=True, output={"summary": summary})


async def system_battery() -> ToolResult:
    """Get battery status: percentage, charging, time remaining."""
    try:
        import psutil
        bat = psutil.sensors_battery()
        if bat is None:
            return ToolResult(
                success=True,
                output={"battery": False, "message": "No battery detected. This appears to be a desktop or external power."},
            )
        percent = bat.percent
        plugged = bat.power_plugged
        secs = bat.secsleft
        if secs == -1:
            time_left = "unknown"
        elif secs < 0:
            time_left = "unlimited (plugged in)"
        else:
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            time_left = f"{hours}h {mins}m"
        charging = "Charging" if plugged else "On battery"
        summary = f"Battery: {percent}% - {charging} - Time remaining: {time_left}"
        return ToolResult(
            success=True,
            output={
                "battery": True,
                "percent": percent,
                "plugged": plugged,
                "charging": plugged,
                "secs_left": secs,
                "time_left": time_left,
                "summary": summary,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=f"Battery check failed: {e}")


async def system_wallpaper(path: str = "") -> ToolResult:
    """Set or get the desktop wallpaper.

    If path is empty, returns the current wallpaper.
    If path is provided, sets it as the wallpaper.
    Accepts .jpg, .png, .bmp files.
    """
    import ctypes
    from pathlib import Path

    try:
        if not path:
            # Get current wallpaper
            SPI_GETDESKWALLPAPER = 0x0073
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.SystemParametersInfoW(
                SPI_GETDESKWALLPAPER, 512, buf, 0
            )
            current = buf.value or "(no wallpaper set)"
            return ToolResult(
                success=True,
                output={"current_wallpaper": current, "summary": f"Current wallpaper: {current}"},
            )

        # Set wallpaper
        wallpaper_path = Path(path).resolve()
        if not wallpaper_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        if wallpaper_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            return ToolResult(
                success=False,
                error=f"Unsupported image format: {wallpaper_path.suffix}. Use .jpg, .png, or .bmp.",
            )

        SPI_SETDESKWALLPAPER = 0x0014
        SPIF_UPDATEINIFILE = 0x01
        SPIF_SENDCHANGE = 0x02
        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER, 0, str(wallpaper_path),
            SPIF_UPDATEINIFILE | SPIF_SENDCHANGE,
        )
        if result:
            return ToolResult(
                success=True,
                output={"wallpaper": str(wallpaper_path), "summary": f"Wallpaper changed to {wallpaper_path.name}"},
            )
        return ToolResult(success=False, error="SystemParametersInfo returned failure")
    except Exception as e:
        return ToolResult(success=False, error=f"Wallpaper operation failed: {e}")


async def system_volume(action: str = "get", level: int = 0) -> ToolResult:
    """Control system volume.

    action: 'get', 'set', 'mute', 'unmute'
    level: 0-100 (only for 'set')
    """
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL, CoInitialize
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        try:
            CoInitialize()
        except Exception:
            pass

        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        if action == "get":
            mute = volume.GetMute()
            vol = round(volume.GetMasterVolumeLevelScalar() * 100)
            state = "Muted" if mute else f"Volume: {vol}%"
            return ToolResult(
                success=True,
                output={"volume": vol, "muted": bool(mute), "summary": state},
            )

        elif action == "set":
            level = max(0, min(100, level))
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            return ToolResult(
                success=True,
                output={"volume": level, "summary": f"Volume set to {level}%"},
            )

        elif action == "mute":
            volume.SetMute(1, None)
            return ToolResult(success=True, output={"muted": True, "summary": "Muted"})

        elif action == "unmute":
            volume.SetMute(0, None)
            return ToolResult(success=True, output={"muted": False, "summary": "Unmuted"})

        return ToolResult(success=False, error=f"Unknown volume action: {action}")
    except ImportError:
        return ToolResult(success=False, error="pycaw not installed. Run: pip install pycaw")
    except Exception as e:
        return ToolResult(success=False, error=f"Volume control failed: {e}")


# ---------------------------------------------------------------------------
# VS Code tools (Day 3)
# ---------------------------------------------------------------------------


async def vscode_open_folder(folder_path: str) -> ToolResult:
    """Open a folder in VS Code."""
    from pengu.os.vscode import open_folder
    result = open_folder(folder_path)
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


async def vscode_open_file(file_path: str, line: int = 0) -> ToolResult:
    """Open a file in VS Code, optionally at a specific line."""
    from pengu.os.vscode import open_file, open_file_at_line
    if line > 0:
        result = open_file_at_line(file_path, line)
    else:
        result = open_file(file_path)
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


async def vscode_focus() -> ToolResult:
    """Bring VS Code to the foreground."""
    from pengu.os.vscode import focus_vscode
    result = focus_vscode()
    return ToolResult(
        success=result["success"],
        output=result if result["success"] else None,
        error=result.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_deterministic_tools(registry: ToolRegistry) -> None:
    """Register all deterministic tools in the registry."""
    tools = [
        # --- Filesystem ---
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

        # --- Terminal ---
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

        # --- Application ---
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
            name="application.close",
            description="Close an application by name",
            category="application",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "application": {"type": "string", "description": "Application name"},
                },
                "required": ["application"],
            },
            handler=app_close,
        ),
        Tool(
            name="application.is_running",
            description="Check if an application is currently running",
            category="application",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "application": {"type": "string", "description": "Application name"},
                },
                "required": ["application"],
            },
            handler=app_is_running,
        ),
        Tool(
            name="application.list_installed",
            description="List all discovered installed applications",
            category="application",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=app_list_installed,
        ),
        Tool(
            name="application.list_running",
            description="List currently running applications with visible windows",
            category="application",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=app_list_running,
        ),

        # --- Process management ---
        Tool(
            name="process.list",
            description="List running processes with optional filtering",
            category="process",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "name_filter": {"type": "string", "default": "", "description": "Filter by process name"},
                    "max_results": {"type": "integer", "default": 20},
                },
            },
            handler=process_list,
        ),
        Tool(
            name="process.info",
            description="Get detailed info for a specific process",
            category="process",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID"},
                },
                "required": ["pid"],
            },
            handler=process_info,
        ),
        Tool(
            name="process.terminate",
            description="Safely terminate a process by PID",
            category="process",
            permission_level=PermissionLevel.HIGH_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "Process ID"},
                    "force": {"type": "boolean", "default": False},
                },
                "required": ["pid"],
            },
            handler=process_terminate,
        ),

        # --- Git ---
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

        # --- Network ---
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

        # --- System info (Day 3) ---
        Tool(
            name="system.info",
            description="Get human-readable system info summary",
            category="system",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=system_info,
        ),
        Tool(
            name="system.battery",
            description="Get battery status: percentage, charging state, time remaining",
            category="system",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=system_battery,
        ),
        Tool(
            name="system.wallpaper",
            description="Get or set the desktop wallpaper",
            category="system",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "", "description": "Image file path to set as wallpaper (empty = get current)"},
                },
            },
            handler=system_wallpaper,
        ),
        Tool(
            name="system.volume",
            description="Get or set system volume (get/set/mute/unmute)",
            category="system",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
            },
            handler=system_volume,
        ),

        # --- VS Code (Day 3) ---
        Tool(
            name="vscode.open_folder",
            description="Open a folder/project in VS Code",
            category="vscode",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "folder_path": {"type": "string", "description": "Folder path or project name"},
                },
                "required": ["folder_path"],
            },
            handler=vscode_open_folder,
        ),
        Tool(
            name="vscode.open_file",
            description="Open a file in VS Code, optionally at a specific line",
            category="vscode",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File path"},
                    "line": {"type": "integer", "default": 0, "description": "Line number (0 for top)"},
                },
                "required": ["file_path"],
            },
            handler=vscode_open_file,
        ),
        Tool(
            name="vscode.focus",
            description="Bring VS Code to the foreground",
            category="vscode",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=vscode_focus,
        ),
    ]

    for tool in tools:
        registry.register(tool)

    logger.info("deterministic_tools_registered", count=len(tools))
