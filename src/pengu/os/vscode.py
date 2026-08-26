"""
VS Code Integration — open folders, files, and manage VS Code.

Uses the VS Code CLI (`code`) for structured operations.
Falls back to filesystem operations when CLI is unavailable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.os.vscode")


def _is_vscode_available() -> bool:
    """Check if VS Code CLI is available."""
    import shutil
    return shutil.which("code") is not None


def _run_vscode_cmd(args: list[str], timeout: int = 10) -> dict:
    """Run a VS Code CLI command."""
    if not _is_vscode_available():
        return {
            "success": False,
            "error": "VS Code CLI not found. Install VS Code and ensure 'code' is on PATH.",
        }

    cmd = ["code"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000] if result.stdout else "",
            "stderr": result.stderr[:1000] if result.stderr else "",
            "exit_code": result.returncode,
        }
    except FileNotFoundError:
        return {"success": False, "error": "VS Code CLI ('code') not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "VS Code command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def open_folder(folder_path: str) -> dict:
    """
    Open a folder in VS Code.
    
    Resolves the path through common project locations.
    """
    if not _is_vscode_available():
        return {"success": False, "error": "VS Code not available"}

    # Resolve the folder path
    resolved = _resolve_path(folder_path)
    if resolved is None:
        return {
            "success": False,
            "error": f"Folder not found: {folder_path}. Searched: current directory, home, ~/projects",
        }

    result = _run_vscode_cmd([str(resolved)])
    if result["success"]:
        logger.info("vscode_folder_opened", path=str(resolved))
        return {
            "success": True,
            "action": "open_folder",
            "path": str(resolved),
            "name": resolved.name,
        }
    else:
        return {"success": False, "error": result["error"]}


def open_file(file_path: str) -> dict:
    """Open a file in VS Code."""
    if not _is_vscode_available():
        return {"success": False, "error": "VS Code not available"}

    p = Path(file_path).resolve()
    if not p.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    result = _run_vscode_cmd(["-g", str(p)])
    if result["success"]:
        logger.info("vscode_file_opened", path=str(p))
        return {
            "success": True,
            "action": "open_file",
            "path": str(p),
            "line": 1,
        }
    return {"success": False, "error": result["error"]}


def open_file_at_line(file_path: str, line: int) -> dict:
    """Open a file at a specific line in VS Code."""
    if not _is_vscode_available():
        return {"success": False, "error": "VS Code not available"}

    p = Path(file_path).resolve()
    if not p.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    result = _run_vscode_cmd(["-g", f"{p}:{line}"])
    if result["success"]:
        return {
            "success": True,
            "action": "open_file",
            "path": str(p),
            "line": line,
        }
    return {"success": False, "error": result["error"]}


def open_new_window() -> dict:
    """Open a new VS Code window."""
    if not _is_vscode_available():
        return {"success": False, "error": "VS Code not available"}

    result = _run_vscode_cmd(["--new-window"])
    return {"success": result["success"], "action": "new_window"}


def focus_vscode() -> dict:
    """Bring VS Code to the foreground."""
    import psutil

    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info["name"] or "").lower()
            exe = (proc.info["exe"] or "").lower()
            if "code" in name and ("electron" in exe or "code" in exe):
                # Use PowerShell to bring window to front
                ps_cmd = (
                    f"(Get-Process -Id {proc.pid}).MainWindowHandle | "
                    f"ForEach-Object {{ Add-Type -Name Win -Namespace User32 "
                    f'-MemberDefinition \'[DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr hWnd);\' ; '
                    f'[User32.Win]::SetForegroundWindow($_) }}'
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=5,
                )
                logger.info("vscode_focused", pid=proc.pid)
                return {"success": True, "action": "focus", "pid": proc.pid}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {"success": False, "error": "VS Code process not found"}


def _resolve_path(name: str) -> Optional[Path]:
    """
    Resolve a folder/project name to an actual path.
    
    Searches:
      1. Exact path
      2. Current working directory
      3. Home directory
      4. ~/projects/
      5. ~/Documents/
    """
    # Exact path
    p = Path(name).resolve()
    if p.exists() and p.is_dir():
        return p

    # Home directory
    home = Path.home()
    candidates = [
        home / name,
        Path.cwd() / name,
    ]

    # Common project locations
    for parent_name in ["projects", "Projects", "Documents", "repos", "Repos", "code", "Code", "dev", "Dev"]:
        parent = home / parent_name
        if parent.exists():
            candidates.append(parent / name)
            # Also search subdirectories (one level)
            try:
                for d in parent.iterdir():
                    if d.is_dir() and name.lower() in d.name.lower():
                        candidates.append(d)
            except OSError:
                pass

    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()

    return None
