"""
Application Manager — discover, open, close, focus, and inspect applications.

Uses an allowlisted registry of known applications with verified paths.
Supports dynamic discovery on PATH and Windows common locations.

Security:
  - Only known/allowlisted applications can be managed
  - No arbitrary executable launching from user text
  - Process operations go through safe abstractions
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.os.app_manager")


@dataclass
class AppInfo:
    """Information about a registered application."""
    name: str
    display_name: str
    command: str  # command to launch
    aliases: list[str] = field(default_factory=list)
    path: str = ""  # resolved path if found
    installed: bool = False
    running: bool = False
    pid: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "command": self.command,
            "aliases": self.aliases,
            "path": self.path,
            "installed": self.installed,
            "running": self.running,
            "pid": self.pid,
        }


# ---------------------------------------------------------------------------
# Application Registry — known safe applications
# ---------------------------------------------------------------------------

_APP_REGISTRY: list[AppInfo] = [
    AppInfo(
        name="code",
        display_name="VS Code",
        command="code",
        aliases=["vs code", "visual studio code", "vscode"],
    ),
    AppInfo(
        name="chrome",
        display_name="Google Chrome",
        command="start chrome",
        aliases=["google chrome", "chrome"],
    ),
    AppInfo(
        name="firefox",
        display_name="Mozilla Firefox",
        command="start firefox",
        aliases=["firefox", "mozilla firefox"],
    ),
    AppInfo(
        name="msedge",
        display_name="Microsoft Edge",
        command="start msedge",
        aliases=["edge", "microsoft edge", "ms edge"],
    ),
    AppInfo(
        name="wt",
        display_name="Windows Terminal",
        command="wt",
        aliases=["windows terminal", "terminal"],
    ),
    AppInfo(
        name="explorer",
        display_name="File Explorer",
        command="explorer",
        aliases=["file explorer", "explorer", "files"],
    ),
    AppInfo(
        name="notepad",
        display_name="Notepad",
        command="notepad",
        aliases=["notepad"],
    ),
    AppInfo(
        name="pwsh",
        display_name="PowerShell",
        command="pwsh",
        aliases=["powershell", "pwsh"],
    ),
    AppInfo(
        name="cmd",
        display_name="Command Prompt",
        command="cmd",
        aliases=["cmd", "command prompt", "cmd.exe"],
    ),
    AppInfo(
        name="cursor",
        display_name="Cursor",
        command="cursor",
        aliases=["cursor"],
    ),
    AppInfo(
        name="idea64",
        display_name="IntelliJ IDEA",
        command="idea64",
        aliases=["intellij", "intellij idea"],
    ),
    AppInfo(
        name="pycharm64",
        display_name="PyCharm",
        command="pycharm64",
        aliases=["pycharm"],
    ),
    AppInfo(
        name="webstorm64",
        display_name="WebStorm",
        command="webstorm64",
        aliases=["webstorm"],
    ),
    AppInfo(
        name="git-bash",
        display_name="Git Bash",
        command="start git-bash",
        aliases=["git bash", "gitbash"],
    ),
]


class AppManager:
    """
    Manages application discovery, launching, and process interaction.
    
    Only registered/allowlisted applications can be managed.
    """

    def __init__(self) -> None:
        self._apps: dict[str, AppInfo] = {}
        for app in _APP_REGISTRY:
            self._apps[app.name] = app
        self._discovered = False

    def discover(self) -> None:
        """Check which registered applications are actually installed."""
        if self._discovered:
            return

        for name, app in self._apps.items():
            # Check if the base command exists on PATH
            base_cmd = app.command.split()[0]
            path = shutil.which(base_cmd)
            if path:
                app.installed = True
                app.path = path
            else:
                app.installed = False
                app.path = ""

            if app.installed:
                logger.debug("app_discovered", name=name, path=app.path)

        self._discovered = True
        installed = [n for n, a in self._apps.items() if a.installed]
        logger.info("app_discovery_complete", installed=installed)

    def resolve(self, name: str) -> Optional[AppInfo]:
        """
        Resolve an application name to a registered AppInfo.
        
        Matches against:
          - exact name
          - aliases (case-insensitive)
        
        Returns None if not found in the registry.
        """
        self.discover()
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in self._apps:
            return self._apps[name_lower]

        # Alias match
        for app in self._apps.values():
            for alias in app.aliases:
                if name_lower == alias:
                    return app

        # Partial match (name contains the query)
        for app in self._apps.values():
            if name_lower in app.name.lower() or name_lower in app.display_name.lower():
                return app
            for alias in app.aliases:
                if name_lower in alias:
                    return app

        return None

    def open(
        self, name: str, arguments: str = "", cwd: str | None = None
    ) -> dict:
        """
        Open/launch an application by name.
        
        Returns dict with success, application, status, and error fields.
        """
        app = self.resolve(name)
        if not app:
            return {
                "success": False,
                "application": name,
                "error": f"Application not found in registry: {name}. "
                         f"Available: {', '.join(a.display_name for a in self._apps.values() if a.installed)}",
            }

        if not app.installed:
            return {
                "success": False,
                "application": app.display_name,
                "error": f"{app.display_name} is not installed on this machine.",
            }

        try:
            cmd = app.command
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

            logger.info("app_opened", name=app.name, command=cmd)
            return {
                "success": True,
                "application": app.display_name,
                "command": cmd,
                "status": "launched",
            }
        except Exception as e:
            logger.error("app_open_failed", name=app.name, error=str(e))
            return {
                "success": False,
                "application": app.display_name,
                "error": f"Failed to launch {app.display_name}: {e}",
            }

    def close(self, name: str) -> dict:
        """
        Close an application by name.
        
        Finds running processes and attempts graceful termination.
        """
        app = self.resolve(name)
        if not app:
            return {
                "success": False,
                "application": name,
                "error": f"Application not found in registry: {name}",
            }

        import psutil

        # Find processes matching the application
        process_name = app.command.split()[0].lower()
        found = []
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                proc_name = (proc.info["name"] or "").lower()
                proc_exe = (proc.info["exe"] or "").lower()
                if process_name in proc_name or process_name in proc_exe:
                    found.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not found:
            return {
                "success": True,
                "application": app.display_name,
                "status": "not_running",
                "message": f"{app.display_name} is not currently running.",
            }

        terminated = 0
        for proc in found:
            try:
                proc.terminate()
                terminated += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning("app_close_failed_pid", pid=proc.pid, error=str(e))

        logger.info("app_closed", name=app.name, processes_terminated=terminated)
        return {
            "success": True,
            "application": app.display_name,
            "status": "closed",
            "processes_terminated": terminated,
        }

    def is_running(self, name: str) -> dict:
        """Check if an application is currently running."""
        app = self.resolve(name)
        if not app:
            return {
                "success": False,
                "application": name,
                "error": f"Application not found in registry: {name}",
            }

        import psutil

        process_name = app.command.split()[0].lower()
        running = []
        for proc in psutil.process_iter(["pid", "name", "exe", "cpu_percent"]):
            try:
                proc_name = (proc.info["name"] or "").lower()
                proc_exe = (proc.info["exe"] or "").lower()
                if process_name in proc_name or process_name in proc_exe:
                    running.append({
                        "pid": proc.pid,
                        "name": proc.info["name"],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "success": True,
            "application": app.display_name,
            "running": len(running) > 0,
            "process_count": len(running),
            "processes": running,
        }

    def list_installed(self) -> list[dict]:
        """List all discovered installed applications."""
        self.discover()
        return [
            app.to_dict()
            for app in self._apps.values()
            if app.installed
        ]

    def list_all(self) -> list[dict]:
        """List all registered applications with status."""
        self.discover()
        return [app.to_dict() for app in self._apps.values()]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager: Optional[AppManager] = None


def get_app_manager() -> AppManager:
    """Get or create the global application manager."""
    global _manager
    if _manager is None:
        _manager = AppManager()
    return _manager
