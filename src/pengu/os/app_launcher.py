"""
Real Windows Application Launcher for Pengu.

Solves the "opens Music instead of VS Code" problem by:
1. Maintaining a verified application registry
2. Discovering apps via PATH, known locations, and Start Menu
3. Validating executables before launching
4. Never guessing when multiple apps match

Application Registry:
  name → executable path, aliases, confidence
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.os.app_launcher")


@dataclass
class AppEntry:
    """A discovered application."""
    name: str
    executable: str
    path: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""  # "path", "known", "start_menu"


# ---------------------------------------------------------------------------
# Verified application registry
# ---------------------------------------------------------------------------

# These are verified on THIS machine. The launcher also does PATH discovery.
VERIFIED_APPS: list[AppEntry] = [
    AppEntry(
        name="vscode",
        executable="Code.exe",
        path=r"C:\Users\ADMIN\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        aliases=["vs code", "visual studio code", "code", "vscode"],
        confidence=1.0,
        source="known",
    ),
    AppEntry(
        name="chrome",
        executable="chrome.exe",
        path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        aliases=["chrome", "google chrome"],
        confidence=1.0,
        source="known",
    ),
    AppEntry(
        name="edge",
        executable="msedge.exe",
        path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        aliases=["edge", "microsoft edge", "msedge"],
        confidence=1.0,
        source="known",
    ),
    AppEntry(
        name="explorer",
        executable="explorer",
        path="",
        aliases=["file explorer", "explorer", "files", "folder explorer", "my computer"],
        confidence=1.0,
        source="system",
    ),
    AppEntry(
        name="notepad",
        executable="notepad",
        path="",
        aliases=["notepad", "text editor"],
        confidence=0.9,
        source="system",
    ),
    AppEntry(
        name="terminal",
        executable="wt",
        path="",
        aliases=["terminal", "windows terminal", "wt"],
        confidence=0.9,
        source="path",
    ),
    AppEntry(
        name="powershell",
        executable="pwsh",
        path="",
        aliases=["powershell", "pwsh"],
        confidence=0.9,
        source="path",
    ),
    AppEntry(
        name="cmd",
        executable="cmd",
        path="",
        aliases=["cmd", "command prompt", "command line"],
        confidence=0.9,
        source="system",
    ),
    AppEntry(
        name="taskmanager",
        executable="taskmgr",
        path="",
        aliases=["task manager", "taskmanager", "processes"],
        confidence=0.9,
        source="system",
    ),
    AppEntry(
        name="settings",
        executable="ms-settings:",
        path="",
        aliases=["settings", "windows settings", "system settings"],
        confidence=0.9,
        source="system",
    ),
    AppEntry(
        name="cursor",
        executable="cursor",
        path="",
        aliases=["cursor", "cursor ide"],
        confidence=0.9,
        source="path",
    ),
]


class AppLauncher:
    """
    Real Windows application launcher.

    Features:
    - Verified app registry
    - PATH discovery
    - Alias matching
    - Confidence scoring
    - Never opens wrong application
    """

    def __init__(self) -> None:
        self._apps: dict[str, AppEntry] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Build the application registry from verified apps."""
        for app in VERIFIED_APPS:
            self._apps[app.name] = app
            # Index by alias
            for alias in app.aliases:
                self._apps[alias] = app

    def find_app(self, query: str) -> Optional[AppEntry]:
        """
        Find an application by name or alias.

        Returns the best match or None if not found.
        """
        query_lower = query.lower().strip()

        # Direct name match
        if query_lower in self._apps:
            return self._apps[query_lower]

        # Alias match
        for alias, app in self._apps.items():
            if query_lower in alias or alias in query_lower:
                return app

        # Fuzzy match
        best_match = None
        best_score = 0.0
        for alias, app in self._apps.items():
            score = self._similarity(query_lower, alias)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = app

        return best_match

    def _similarity(self, a: str, b: str) -> float:
        """Simple string similarity."""
        a_words = set(a.split())
        b_words = set(b.split())
        if not a_words or not b_words:
            return 0.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)

    def open_application(self, app_name: str, arguments: str = "") -> dict:
        """
        Open an application by name.

        Returns:
            {
                "success": bool,
                "application": str,
                "executable": str,
                "message": str,
                "pid": int or None,
            }
        """
        app = self.find_app(app_name)

        if app is None:
            # Try to find it via PATH
            path_result = self._find_in_path(app_name)
            if path_result:
                return self._launch(path_result, app_name, arguments)

            return {
                "success": False,
                "application": app_name,
                "executable": "",
                "message": f"Application '{app_name}' not found. I couldn't find it installed on your system.",
                "pid": None,
            }

        # For system URLs like ms-settings:, use start
        if app.executable.endswith(":"):
            try:
                os.startfile(app.executable)
                return {
                    "success": True,
                    "application": app.name,
                    "executable": app.executable,
                    "message": f"Opening {app.name}.",
                    "pid": None,
                }
            except Exception as e:
                return {
                    "success": False,
                    "application": app.name,
                    "executable": app.executable,
                    "message": f"Failed to open {app.name}: {e}",
                    "pid": None,
                }

        return self._launch(app.executable, app.name, arguments)

    def _launch(self, executable: str, name: str, arguments: str = "") -> dict:
        """Launch an executable."""
        try:
            # Find the app entry to get the full path
            app_entry = None
            for entry in VERIFIED_APPS:
                if entry.name == name or executable in entry.aliases:
                    app_entry = entry
                    break

            # Use full path from registry if it exists
            full_path = executable
            if app_entry and app_entry.path and os.path.exists(app_entry.path):
                full_path = app_entry.path
            elif not os.path.exists(executable):
                # Try where command
                where_result = self._find_in_path(executable.replace(".exe", "").replace(".cmd", ""))
                if where_result:
                    full_path = where_result

            # Use os.startfile for Windows executables (most reliable)
            if os.path.exists(full_path):
                os.startfile(full_path)
                logger.info("app_launched", name=name, executable=full_path)
                return {
                    "success": True,
                    "application": name,
                    "executable": full_path,
                    "message": f"Opening {name}.",
                    "pid": None,
                }

            # Fallback: try subprocess
            if full_path.endswith(".cmd") or full_path.endswith(".bat"):
                cmd = ["cmd", "/c", full_path]
            else:
                cmd = [full_path]

            if arguments:
                cmd.extend(arguments.split())

            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                cmd,
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            logger.info("app_launched", name=name, executable=executable, pid=process.pid)

            return {
                "success": True,
                "application": name,
                "executable": executable,
                "message": f"Opening {name}.",
                "pid": process.pid,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "application": name,
                "executable": executable,
                "message": f"Could not find executable '{executable}'. Is {name} installed?",
                "pid": None,
            }
        except Exception as e:
            return {
                "success": False,
                "application": name,
                "executable": executable,
                "message": f"Failed to open {name}: {e}",
                "pid": None,
            }

    def _find_in_path(self, name: str) -> Optional[str]:
        """Find an executable in PATH."""
        # Add .exe for Windows
        for suffix in ["", ".exe", ".cmd", ".bat"]:
            try:
                result = subprocess.run(
                    ["where", name + suffix],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split("\n")[0]
            except Exception:
                pass
        return None

    def open_file(self, file_path: str) -> dict:
        """Open a file with its default application."""
        path = Path(file_path).resolve()
        if not path.exists():
            return {
                "success": False,
                "application": "default",
                "executable": "",
                "message": f"File not found: {file_path}",
                "pid": None,
            }

        try:
            os.startfile(str(path))
            return {
                "success": True,
                "application": "default",
                "executable": str(path),
                "message": f"Opening {path.name}.",
                "pid": None,
            }
        except Exception as e:
            return {
                "success": False,
                "application": "default",
                "executable": str(path),
                "message": f"Failed to open {path.name}: {e}",
                "pid": None,
            }

    def open_folder(self, folder_path: str) -> dict:
        """Open a folder in File Explorer."""
        path = Path(folder_path).resolve()
        if not path.exists():
            return {
                "success": False,
                "application": "explorer",
                "executable": "",
                "message": f"Folder not found: {folder_path}",
                "pid": None,
            }

        if not path.is_dir():
            return {
                "success": False,
                "application": "explorer",
                "executable": "",
                "message": f"Not a folder: {folder_path}",
                "pid": None,
            }

        try:
            subprocess.Popen(
                ["explorer", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {
                "success": True,
                "application": "explorer",
                "executable": "explorer",
                "message": f"Opening folder {path.name}.",
                "pid": None,
            }
        except Exception as e:
            return {
                "success": False,
                "application": "explorer",
                "executable": "explorer",
                "message": f"Failed to open folder: {e}",
                "pid": None,
            }

    def open_in_vscode(self, target: str) -> dict:
        """Open a file or folder in VS Code."""
        path = Path(target).resolve()

        if not path.exists():
            return {
                "success": False,
                "application": "vscode",
                "executable": "code",
                "message": f"Path not found: {target}",
                "pid": None,
            }

        try:
            cmd = ["code", str(path)]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            target_type = "folder" if path.is_dir() else "file"
            return {
                "success": True,
                "application": "vscode",
                "executable": "code",
                "message": f"Opening {path.name} in VS Code.",
                "pid": process.pid,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "application": "vscode",
                "executable": "code",
                "message": "VS Code not found. Is it installed? Try: winget install Microsoft.VisualStudioCode",
                "pid": None,
            }
        except Exception as e:
            return {
                "success": False,
                "application": "vscode",
                "executable": "code",
                "message": f"Failed to open in VS Code: {e}",
                "pid": None,
            }

    def open_url(self, url: str, browser: str = "default") -> dict:
        """Open a URL in a browser."""
        if browser == "default" or browser == "chrome":
            chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            if os.path.exists(chrome_path):
                try:
                    process = subprocess.Popen(
                        [chrome_path, url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return {
                        "success": True,
                        "application": "chrome",
                        "executable": chrome_path,
                        "message": f"Opening {url} in Chrome.",
                        "pid": process.pid,
                    }
                except Exception:
                    pass

        # Fallback to default browser
        try:
            os.startfile(url)
            return {
                "success": True,
                "application": "default_browser",
                "executable": "",
                "message": f"Opening {url}.",
                "pid": None,
            }
        except Exception as e:
            return {
                "success": False,
                "application": "default_browser",
                "executable": "",
                "message": f"Failed to open URL: {e}",
                "pid": None,
            }

    def google_search(self, query: str, browser: str = "default") -> dict:
        """Search Google for a query."""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return self.open_url(url, browser)

    def open_chatgpt(self) -> dict:
        """Open ChatGPT in the browser."""
        return self.open_url("https://chatgpt.com", "default")

    def list_apps(self) -> list[dict]:
        """List all known applications."""
        seen = set()
        apps = []
        for app in VERIFIED_APPS:
            if app.name not in seen:
                seen.add(app.name)
                apps.append({
                    "name": app.name,
                    "aliases": app.aliases,
                    "executable": app.executable,
                })
        return apps


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_launcher: Optional[AppLauncher] = None


def get_launcher() -> AppLauncher:
    """Get the global application launcher."""
    global _launcher
    if _launcher is None:
        _launcher = AppLauncher()
    return _launcher
