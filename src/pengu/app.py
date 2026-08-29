"""
Pengu Application — production voice-first desktop assistant.

Architecture:
  Voice → VAD + STT wake word → Command Parser → Deterministic Tool / LLM → TTS → Standby

Features:
  - STT-based wake word detection ("hello pengu")
  - Deterministic-first command routing
  - Model auto-discovery (LM Studio, Ollama)
  - TTS barge-in support
  - Full diagnostics
  - Desktop overlay and system tray integration
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

from pengu.config import get_config
from pengu.context import get_context
from pengu.logging import get_logger, setup_logging
from pengu.os.app_launcher import get_launcher
from pengu.tools.deterministic import register_deterministic_tools
from pengu.tools.registry import ToolRegistry
from pengu.voice.engine import VoiceConfig, VoiceEngine, VoiceState
from pengu.ui.overlay import PenguOverlay, OverlayState
from pengu.ui.tray import PenguTray, TrayState
from pengu.hotkey import get_hotkey

logger = get_logger("pengu.app")


def _get_pengu_root() -> Path:
    """Get the Pengu project root directory dynamically."""
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Command Parser — deterministic-first routing
# ---------------------------------------------------------------------------

class CommandParser:
    """
    Deterministic command parser — handles common desktop commands
    without needing the LLM.
    """

    def __init__(self, pengu_root: Path) -> None:
        self._pengu_root = pengu_root
        self._launcher = get_launcher()

    def parse(self, text: str) -> Optional[dict]:
        """
        Parse a voice command into a structured action.
        Returns None if the command should be sent to the LLM.
        """
        text_lower = text.lower().strip()

        # Strip common prefixes
        for prefix in ["pengu,", "pengu ", "hey pengu,", "hey pengu ", "okay pengu,", "okay pengu "]:
            if text_lower.startswith(prefix):
                text_lower = text_lower[len(prefix):].strip()
                break

        # ---- STOP / CANCEL ----
        if text_lower in ["stop", "cancel", "never mind", "nevermind", "forget it", "shut up", "quiet"]:
            return {"action": "stop", "speak": "OK, stopped."}

        # ---- HELP ----
        if any(w in text_lower for w in ["what can you do", "help me", "your commands", "what are your commands"]):
            return {
                "action": "help",
                "speak": (
                    "I can open applications like VS Code, Chrome, and File Explorer. "
                    "I can search Google, open ChatGPT, work with files and folders, "
                    "run Git commands, show system information, and answer questions using a local AI model. "
                    "Just say hello Pengu, then tell me what you need."
                ),
            }

        # ---- DIAGNOSTICS ----
        if any(w in text_lower for w in ["diagnostics", "diagnostic", "run diagnostics", "test yourself", "doctor", "check yourself"]):
            return {"action": "diagnostics", "speak": self._run_diagnostics()}

        # ---- OPEN PENGU ----
        if "open pengu" in text_lower or "open pengo" in text_lower:
            if any(w in text_lower for w in ["vs code", "visual studio code", "in code"]):
                result = self._launcher.open_in_vscode(str(self._pengu_root))
                get_context().update_app("vscode")
                return {"action": "open_in_vscode", "speak": result["message"]}
            elif any(w in text_lower for w in ["in explorer", "folder", "in file"]):
                result = self._launcher.open_folder(str(self._pengu_root))
                get_context().update_directory(str(self._pengu_root))
                return {"action": "open_folder", "speak": result["message"]}
            else:
                result = self._launcher.open_folder(str(self._pengu_root))
                get_context().update_directory(str(self._pengu_root))
                return {"action": "open_folder", "speak": result["message"]}

        # ---- OPEN [APP] IN VS CODE ----
        m = re.match(
            r'^(?:open|launch)\s+(.+?)\s+in\s+(?:vs\s*code|visual\s+studio\s+code|code)\s*\.?$',
            text_lower,
        )
        if m:
            target = m.group(1).strip()
            folder = self._resolve_project(target)
            if folder:
                result = self._launcher.open_in_vscode(str(folder))
                return {"action": "open_in_vscode", "speak": result["message"]}
            else:
                return {"action": "error", "speak": f"I couldn't find the {target} folder."}

        # ---- SEARCH CHATGPT ----
        m = re.match(
            r'^(?:search|ask)\s+chatgpt\s+(?:for\s+|about\s+)?(.+)',
            text_lower,
        )
        if m:
            query = m.group(1).strip().rstrip(".")
            if query:
                url = f"https://chatgpt.com/?q={query.replace(' ', '+')}"
                result = self._launcher.open_url(url)
                get_context().update_url(url, f"ChatGPT: {query}")
                return {"action": "open_chatgpt_search", "speak": f"Searching ChatGPT for {query}."}

        # ---- OPEN CHATGPT ----
        if "chatgpt" in text_lower or "chat gpt" in text_lower:
            if any(w in text_lower for w in ["search", "ask", "query"]):
                # Already handled above
                pass
            else:
                result = self._launcher.open_url("https://chatgpt.com")
                get_context().update_url("https://chatgpt.com", "ChatGPT")
                return {"action": "open_chatgpt", "speak": "Opening ChatGPT."}

        # ---- SEARCH GOOGLE ----
        m = re.match(
            r'^(?:search|google|look\s+up|find)\s+(?:google\s+)?(?:for\s+)?(.+)',
            text_lower,
        )
        if m:
            query = m.group(1).strip().rstrip(".")
            if query:
                result = self._launcher.google_search(query)
                get_context().update_url(f"https://www.google.com/search?q={query.replace(' ', '+')}", f"Google: {query}")
                return {"action": "google_search", "speak": f"Searching Google for {query}."}

        # ---- OPEN CHROME ----
        if "open chrome" in text_lower or "launch chrome" in text_lower or "start chrome" in text_lower:
            result = self._launcher.open_application("chrome")
            get_context().update_app("chrome")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN EDGE ----
        if "open edge" in text_lower or "launch edge" in text_lower:
            result = self._launcher.open_application("edge")
            get_context().update_app("edge")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN KNOWN FOLDERS (Downloads, Documents, etc.) ----
        m = re.match(
            r'^(?:open|go\s+to|show|navigate\s+to)\s+(.+?)(?:\s+folder)?\s*\.?$',
            text_lower,
        )
        if m:
            folder_name = m.group(1).strip()
            resolved = self._resolve_shell_folder(folder_name)
            if resolved:
                result = self._launcher.open_folder(str(resolved))
                ctx = get_context()
                ctx.update_directory(str(resolved))
                ctx.record_action(f"open_folder:{folder_name}", str(resolved))
                return {"action": "open_folder", "speak": result["message"]}

        # ---- OPEN EXPLORER / FILE EXPLORER ----
        if any(w in text_lower for w in ["open explorer", "open file explorer", "open files", "open my computer", "open file manager"]):
            result = self._launcher.open_application("explorer")
            ctx = get_context()
            ctx.update_app("File Explorer")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN TERMINAL / COMMAND PROMPT ----
        if any(w in text_lower for w in ["open terminal", "open command prompt", "open cmd", "open powershell"]):
            if "powershell" in text_lower:
                result = self._launcher.open_application("powershell")
                get_context().update_app("powershell")
            elif "cmd" in text_lower or "command prompt" in text_lower:
                result = self._launcher.open_application("cmd")
                get_context().update_app("cmd")
            else:
                result = self._launcher.open_application("terminal")
                get_context().update_app("terminal")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN NOTEPAD ----
        if "open notepad" in text_lower or "launch notepad" in text_lower:
            result = self._launcher.open_application("notepad")
            get_context().update_app("notepad")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN VS CODE ----
        if any(w in text_lower for w in ["open vs code", "open visual studio code", "launch vs code", "launch visual studio code", "open code"]):
            result = self._launcher.open_application("vscode")
            get_context().update_app("vscode")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN TASK MANAGER ----
        if "task manager" in text_lower:
            result = self._launcher.open_application("taskmanager")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN SETTINGS ----
        if "open settings" in text_lower or "system settings" in text_lower:
            result = self._launcher.open_application("settings")
            return {"action": "open_app", "speak": result["message"]}

        # ---- OPEN [KNOWN URLS] ----
        known_urls = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "github": "https://github.com",
            "gmail": "https://mail.google.com",
        }
        for name, url in known_urls.items():
            if f"open {name}" in text_lower or f"launch {name}" in text_lower:
                result = self._launcher.open_url(url)
                get_context().update_url(url, name)
                return {"action": "open_url", "speak": f"Opening {name}."}

        # ---- GENERIC OPEN [ANYTHING] ----
        m = re.match(
            r'^(?:open|launch|start|run)\s+(.+?)(?:\s+in\s+(.+))?\s*\.?$',
            text_lower,
        )
        if m:
            target = m.group(1).strip()
            context = m.group(2).strip() if m.group(2) else None

            # "open X in VS Code"
            if context and any(w in context for w in ["vs code", "visual studio code", "code"]):
                folder = self._resolve_project(target)
                if folder:
                    result = self._launcher.open_in_vscode(str(folder))
                    get_context().update_app("vscode")
                    return {"action": "open_in_vscode", "speak": result["message"]}

            # Check if it's a known folder first (including shell folders)
            resolved_folder = self._resolve_shell_folder(target)
            if resolved_folder:
                result = self._launcher.open_folder(str(resolved_folder))
                get_context().update_directory(str(resolved_folder))
                return {"action": "open_folder", "speak": result["message"]}

            # Check if it's a known project/folder
            folder = self._resolve_project(target)
            if folder and folder.exists():
                if folder.is_dir():
                    result = self._launcher.open_folder(str(folder))
                    get_context().update_directory(str(folder))
                    return {"action": "open_folder", "speak": result["message"]}
                else:
                    result = self._launcher.open_file(str(folder))
                    get_context().update_file(str(folder))
                    return {"action": "open_file", "speak": result["message"]}

            # Try as application
            app = self._launcher.find_app(target)
            if app:
                result = self._launcher.open_application(target)
                get_context().update_app(target)
                return {"action": "open_app", "speak": result["message"]}

            return {"action": "error", "speak": f"I couldn't find {target} on your system."}

        # ---- CLOSE [APP] ----
        m = re.match(r'^(?:close|quit|exit)\s+(.+?)\s*\.?$', text_lower)
        if m:
            target = m.group(1).strip()
            return {"action": "close_app", "speak": f"Closing {target} is not yet supported. You can close it manually."}

        # ---- FILE OPERATIONS ----
        m = re.search(
            r'(?:create|make|new)\s+(?:a\s+)?(?:file\s+(?:called\s+|named\s+)?)?([^\s]+\.(\w+))',
            text_lower,
        )
        if m:
            filename = m.group(1).strip()
            file_path = self._pengu_root / filename
            try:
                file_path.touch()
                return {"action": "file_created", "speak": f"Created {filename} in the Pengu folder."}
            except Exception as e:
                return {"action": "error", "speak": f"Failed to create {filename}: {e}"}

        m = re.search(
            r'(?:create|make|new)\s+(?:a\s+)?folder\s+(?:called\s+|named\s+)?([^\s]+)',
            text_lower,
        )
        if m:
            foldername = m.group(1).strip()
            folder_path = self._pengu_root / foldername
            try:
                folder_path.mkdir(exist_ok=True)
                return {"action": "folder_created", "speak": f"Created folder {foldername}."}
            except Exception as e:
                return {"action": "error", "speak": f"Failed to create folder {foldername}: {e}"}

        # ---- GIT COMMANDS ----
        if "git status" in text_lower:
            output = self._run_git("status")
            return {"action": "git_result", "speak": f"Git status: {output[:200]}"}

        if "git log" in text_lower:
            output = self._run_git("log --oneline -5")
            return {"action": "git_result", "speak": f"Recent commits: {output[:200]}"}

        if "git diff" in text_lower:
            output = self._run_git("diff --stat")
            return {"action": "git_result", "speak": f"Changes: {output[:200]}"}

        # ---- SYSTEM INFO ----
        if any(w in text_lower for w in [
            "system info", "system information", "what cpu", "what ram",
            "what processor", "computer info", "what computer", "show system",
            "check system", "what are my specs", "what are my system specs",
        ]):
            return {"action": "system_info", "speak": self._get_system_info()}

        # ---- LIST FILES ----
        if any(w in text_lower for w in ["list files", "show files", "what's in", "what is in", "show me files"]):
            return {"action": "list_files", "speak": self._list_files()}

        # ---- VOLUME ----
        m = re.match(r'(?:set|turn|change)\s+volume\s+(?:to\s+)?(\d+)', text_lower)
        if m:
            return {"action": "volume", "speak": "Volume control is not yet implemented."}

        # None = send to LLM
        return None

    # Windows shell folder names to real paths
    _SHELL_FOLDERS: dict[str, str] = {
        "downloads": "Downloads",
        "documents": "Documents",
        "desktop": "Desktop",
        "pictures": "Pictures",
        "music": "Music",
        "videos": "Videos",
        "pengu": "pengu",
        "projects": "projects",
    }

    def _resolve_shell_folder(self, name: str) -> Optional[Path]:
        """Resolve a folder name like 'Downloads' to the real path."""
        name_lower = name.lower().strip().rstrip(".")

        # Check shell folder names
        shell_name = self._SHELL_FOLDERS.get(name_lower)
        if shell_name:
            if name_lower in ("pengu", "projects"):
                # Relative to home or pengu root
                home_dir = Path.home()
                candidates = [
                    home_dir / shell_name,
                    home_dir / "projects" / shell_name,
                ]
                # Also check FoveaEdge_old path
                if (home_dir / "projects" / "FoveaEdge_old" / shell_name).exists():
                    candidates.append(home_dir / "projects" / "FoveaEdge_old" / shell_name)
            else:
                # Standard Windows shell folder
                candidates = [Path.home() / shell_name]

            for c in candidates:
                if c.exists() and c.is_dir():
                    return c

        # Try direct path
        direct = Path(name)
        if direct.exists() and direct.is_dir():
            return direct

        # Try relative to home
        home_dir = Path.home()
        home_sub = home_dir / name_lower
        if home_sub.exists() and home_sub.is_dir():
            return home_sub

        # Try projects directory
        projects_dir = home_dir / "projects"
        if projects_dir.exists():
            for d in projects_dir.iterdir():
                if d.is_dir() and name_lower in d.name.lower():
                    return d

        return None

    def _resolve_project(self, name: str) -> Optional[Path]:
        """Resolve a project name to a path."""
        name = name.strip().rstrip(".")

        if "pengu" in name.lower():
            return self._pengu_root

        candidates = [
            self._pengu_root / name,
            Path.home() / "projects" / name,
            Path.home() / name,
        ]
        projects = Path.home() / "projects"
        if projects.exists():
            for d in projects.iterdir():
                if d.is_dir() and name.lower() in d.name.lower():
                    candidates.append(d)
        for c in candidates:
            if c.exists():
                return c
        return None

    def _run_git(self, args: str) -> str:
        try:
            result = subprocess.run(
                ["git"] + args.split(),
                cwd=str(self._pengu_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() or result.stderr.strip() or "Done."
        except Exception as e:
            return f"Git error: {e}"

    def _get_system_info(self) -> str:
        import platform
        import psutil
        info = {
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": platform.processor(),
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "Free": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        }
        return "System: " + ", ".join(f"{k}: {v}" for k, v in info.items())

    def _list_files(self) -> str:
        entries = list(self._pengu_root.iterdir())
        dirs = [e.name for e in entries if e.is_dir()][:8]
        files = [e.name for e in entries if e.is_file()][:8]
        parts = []
        if dirs:
            parts.append(f"Folders: {', '.join(dirs)}")
        if files:
            parts.append(f"Files: {', '.join(files)}")
        return "Pengu contains: " + "; ".join(parts) if parts else "Pengu folder is empty."

    def _run_diagnostics(self) -> str:
        checks = []
        launcher = get_launcher()
        checks.append(f"VS Code: {'OK' if launcher.find_app('vscode') else 'NOT FOUND'}")
        checks.append(f"Chrome: {'OK' if launcher.find_app('chrome') else 'NOT FOUND'}")
        checks.append(f"Explorer: OK")
        import shutil
        checks.append(f"Git: {'OK' if shutil.which('git') else 'NOT FOUND'}")
        checks.append("Python: OK")
        return "Diagnostics: " + "; ".join(checks)


# ---------------------------------------------------------------------------
# Model Discovery
# ---------------------------------------------------------------------------

class ModelDiscovery:
    """Auto-detect available local model runtimes and models."""

    def __init__(self) -> None:
        self._active_provider: Optional[str] = None
        self._active_model: Optional[str] = None
        self._available_models: list[dict] = []

    async def discover(self) -> dict[str, Any]:
        """Discover available model runtimes and models."""
        result = {"lm_studio": None, "ollama": None, "active": None}

        # Check LM Studio
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:1234/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_ids = [m.get("id", "unknown") for m in models]
                        result["lm_studio"] = model_ids
                        self._active_provider = "lmstudio"
                        self._active_model = model_ids[0]
                        self._available_models = [{"provider": "lmstudio", "id": mid} for mid in model_ids]
                        logger.info("lmstudio_models_found", models=model_ids)
        except Exception:
            pass

        # Check Ollama
        if not self._active_provider:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get("http://localhost:11434/api/tags")
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("models", [])
                        if models:
                            model_names = [m.get("name", "unknown") for m in models]
                            result["ollama"] = model_names
                            self._active_provider = "ollama"
                            self._active_model = model_names[0]
                            self._available_models = [{"provider": "ollama", "id": n} for n in model_names]
                            logger.info("ollama_models_found", models=model_names)
            except Exception:
                pass

        result["active"] = {
            "provider": self._active_provider,
            "model": self._active_model,
        }
        return result

    @property
    def active_provider(self) -> Optional[str]:
        return self._active_provider

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    @property
    def available_models(self) -> list[dict]:
        return self._available_models


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class PenguApp:
    """Main Pengu application."""

    def __init__(self) -> None:
        self._config = get_config()
        self._voice_config = VoiceConfig()
        self._tool_registry = ToolRegistry()
        self._voice: Optional[VoiceEngine] = None
        self._overlay: Optional[PenguOverlay] = None
        self._tray: Optional[PenguTray] = None
        self._parser = CommandParser(_get_pengu_root())
        self._model_discovery = ModelDiscovery()
        self._provider = None  # LMStudioProvider set after discovery
        self._pipeline = None  # CommandPipeline built after discovery
        self._running = False

    async def start(self) -> None:
        """Start the Pengu application."""
        logger.info("pengu_starting")
        self._running = True

        # Register tools
        register_deterministic_tools(self._tool_registry)

        # Discover models
        discovery_result = await self._model_discovery.discover()
        logger.info("model_discovery", **discovery_result)

        # Initialize LLM provider if available
        if self._model_discovery.active_provider == "lmstudio":
            from pengu.models.lmstudio import LMStudioProvider
            self._provider = LMStudioProvider()
            await self._provider.health_check()

        # Start overlay
        self._overlay = PenguOverlay()
        self._overlay.start()
        time.sleep(0.5)

        # Start tray
        self._tray = PenguTray(
            on_start=self._on_tray_start,
            on_pause=self._on_tray_pause,
            on_resume=self._on_tray_resume,
            on_exit=self._on_tray_exit,
        )
        self._tray.start()

        # Start voice engine
        self._voice = VoiceEngine(
            config=self._voice_config,
            command_callback=self._process_command,
            state_callback=self._on_voice_state_change,
        )
        status = await self._voice.initialize()
        if status.get("stt") or status.get("tts"):
            await self._voice.start()

        # Register global hotkey (Ctrl+Alt+P)
        hotkey = get_hotkey()
        if hotkey.register_default(callback=self._on_hotkey_pressed):
            hotkey.start()
            logger.info("hotkey_registered", combo="Ctrl+Alt+P")
        else:
            logger.warning("hotkey_registration_failed")

        logger.info("pengu_ready", model=self._model_discovery.active_model)

        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the Pengu application."""
        self._running = False
        # Stop hotkey listener
        hotkey = get_hotkey()
        hotkey.stop()
        if self._voice:
            await self._voice.stop()
        if self._overlay:
            self._overlay.stop()
        if self._tray:
            self._tray.stop()
        if self._provider:
            await self._provider.close()

    def _process_command(self, text: str) -> Optional[str]:
        """Process a voice command. Returns spoken response."""
        ctx = get_context()
        logger.info("processing_command", text=text)

        # Resolve follow-up commands using context
        resolved = ctx.resolve_followup(text)
        if resolved != text:
            logger.info("context_resolved", original=text, resolved=resolved)
            text = resolved

        # Try deterministic parser first (fast, covers known apps/folders/git)
        result = self._parser.parse(text)
        if result:
            ctx.add_turn(text, result.get("speak", "Done."), action_taken=result.get("action", ""))
            return result.get("speak", "Done.")

        # Check for multi-step tasks ("open Chrome and search for X")
        import re
        if re.search(r'\b(?:and|then|;|also)\s+', text, re.IGNORECASE):
            return self._process_multi_step(text)

        # Fall through to the full command pipeline
        return self._process_with_pipeline(text)

    def _process_multi_step(self, text: str) -> str:
        """Process a multi-step command using the TaskPlanner."""
        import concurrent.futures
        try:
            from pengu.agent.planner import get_planner, get_executor
            planner = get_planner()
            executor = get_executor()
            plan = planner.create_plan(text)
            logger.info("multi_step_plan", steps=len(plan.steps), goal=plan.goal[:80])

            # Execute the plan in a thread to avoid event-loop conflicts
            def _run_plan():
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(executor.execute_plan(plan))
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_plan)
                response = future.result(timeout=60)

            ctx = get_context()
            ctx.add_turn(text, response, action_taken="multi_step")
            return response
        except Exception as e:
            logger.error("multi_step_error", error=str(e))
            return self._process_with_pipeline(text)

    def _process_with_pipeline(self, text: str) -> str:
        """Process a command through the full CommandPipeline."""
        from pengu.pipeline import CommandPipeline

        # Build a lightweight pipeline if not already built
        if not hasattr(self, '_pipeline') or self._pipeline is None:
            self._pipeline = CommandPipeline(self._tool_registry, self._provider)

        # Run the pipeline in a thread to avoid event-loop conflicts
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    def _run_pipeline():
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(
                                self._pipeline.process(text)
                            )
                        finally:
                            new_loop.close()
                    future = pool.submit(_run_pipeline)
                    pipeline_result = future.result(timeout=30)
            else:
                pipeline_result = loop.run_until_complete(
                    self._pipeline.process(text)
                )

            ctx = get_context()
            ctx.add_turn(text, pipeline_result.response, action_taken=pipeline_result.tool_used)
            return pipeline_result.response

        except Exception as e:
            logger.error("pipeline_error", error=str(e))
            return self._chat_with_llm(text)

    def _chat_with_llm(self, text: str) -> str:
        """Chat with the local LLM for general questions."""
        ctx = get_context()
        if not self._provider or not self._provider.is_available():
            return (
                "I can execute desktop commands, but no local language model "
                "is currently available for general questions. "
                "Please start LM Studio and load a model."
            )

        from pengu.models.base import ChatMessage
        loop = asyncio.new_event_loop()
        try:
            # Include context in the system prompt for multi-turn awareness
            context_summary = ctx.get_summary()
            context_info = ""
            if context_summary.get("current_app"):
                context_info += f"Current app: {context_summary['current_app']}. "
            if context_summary.get("current_url"):
                context_info += f"Current URL: {context_summary['current_url']}. "
            if context_summary.get("current_directory"):
                context_info += f"Current directory: {context_summary['current_directory']}. "

            system_msg = (
                "You are Pengu, a local-first desktop assistant running on Windows. "
                "Be concise and helpful. Respond in 1-2 sentences."
            )
            if context_info:
                system_msg += f"\nSession context: {context_info}"

            messages = [
                ChatMessage(role="system", content=system_msg),
                ChatMessage(role="user", content=text),
            ]
            response = loop.run_until_complete(
                self._provider.chat(messages, temperature=0.7, max_tokens=256)
            )
            resp_text = response.content if not response.error else f"Model error: {response.error}"
            ctx.add_turn(text, resp_text, action_taken="llm_chat")
            return resp_text
        except Exception as e:
            return f"Model error: {e}"
        finally:
            loop.close()

    def _on_voice_state_change(self, state: VoiceState) -> None:
        """Handle voice state changes — update overlay and tray."""
        state_map = {
            VoiceState.OFFLINE: OverlayState.ERROR,
            VoiceState.STARTING: OverlayState.THINKING,
            VoiceState.STANDBY: OverlayState.STANDBY,
            VoiceState.WAKE_DETECTED: OverlayState.LISTENING,
            VoiceState.ACKNOWLEDGING: OverlayState.SPEAKING,
            VoiceState.LISTENING: OverlayState.LISTENING,
            VoiceState.TRANSCRIBING: OverlayState.THINKING,
            VoiceState.THINKING: OverlayState.THINKING,
            VoiceState.EXECUTING: OverlayState.EXECUTING,
            VoiceState.SPEAKING: OverlayState.SPEAKING,
            VoiceState.ERROR: OverlayState.ERROR,
            VoiceState.STOPPING: OverlayState.THINKING,
        }
        if self._overlay:
            self._overlay.set_state(state_map.get(state, OverlayState.STANDBY))

        if self._tray:
            if state in (VoiceState.LISTENING, VoiceState.WAKE_DETECTED, VoiceState.ACKNOWLEDGING):
                self._tray.set_state(TrayState.LISTENING)
            elif state == VoiceState.ERROR:
                self._tray.set_state(TrayState.ERROR)
            else:
                self._tray.set_state(TrayState.LISTENING)

    def _on_tray_start(self) -> None:
        if self._voice and not self._voice.is_running:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._voice.start())

    def _on_tray_pause(self) -> None:
        if self._voice:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._voice.stop())

    def _on_tray_resume(self) -> None:
        if self._voice:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._voice.start())

    def _on_hotkey_pressed(self) -> None:
        """Handle global hotkey press — toggle voice activation."""
        if self._voice:
            if self._voice.is_running:
                # If currently in STANDBY, simulate wake
                from pengu.voice.engine import VoiceState
                if self._voice.state == VoiceState.STANDBY:
                    # Trigger wake manually
                    self._voice._wake_detector._last_wake_time = 0
                    logger.info("hotkey_wake_triggered")

    def _on_tray_exit(self) -> None:
        self._running = False


async def main() -> None:
    """Main entry point."""
    setup_logging(level="INFO", json_output=False)
    app = PenguApp()
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
