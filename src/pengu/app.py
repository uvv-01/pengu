"""
Pengu Application — production voice-first desktop assistant.

Architecture:
  Voice → STT → Command Parser → Deterministic Tool / LLM → TTS → Standby
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from pengu.config import get_config
from pengu.logging import get_logger, setup_logging
from pengu.models.lmstudio import LMStudioProvider
from pengu.os.app_launcher import get_launcher
from pengu.tools.deterministic import register_deterministic_tools
from pengu.tools.registry import ToolRegistry
from pengu.voice.engine import VoiceConfig, VoiceEngine, VoiceState
from pengu.ui.overlay import PenguOverlay, OverlayState
from pengu.ui.tray import PenguTray, TrayState

logger = get_logger("pengu.app")


def _get_pengu_root() -> Path:
    """Get the Pengu project root directory."""
    return Path(__file__).resolve().parent.parent.parent


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

        # ---- APPLICATION LAUNCHING ----
        # "open VS Code" / "launch Chrome" / "start Edge"
        m = re.match(
            r'^(?:open|launch|start|run)\s+(.+?)(?:\s+(?:in|with)\s+(.+))?$',
            text_lower,
        )
        if m:
            target = m.group(1).strip().rstrip(".")
            context = m.group(2).strip() if m.group(2) else None

            # "open Pengu in VS Code"
            if context and ("vs code" in context or "visual studio code" in context or "code" in context):
                folder = self._resolve_project(target)
                return {"action": "open_in_vscode", "target": folder, "speak": f"Opening {target} in Visual Studio Code."}

            # "open ChatGPT"
            if "chatgpt" in target or "chat gpt" in target:
                return {"action": "open_chatgpt", "speak": "Opening ChatGPT."}

            # "open Google" / "open YouTube" / "open GitHub"
            known_urls = {
                "google": "https://www.google.com",
                "youtube": "https://www.youtube.com",
                "github": "https://github.com",
                "gmail": "https://mail.google.com",
            }
            for name, url in known_urls.items():
                if name in target:
                    return {"action": "open_url", "target": url, "speak": f"Opening {name}."}

            # "search Google for X"
            if "google" in target or "search" in target:
                search_term = target
                for word in ["google", "search", "for", "the", "web", "and"]:
                    search_term = search_term.replace(word, "")
                search_term = search_term.strip()
                if search_term:
                    return {"action": "google_search", "target": search_term, "speak": f"Searching Google for {search_term}."}

            # Check if it's a known folder
            folder = self._resolve_project(target)
            if folder and folder.exists():
                if folder.is_dir():
                    return {"action": "open_folder", "target": str(folder), "speak": f"Opening {folder.name} folder."}
                else:
                    return {"action": "open_file", "target": str(folder), "speak": f"Opening {folder.name}."}

            # Try as application name
            app = self._launcher.find_app(target)
            if app:
                result = self._launcher.open_application(target)
                if result["success"]:
                    return {"action": "open_app", "target": app.name, "speak": result["message"]}
                else:
                    return {"action": "error", "speak": result["message"]}

            # Unknown application
            return {"action": "error", "speak": f"I couldn't find {target} on your system."}

        # ---- SEARCH GOOGLE ----
        m = re.match(
            r'^(?:search|google|look\s+up|find)\s+(?:google\s+)?(?:for\s+)?(.+)',
            text_lower,
        )
        if m:
            query = m.group(1).strip()
            if query:
                return {"action": "google_search", "target": query, "speak": f"Searching Google for {query}."}

        # ---- SEARCH CHATGPT ----
        m = re.match(
            r'^(?:search|ask)\s+chatgpt\s+(?:for\s+)?(.+)',
            text_lower,
        )
        if m:
            query = m.group(1).strip()
            if query:
                url = f"https://chatgpt.com/?q={query.replace(' ', '+')}"
                return {"action": "open_url", "target": url, "speak": f"Opening ChatGPT with your question."}

        # ---- FILE OPERATIONS ----
        # "create file X" / "create a file called X"
        m = re.search(
            r'(?:create|make|new)\s+(?:a\s+)?(?:file\s+(?:called\s+|named\s+)?)?([^\s]+\.\w+)',
            text_lower,
        )
        if m:
            filename = m.group(1).strip()
            file_path = self._pengu_root / filename
            try:
                file_path.touch()
                return {"action": "file_created", "target": filename, "speak": f"Created {filename} in the Pengu folder."}
            except Exception as e:
                return {"action": "error", "speak": f"Failed to create {filename}: {e}"}

        # "create folder X"
        m = re.search(
            r'(?:create|make|new)\s+(?:a\s+)?folder\s+(?:called\s+|named\s+)?([^\s]+)',
            text_lower,
        )
        if m:
            foldername = m.group(1).strip()
            folder_path = self._pengu_root / foldername
            try:
                folder_path.mkdir(exist_ok=True)
                return {"action": "folder_created", "target": foldername, "speak": f"Created folder {foldername}."}
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
        if any(w in text_lower for w in ["system info", "system information", "what cpu", "what ram", "what processor", "computer info"]):
            return {"action": "system_info", "speak": self._get_system_info()}

        # ---- LIST FILES ----
        if any(w in text_lower for w in ["list files", "show files", "what's in", "what is in", "show me files"]):
            return {"action": "list_files", "speak": self._list_files()}

        # ---- OPEN PENGU ----
        if "open pengu" in text_lower:
            if "vs code" in text_lower or "code" in text_lower:
                result = self._launcher.open_in_vscode(str(self._pengu_root))
                return {"action": "open_in_vscode", "speak": result["message"]}
            else:
                result = self._launcher.open_folder(str(self._pengu_root))
                return {"action": "open_folder", "speak": result["message"]}

        # ---- STOP ----
        if text_lower in ["stop", "cancel", "never mind", "nevermind", "forget it"]:
            return {"action": "stop", "speak": "OK, stopped."}

        # ---- DIAGNOSTICS ----
        if any(w in text_lower for w in ["diagnostics", "diagnostic", "run diagnostics", "test yourself", "doctor"]):
            return {"action": "diagnostics", "speak": self._run_diagnostics()}

        # ---- HELP ----
        if any(w in text_lower for w in ["help", "what can you do", "commands"]):
            return {"action": "help", "speak": "I can open applications, search Google, work with files, run Git commands, and answer questions using a local AI model. Try saying: open VS Code, search Google for Python, or what is Python?"}

        # None = send to LLM
        return None

    def _resolve_project(self, name: str) -> Optional[Path]:
        """Resolve a project name to a path."""
        name = name.strip().rstrip(".")

        # Check if it's the Pengu project itself
        if "pengu" in name.lower():
            return self._pengu_root

        candidates = [
            self._pengu_root / name,
            Path.home() / "projects" / name,
            Path.home() / name,
        ]
        # Fuzzy match in projects dir
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
        checks.append(f"Python: OK")
        return "Diagnostics: " + "; ".join(checks)


class PenguApp:
    """Main Pengu application."""

    def __init__(self) -> None:
        self._config = get_config()
        self._voice_config = VoiceConfig()
        self._tool_registry = ToolRegistry()
        self._provider: Optional[LMStudioProvider] = None
        self._voice: Optional[VoiceEngine] = None
        self._overlay: Optional[PenguOverlay] = None
        self._tray: Optional[PenguTray] = None
        self._parser = CommandParser(_get_pengu_root())
        self._running = False

    async def start(self) -> None:
        logger.info("pengu_starting")
        self._running = True

        register_deterministic_tools(self._tool_registry)

        self._provider = LMStudioProvider()
        await self._provider.health_check()

        self._overlay = PenguOverlay()
        self._overlay.start()
        time.sleep(0.5)

        self._tray = PenguTray(
            on_start=self._on_tray_start,
            on_pause=self._on_tray_pause,
            on_resume=self._on_tray_resume,
            on_exit=self._on_tray_exit,
        )
        self._tray.start()

        self._voice = VoiceEngine(
            config=self._voice_config,
            command_callback=self._process_command,
            state_callback=self._on_voice_state_change,
        )
        status = await self._voice.initialize()
        if status.get("stt") or status.get("tts"):
            await self._voice.start()

        logger.info("pengu_ready")

        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
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
        # Try deterministic parser first
        result = self._parser.parse(text)
        if result:
            return result.get("speak", "Done.")

        # Fall through to LLM
        return self._chat_with_llm(text)

    def _chat_with_llm(self, text: str) -> str:
        if not self._provider or not self._provider.is_available():
            return "I can execute desktop commands, but no local language model is currently available for general questions. Please start LM Studio and load a model."

        from pengu.models.base import ChatMessage
        loop = asyncio.new_event_loop()
        try:
            messages = [
                ChatMessage(role="system", content="You are Pengu, a local-first desktop assistant. Be concise. Respond in 1-2 sentences."),
                ChatMessage(role="user", content=text),
            ]
            response = loop.run_until_complete(
                self._provider.chat(messages, temperature=0.7, max_tokens=256)
            )
            return response.content if not response.error else f"Model error: {response.error}"
        except Exception as e:
            return f"Model error: {e}"
        finally:
            loop.close()

    def _on_voice_state_change(self, state: VoiceState) -> None:
        state_map = {
            VoiceState.STANDBY: OverlayState.STANDBY,
            VoiceState.WAKE_DETECTED: OverlayState.LISTENING,
            VoiceState.LISTENING: OverlayState.LISTENING,
            VoiceState.TRANSCRIBING: OverlayState.THINKING,
            VoiceState.THINKING: OverlayState.THINKING,
            VoiceState.EXECUTING: OverlayState.EXECUTING,
            VoiceState.SPEAKING: OverlayState.SPEAKING,
            VoiceState.ERROR: OverlayState.ERROR,
        }
        if self._overlay:
            self._overlay.set_state(state_map.get(state, OverlayState.STANDBY))
        if self._tray:
            tray_state = TrayState.LISTENING if state in (VoiceState.LISTENING, VoiceState.WAKE_DETECTED) else TrayState.LISTENING
            if state == VoiceState.ERROR:
                tray_state = TrayState.ERROR
            self._tray.set_state(tray_state)

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

    def _on_tray_exit(self) -> None:
        self._running = False


async def main() -> None:
    setup_logging(level="INFO", json_output=False)
    app = PenguApp()
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
