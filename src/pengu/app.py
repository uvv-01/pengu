"""
Pengu Application — the real voice-first desktop assistant.

Ties together:
  - Voice engine (microphone, wake word, STT, TTS)
  - Desktop overlay UI
  - System tray
  - Application launcher
  - Command pipeline
  - Local LLM

Usage:
    python -m pengu
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
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


class PenguApp:
    """
    Main Pengu application — voice-first desktop assistant.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._voice_config = VoiceConfig()
        self._tool_registry = ToolRegistry()
        self._provider: Optional[LMStudioProvider] = None
        self._voice: Optional[VoiceEngine] = None
        self._overlay: Optional[PenguOverlay] = None
        self._tray: Optional[PenguTray] = None
        self._launcher = get_launcher()
        self._running = False
        self._context: dict = {}  # Short-term conversation context

    async def start(self) -> None:
        """Start the Pengu application."""
        logger.info("pengu_starting", version=self._config.version)
        self._running = True

        # Register deterministic tools
        register_deterministic_tools(self._tool_registry)
        logger.info("tools_registered", count=len(self._tool_registry.list_tools()))

        # Initialize LM Studio provider
        self._provider = LMStudioProvider()
        healthy = await self._provider.health_check()
        if healthy:
            logger.info("lmstudio_connected")
        else:
            logger.warning("lmstudio_unavailable")

        # Start overlay UI
        self._overlay = PenguOverlay()
        self._overlay.start()
        time.sleep(0.5)

        # Start system tray
        self._tray = PenguTray(
            on_start=self._on_tray_start,
            on_pause=self._on_tray_pause,
            on_resume=self._on_tray_resume,
            on_exit=self._on_tray_exit,
        )
        self._tray.start()

        # Initialize voice engine
        self._voice = VoiceEngine(
            config=self._voice_config,
            command_callback=self._process_command,
            state_callback=self._on_voice_state_change,
        )
        status = await self._voice.initialize()
        if status.get("stt") or status.get("tts"):
            await self._voice.start()
            logger.info("voice_engine_started")
        else:
            logger.warning("voice_engine_unavailable")

        logger.info("pengu_ready")

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        logger.info("pengu_stopping")
        self._running = False
        if self._voice:
            await self._voice.stop()
        if self._overlay:
            self._overlay.stop()
        if self._tray:
            self._tray.stop()
        if self._provider:
            await self._provider.close()
        logger.info("pengu_stopped")

    def _process_command(self, text: str) -> Optional[str]:
        """
        Process a voice command. Returns spoken response.
        This runs in the voice engine's thread.
        """
        text_lower = text.lower().strip()

        try:
            # ---- DIAGNOSTICS ----
            if any(w in text_lower for w in ["diagnostics", "diagnostic", "run diagnostics", "test yourself"]):
                return self._run_diagnostics()

            # ---- APPLICATION LAUNCHING ----
            # "open VS Code" / "open Chrome" / "launch Edge"
            if any(text_lower.startswith(w) for w in ["open ", "launch ", "start "]):
                target = text_lower
                for prefix in ["open ", "launch ", "start "]:
                    if target.startswith(prefix):
                        target = target[len(prefix):]
                        break

                # Check for "in VS Code" pattern
                if " in vs code" in target or " in code" in target:
                    folder = target.replace(" in vs code", "").replace(" in code", "").strip()
                    folder = self._resolve_folder(folder)
                    result = self._launcher.open_in_vscode(folder)
                    return result["message"]

                # Check for "in Chrome" pattern
                if " in chrome" in target:
                    query = target.replace(" in chrome", "").strip()
                    if "search" in query or "google" in query:
                        search_term = query.replace("search", "").replace("google", "").replace("for", "").strip()
                        result = self._launcher.google_search(search_term, "chrome")
                        return result["message"]
                    result = self._launcher.open_url(target.replace(" in chrome", "").strip())
                    return result["message"]

                # Check if it's a URL
                if target.startswith("http://") or target.startswith("https://"):
                    result = self._launcher.open_url(target)
                    return result["message"]

                # Check for Google search
                if "google" in target or "search" in target:
                    search_term = target
                    for word in ["google", "search", "for", "the", "web"]:
                        search_term = search_term.replace(word, "")
                    search_term = search_term.strip()
                    if search_term:
                        result = self._launcher.google_search(search_term)
                        return result["message"]

                # Check for ChatGPT
                if "chatgpt" in target or "chat gpt" in target:
                    result = self._launcher.open_chatgpt()
                    return result["message"]

                # Check for folder/file paths
                resolved = self._resolve_folder(target)
                if resolved and Path(resolved).exists():
                    if Path(resolved).is_dir():
                        result = self._launcher.open_folder(resolved)
                    else:
                        result = self._launcher.open_file(resolved)
                    return result["message"]

                # Try as application name
                result = self._launcher.open_application(target)
                return result["message"]

            # ---- FILE OPERATIONS ----
            # "create test.py" / "create a file called test.py"
            if any(w in text_lower for w in ["create file", "create a file", "new file", "make a file", "create "]):
                file_name = self._extract_filename(text_lower)
                if file_name:
                    return self._create_file(file_name)

            # "create folder X" / "create a folder called X"
            if "create folder" in text_lower or "new folder" in text_lower or "make a folder" in text_lower:
                folder_name = self._extract_folder_name(text_lower)
                if folder_name:
                    return self._create_folder(folder_name)

            # ---- GIT COMMANDS ----
            if "git status" in text_lower:
                return self._run_git("status")
            if "git log" in text_lower:
                return self._run_git("log --oneline -5")
            if "git diff" in text_lower:
                return self._run_git("diff")

            # ---- SYSTEM INFO ----
            if any(w in text_lower for w in ["system info", "system information", "what cpu", "what ram", "what processor"]):
                return self._get_system_info()

            # ---- LIST FILES ----
            if any(w in text_lower for w in ["list files", "show files", "what's in", "what is in", "show me the files"]):
                return self._list_files()

            # ---- CHAT / REASONING ----
            # Fall through to LLM for open-ended questions
            return self._chat_with_llm(text)

        except Exception as e:
            logger.error("command_processing_error", text=text, error=str(e))
            return f"Sorry, I encountered an error: {e}"

    def _resolve_folder(self, name: str) -> str:
        """Resolve a folder name to a full path."""
        name = name.strip().rstrip(".")

        # Check known project locations
        home = Path.home()
        projects = home / "projects"
        candidates = [
            Path(name),
            home / name,
            projects / name,
        ]

        # Also check common project directories
        if projects.exists():
            for d in projects.iterdir():
                if d.is_dir() and name.lower() in d.name.lower():
                    candidates.append(d)

        # Check the Pengu project specifically
        pengu_path = Path(__file__).parent.parent.parent.parent
        if "pengu" in name.lower():
            candidates.insert(0, pengu_path)

        for c in candidates:
            if c.exists() and c.is_dir():
                return str(c)

        return name  # Return as-is if not found

    def _extract_filename(self, text: str) -> Optional[str]:
        """Extract filename from command text."""
        import re
        patterns = [
            r"create\s+(?:a\s+)?(?:file\s+(?:called\s+)?|named\s+)?([^\s]+\.\w+)",
            r"new\s+(?:file\s+)?([^\s]+\.\w+)",
            r"make\s+(?:a\s+)?(?:file\s+)?([^\s]+\.\w+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_folder_name(self, text: str) -> Optional[str]:
        """Extract folder name from command text."""
        import re
        patterns = [
            r"create\s+(?:a\s+)?folder\s+(?:called\s+|named\s+)?([^\s]+)",
            r"new\s+folder\s+([^\s]+)",
            r"make\s+(?:a\s+)?folder\s+([^\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _create_file(self, filename: str) -> str:
        """Create a file in the Pengu project directory."""
        pengu_path = Path(__file__).parent.parent.parent.parent
        file_path = pengu_path / filename

        try:
            file_path.touch()
            logger.info("file_created", path=str(file_path))
            return f"Created {filename} in the Pengu folder."
        except Exception as e:
            return f"Failed to create {filename}: {e}"

    def _create_folder(self, foldername: str) -> str:
        """Create a folder in the Pengu project directory."""
        pengu_path = Path(__file__).parent.parent.parent.parent
        folder_path = pengu_path / foldername

        try:
            folder_path.mkdir(exist_ok=True)
            logger.info("folder_created", path=str(folder_path))
            return f"Created folder {foldername} in the Pengu folder."
        except Exception as e:
            return f"Failed to create folder {foldername}: {e}"

    def _run_git(self, args: str) -> str:
        """Run a git command in the Pengu directory."""
        import subprocess
        pengu_path = Path(__file__).parent.parent.parent.parent

        try:
            result = subprocess.run(
                ["git"] + args.split(),
                cwd=str(pengu_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                error = result.stderr.strip()
                return f"Git error: {error}" if error else "Git command failed."
            return output if output else "Git command completed with no output."
        except Exception as e:
            return f"Git command failed: {e}"

    def _get_system_info(self) -> str:
        """Get system information."""
        import platform
        import psutil

        info = {
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": platform.processor(),
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "Free RAM": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            "Python": platform.python_version(),
        }
        return "System info: " + ", ".join(f"{k}: {v}" for k, v in info.items())

    def _list_files(self, path: str = ".") -> str:
        """List files in a directory."""
        pengu_path = Path(__file__).parent.parent.parent.parent
        target = pengu_path / path

        try:
            entries = list(target.iterdir())
            dirs = [e.name for e in entries if e.is_dir()][:10]
            files = [e.name for e in entries if e.is_file()][:10]

            parts = []
            if dirs:
                parts.append(f"Folders: {', '.join(dirs)}")
            if files:
                parts.append(f"Files: {', '.join(files)}")

            return "Pengu folder contains: " + "; ".join(parts) if parts else "Pengu folder is empty."
        except Exception as e:
            return f"Failed to list files: {e}"

    def _chat_with_llm(self, text: str) -> str:
        """Chat with the local LLM."""
        if not self._provider or not self._provider.is_available():
            return "No local model is loaded. Please start LM Studio and load a model."

        from pengu.models.base import ChatMessage

        loop = asyncio.new_event_loop()
        try:
            messages = [
                ChatMessage(
                    role="system",
                    content="You are Pengu, a local-first desktop assistant. Be concise. Respond in 1-2 sentences.",
                ),
                ChatMessage(role="user", content=text),
            ]
            response = loop.run_until_complete(
                self._provider.chat(messages, temperature=0.7, max_tokens=256)
            )
            if response.error:
                return f"Model error: {response.error}"
            return response.content
        except Exception as e:
            return f"Model error: {e}"
        finally:
            loop.close()

    def _run_diagnostics(self) -> str:
        """Run full system diagnostics."""
        checks = []

        # Microphone
        mic_ok = self._voice and self._voice.get_status().get("microphone_active", False)
        checks.append(f"Microphone: {'OK' if mic_ok else 'NOT AVAILABLE'}")

        # STT
        stt_ok = self._voice and self._voice.get_status().get("stt_available", False)
        checks.append(f"Speech-to-text: {'OK' if stt_ok else 'NOT AVAILABLE'}")

        # TTS
        tts_ok = self._voice and self._voice.get_status().get("tts_available", False)
        checks.append(f"Text-to-speech: {'OK' if tts_ok else 'NOT AVAILABLE'}")

        # Model
        model_ok = self._provider and self._provider.is_available()
        checks.append(f"Local model: {'OK' if model_ok else 'NOT AVAILABLE'}")

        # VS Code
        code_exists = self._launcher.find_app("vscode") is not None
        checks.append(f"VS Code: {'OK' if code_exists else 'NOT FOUND'}")

        # Chrome
        chrome_exists = self._launcher.find_app("chrome") is not None
        checks.append(f"Chrome: {'OK' if chrome_exists else 'NOT FOUND'}")

        # File Explorer
        checks.append("File Explorer: OK")

        # Git
        import shutil
        git_ok = shutil.which("git") is not None
        checks.append(f"Git: {'OK' if git_ok else 'NOT FOUND'}")

        return "Diagnostics: " + "; ".join(checks)

    def _on_voice_state_change(self, state: VoiceState) -> None:
        state_map = {
            VoiceState.IDLE: OverlayState.STANDBY,
            VoiceState.LISTENING: OverlayState.LISTENING,
            VoiceState.PROCESSING: OverlayState.THINKING,
            VoiceState.SPEAKING: OverlayState.SPEAKING,
            VoiceState.ERROR: OverlayState.ERROR,
        }
        if self._overlay:
            self._overlay.set_state(state_map.get(state, OverlayState.STANDBY))
        if self._tray:
            tray_state = TrayState.LISTENING
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
    """Main entry point."""
    setup_logging(level="INFO", json_output=False)
    app = PenguApp()
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
