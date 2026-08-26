"""
Pengu Application — the main entry point for the voice-first desktop assistant.

Ties together:
  - Voice engine (microphone, wake word, STT, TTS)
  - Desktop overlay UI
  - System tray
  - Command pipeline
  - Backend API

Usage:
    python -m pengu
    # or
    pengu
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
import time
from typing import Optional

from pengu.config import get_config
from pengu.logging import get_logger, setup_logging
from pengu.models.lmstudio import LMStudioProvider
from pengu.pipeline import CommandPipeline
from pengu.tools.deterministic import register_deterministic_tools
from pengu.tools.registry import ToolRegistry
from pengu.voice.engine import VoiceConfig, VoiceEngine, VoiceState
from pengu.ui.overlay import PenguOverlay, OverlayState
from pengu.ui.tray import PenguTray, TrayState

logger = get_logger("pengu.app")


class PenguApp:
    """
    Main Pengu application.

    Orchestrates all components for the voice-first desktop assistant experience.
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._voice_config = VoiceConfig()
        self._tool_registry = ToolRegistry()
        self._provider: Optional[LMStudioProvider] = None
        self._pipeline: Optional[CommandPipeline] = None
        self._voice: Optional[VoiceEngine] = None
        self._overlay: Optional[PenguOverlay] = None
        self._tray: Optional[PenguTray] = None
        self._running = False

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

        # Initialize pipeline
        self._pipeline = CommandPipeline(self._tool_registry, self._provider)
        logger.info("pipeline_initialized")

        # Start overlay UI
        self._overlay = PenguOverlay()
        self._overlay.start()
        time.sleep(0.5)  # Give tkinter time to initialize

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
        voice_ok = await self._voice.initialize()
        if voice_ok:
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
        """Stop the Pengu application."""
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

    def _process_command(self, text: str) -> str:
        """
        Process a voice command through the pipeline.

        Returns a spoken response.
        """
        if not self._pipeline:
            return "Pipeline not initialized."

        # Run the async pipeline in a sync context
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._pipeline.process(text))
            return result.response
        except Exception as e:
            logger.error("command_failed", text=text, error=str(e))
            return f"Error: {e}"
        finally:
            loop.close()

    def _on_voice_state_change(self, state: VoiceState) -> None:
        """Handle voice engine state changes."""
        state_map = {
            VoiceState.IDLE: OverlayState.STANDBY,
            VoiceState.LISTENING: OverlayState.LISTENING,
            VoiceState.PROCESSING: OverlayState.THINKING,
            VoiceState.SPEAKING: OverlayState.SPEAKING,
            VoiceState.ERROR: OverlayState.ERROR,
        }

        overlay_state = state_map.get(state, OverlayState.STANDBY)

        if self._overlay:
            self._overlay.set_state(overlay_state)

        if self._tray:
            tray_state = TrayState.LISTENING
            if state == VoiceState.ERROR:
                tray_state = TrayState.ERROR
            self._tray.set_state(tray_state)

    def _on_tray_start(self) -> None:
        """Handle tray Start action."""
        if self._voice and not self._voice.is_running:
            asyncio.get_event_loop().run_until_complete(self._voice.start())

    def _on_tray_pause(self) -> None:
        """Handle tray Pause action."""
        if self._voice:
            asyncio.get_event_loop().run_until_complete(self._voice.stop())

    def _on_tray_resume(self) -> None:
        """Handle tray Resume action."""
        if self._voice:
            asyncio.get_event_loop().run_until_complete(self._voice.start())

    def _on_tray_exit(self) -> None:
        """Handle tray Exit action."""
        self._running = False


async def main() -> None:
    """Main entry point."""
    setup_logging(level="INFO", json_output=False)

    app = PenguApp()

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("signal_received")
        app._running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
