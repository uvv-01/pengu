"""
Pengu System Tray — Windows system tray icon.

Provides:
  - Tray icon with status indicator
  - Context menu (Start, Pause, Settings, Exit)
  - State display (LISTENING, PAUSED, ERROR)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.ui.tray")


class TrayState:
    """Tray icon states."""
    LISTENING = "LISTENING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class PenguTray:
    """
    System tray icon for Pengu.

    Uses pystray for cross-platform tray support.
    """

    def __init__(
        self,
        on_start: Optional[Callable[[], None]] = None,
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_start = on_start
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_exit = on_exit
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._state = TrayState.LISTENING

    def start(self) -> None:
        """Start the tray icon in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("tray_started")

    def stop(self) -> None:
        """Stop the tray icon."""
        if self._icon:
            self._icon.stop()
            self._icon = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("tray_stopped")

    def _run(self) -> None:
        """Run the tray icon."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            # Create a simple icon
            image = self._create_icon_image(TrayState.LISTENING)

            # Create menu
            menu = pystray.Menu(
                pystray.MenuItem("Start Listening", self._handle_start, default=True),
                pystray.MenuItem("Pause", self._handle_pause),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit", self._handle_exit),
            )

            self._icon = pystray.Icon(
                "pengu",
                image,
                "Pengu Assistant",
                menu,
            )

            self._icon.run()

        except ImportError:
            logger.warning("pystray_not_installed")
        except Exception as e:
            logger.error("tray_error", error=str(e))

    def _create_icon_image(self, state: str) -> Any:
        """Create a simple colored circle icon."""
        try:
            from PIL import Image, ImageDraw

            colors = {
                TrayState.LISTENING: "#00d4ff",
                TrayState.PAUSED: "#64748b",
                TrayState.ERROR: "#ef4444",
            }
            color = colors.get(state, "#64748b")

            image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse([8, 8, 56, 56], fill=color)
            return image

        except ImportError:
            return None

    def _handle_start(self, icon: Any, item: Any) -> None:
        if self._on_start:
            self._on_start()
        self.set_state(TrayState.LISTENING)

    def _handle_pause(self, icon: Any, item: Any) -> None:
        if self._state == TrayState.PAUSED:
            if self._on_resume:
                self._on_resume()
            self.set_state(TrayState.LISTENING)
        else:
            if self._on_pause:
                self._on_pause()
            self.set_state(TrayState.PAUSED)

    def _handle_exit(self, icon: Any, item: Any) -> None:
        if self._on_exit:
            self._on_exit()
        if self._icon:
            self._icon.stop()

    def set_state(self, state: str) -> None:
        """Update the tray icon state."""
        self._state = state
        if self._icon:
            try:
                image = self._create_icon_image(state)
                if image:
                    self._icon.icon = image
            except Exception:
                pass

    @property
    def is_running(self) -> bool:
        return self._icon is not None
