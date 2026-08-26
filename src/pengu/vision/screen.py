"""
Screen capture for Pengu vision system.

Provides safe screenshot capture using standard libraries.
Only captures when explicitly requested — no continuous surveillance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.vision.screen")


@dataclass
class Screenshot:
    """Captured screenshot."""
    path: str
    width: int
    height: int
    timestamp: float
    format: str = "png"


class ScreenCapture:
    """
    Safe screen capture for Pengu.

    Only captures when explicitly requested.
    Does NOT continuously capture the screen.
    """

    def __init__(self, output_dir: str = "data/screenshots") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def capture(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[Screenshot]:
        """
        Capture a screenshot.

        Args:
            region: Optional (x, y, width, height) to capture a specific region.
                   If None, captures the full screen.

        Returns:
            Screenshot object or None if capture failed.
        """
        try:
            from PIL import ImageGrab
            import io

            start = time.perf_counter()

            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()

            duration_ms = (time.perf_counter() - start) * 1000

            # Save to file
            timestamp = int(time.time() * 1000)
            filename = f"screenshot_{timestamp}.png"
            filepath = self._output_dir / filename

            screenshot.save(filepath, "PNG")

            result = Screenshot(
                path=str(filepath),
                width=screenshot.width,
                height=screenshot.height,
                timestamp=time.time(),
                format="png",
            )

            logger.info(
                "screenshot_captured",
                path=str(filepath),
                width=screenshot.width,
                height=screenshot.height,
                duration_ms=round(duration_ms, 2),
            )

            return result

        except ImportError:
            logger.warning("pillow_not_installed_for_screenshot")
            return None
        except Exception as e:
            logger.error("screenshot_failed", error=str(e))
            return None

    async def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Optional[Screenshot]:
        """Capture a specific screen region."""
        return await self.capture(region=(x, y, width, height))

    def get_output_dir(self) -> str:
        return str(self._output_dir)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_screen_capture: Optional[ScreenCapture] = None


def get_screen_capture() -> ScreenCapture:
    """Get the screen capture instance."""
    global _screen_capture
    if _screen_capture is None:
        _screen_capture = ScreenCapture()
    return _screen_capture
