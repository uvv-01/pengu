"""
Vision system for Pengu — screen capture and analysis.

Provides:
  - VisionProvider: abstract interface for image analysis
  - ScreenCapture: screenshot capture
  - LocalVisionProvider: uses a local vision model (optional)

All providers are optional. Text mode always works without vision.
"""

from __future__ import annotations

from pengu.vision.provider import VisionProvider, VisionResult, get_vision_provider
from pengu.vision.screen import ScreenCapture, get_screen_capture

__all__ = [
    "VisionProvider",
    "VisionResult",
    "get_vision_provider",
    "ScreenCapture",
    "get_screen_capture",
]
