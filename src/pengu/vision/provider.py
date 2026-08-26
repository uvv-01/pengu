"""
Vision provider interface for Pengu.

Provides image analysis capabilities using local or cloud models.

All providers are optional. Text mode always works without vision.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.vision")


@dataclass
class VisionResult:
    """Result of image analysis."""
    description: str
    confidence: float
    analysis_time_ms: float
    model: str
    objects: list[str] = None
    text_detected: str = ""

    def __post_init__(self):
        if self.objects is None:
            self.objects = []


class VisionProvider(ABC):
    """Abstract base class for vision/image analysis."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._available = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if vision is available."""
        ...

    @abstractmethod
    async def analyze_image(
        self,
        image_data: Any,
        prompt: str = "Describe what you see in this image.",
    ) -> Optional[VisionResult]:
        """Analyze an image and return a description."""
        ...

    @abstractmethod
    async def analyze_screenshot(
        self,
        screenshot_path: str,
        prompt: str = "Describe what you see on this screen.",
    ) -> Optional[VisionResult]:
        """Analyze a screenshot file."""
        ...

    def is_available(self) -> bool:
        return self._available


class LMStudioVisionProvider(VisionProvider):
    """
    Vision analysis using LM Studio's multimodal model.

    Requires: LM Studio running with a multimodal model loaded.
    License: Depends on the model used
    Cost: Free (runs locally)

    Note: Many small models don't support vision.
    Requires a vision-capable model like LLaVA, BakLLaVA, or similar.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1") -> None:
        super().__init__(name="lmstudio-vision")
        self._base_url = base_url

    async def health_check(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    # Check if any model supports vision
                    self._available = len(models) > 0
                    return self._available
        except Exception:
            pass
        self._available = False
        return False

    async def analyze_image(
        self,
        image_data: Any,
        prompt: str = "Describe what you see in this image.",
    ) -> Optional[VisionResult]:
        if not self._available:
            return None

        try:
            import httpx
            import base64

            # Convert image_data to base64 if needed
            if isinstance(image_data, str):
                # Assume it's a file path
                with open(image_data, "rb") as f:
                    img_bytes = f.read()
            elif isinstance(image_data, bytes):
                img_bytes = image_data
            else:
                return None

            img_b64 = base64.b64encode(img_bytes).decode()

            start = time.perf_counter()

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 500,
                    },
                )

            duration_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return VisionResult(
                    description=content,
                    confidence=0.8,
                    analysis_time_ms=duration_ms,
                    model="lmstudio",
                )

        except Exception as e:
            logger.error("vision_analysis_failed", error=str(e))

        return None

    async def analyze_screenshot(
        self,
        screenshot_path: str,
        prompt: str = "Describe what you see on this screen.",
    ) -> Optional[VisionResult]:
        return await self.analyze_image(screenshot_path, prompt)


class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing."""

    def __init__(self) -> None:
        super().__init__(name="mock")
        self._available = True
        self._analysis_result = "Mock analysis result"

    async def health_check(self) -> bool:
        return True

    async def analyze_image(
        self,
        image_data: Any,
        prompt: str = "Describe what you see in this image.",
    ) -> Optional[VisionResult]:
        return VisionResult(
            description=self._analysis_result,
            confidence=0.95,
            analysis_time_ms=50.0,
            model="mock",
        )

    async def analyze_screenshot(
        self,
        screenshot_path: str,
        prompt: str = "Describe what you see on this screen.",
    ) -> Optional[VisionResult]:
        return await self.analyze_image(screenshot_path, prompt)

    def set_analysis_result(self, result: str) -> None:
        self._analysis_result = result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_vision: Optional[VisionProvider] = None


def get_vision_provider() -> VisionProvider:
    """Get the vision provider (lazy initialization)."""
    global _vision
    if _vision is None:
        _vision = LMStudioVisionProvider()
    return _vision


def reset_vision() -> VisionProvider:
    """Reset the vision provider (for testing)."""
    global _vision
    _vision = LMStudioVisionProvider()
    return _vision
