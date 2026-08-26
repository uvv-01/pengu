"""
Text-to-speech provider for Pengu.

Provides:
  - TextToSpeechProvider: abstract interface
  - EdgeTTSProvider: uses edge-tts (free, Microsoft, natural voices)
  - MockTTS: for testing

All providers are optional. Text mode always works without TTS.

edge-tts:
  License: MIT
  Cost: Free (uses Microsoft Edge's free TTS service)
  Requires: internet connection (but no API key)
  Voices: High quality, natural sounding
  Fallback: If offline, TTS is unavailable but text still works
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.voice.tts")


@dataclass
class SpeechResult:
    """Result of text-to-speech synthesis."""
    audio_data: bytes
    format: str  # "mp3", "wav", etc.
    duration_ms: float
    text_length: int
    voice: str


class TextToSpeechProvider(ABC):
    """Abstract base class for text-to-speech."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._available = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if TTS is available."""
        ...

    @abstractmethod
    async def speak(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> Optional[SpeechResult]:
        """Convert text to speech."""
        ...

    @abstractmethod
    async def speak_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
    ) -> Optional[str]:
        """Save speech to a file and return the path."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...

    def is_available(self) -> bool:
        return self._available


class EdgeTTSProvider(TextToSpeechProvider):
    """
    Text-to-speech using edge-tts.

    License: MIT
    Cost: Free (uses Microsoft Edge's free TTS service)
    Requires: internet connection (no API key needed)
    Voices: High quality, natural sounding
    Default voice: en-US-GuyNeural (male) or en-US-JennyNeural (female)
    """

    def __init__(self, voice: str = "en-US-GuyNeural") -> None:
        super().__init__(name="edge-tts")
        self._voice = voice
        self._communicate = None

    async def health_check(self) -> bool:
        try:
            import edge_tts
            self._available = True
            return True
        except ImportError:
            self._available = False
            logger.warning("edge_tts_not_installed")
            return False

    async def speak(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> Optional[SpeechResult]:
        if not self._available:
            await self.health_check()

        if not self._available:
            return None

        try:
            import edge_tts
            import io

            target_voice = voice or self._voice
            start = time.perf_counter()

            communicate = edge_tts.Communicate(text, target_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            duration_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "tts_complete",
                text_length=len(text),
                voice=target_voice,
                audio_size=len(audio_data),
                duration_ms=round(duration_ms, 2),
            )

            return SpeechResult(
                audio_data=audio_data,
                format="mp3",
                duration_ms=duration_ms,
                text_length=len(text),
                voice=target_voice,
            )

        except Exception as e:
            logger.error("tts_failed", error=str(e))
            return None

    async def speak_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
    ) -> Optional[str]:
        result = await self.speak(text, voice)
        if result:
            with open(output_path, "wb") as f:
                f.write(result.audio_data)
            return output_path
        return None

    async def close(self) -> None:
        pass


class MockTTS(TextToSpeechProvider):
    """Mock TTS for testing."""

    def __init__(self) -> None:
        super().__init__(name="mock")
        self._available = True

    async def health_check(self) -> bool:
        return True

    async def speak(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> Optional[SpeechResult]:
        return SpeechResult(
            audio_data=b"mock_audio",
            format="mp3",
            duration_ms=100.0,
            text_length=len(text),
            voice=voice or "mock",
        )

    async def speak_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
    ) -> Optional[str]:
        with open(output_path, "wb") as f:
            f.write(b"mock_audio")
        return output_path

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tts: Optional[TextToSpeechProvider] = None


def get_tts_provider() -> TextToSpeechProvider:
    """Get the TTS provider (lazy initialization)."""
    global _tts
    if _tts is None:
        _tts = EdgeTTSProvider()
    return _tts


def reset_tts() -> TextToSpeechProvider:
    """Reset the TTS provider (for testing)."""
    global _tts
    _tts = EdgeTTSProvider()
    return _tts
