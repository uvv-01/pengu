"""
Wake word detection provider for Pengu.

Target phrase: "Hello Pengu"

Provides:
  - WakeWordProvider: abstract interface
  - OpenWakeWordProvider: uses openWakeWord (free, local)
  - MockWakeWordProvider: for testing

All providers are optional. Text mode always works without wake word.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.voice.wake_word")


@dataclass
class WakeWordEvent:
    """Detected wake word event."""
    timestamp: float
    confidence: float
    phrase: str
    raw_score: float = 0.0


class WakeWordProvider(ABC):
    """Abstract base class for wake word detection."""

    def __init__(self, name: str, phrase: str = "hello pengu") -> None:
        self.name = name
        self.phrase = phrase
        self._available = False
        self._listening = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if wake word detection is available."""
        ...

    @abstractmethod
    async def start_listening(self) -> None:
        """Start listening for the wake word."""
        ...

    @abstractmethod
    async def stop_listening(self) -> None:
        """Stop listening for the wake word."""
        ...

    @abstractmethod
    async def detect_once(self, audio_data: Optional[Any] = None) -> Optional[WakeWordEvent]:
        """Detect wake word in a single audio chunk. Returns None if not detected."""
        ...

    def is_available(self) -> bool:
        return self._available

    def is_listening(self) -> bool:
        return self._listening


class OpenWakeWordProvider(WakeWordProvider):
    """
    Wake word detection using openWakeWord.

    License: Apache-2.0
    Cost: Free, runs locally
    Runtime: openWakeWord Python package

    Note: Requires microphone access and the openWakeWord package.
    """

    def __init__(self, phrase: str = "hello pengu") -> None:
        super().__init__(name="openwakeword", phrase=phrase)
        self._model = None

    async def health_check(self) -> bool:
        try:
            import openwakeword
            self._available = True
            return True
        except ImportError:
            self._available = False
            logger.warning("openwakeword_not_installed")
            return False

    async def start_listening(self) -> None:
        if not self._available:
            await self.health_check()
        if self._available:
            self._listening = True
            logger.info("wake_word_listening", phrase=self.phrase)

    async def stop_listening(self) -> None:
        self._listening = False

    async def detect_once(self, audio_data: Optional[Any] = None) -> Optional[WakeWordEvent]:
        if not self._available or audio_data is None:
            return None

        try:
            import openwakeword
            if self._model is None:
                self._model = openwakeword.ModelMultiple(wakeword_names=["hello_pengu"])

            prediction = self._model.predict(audio_data)
            score = prediction.get("hello_pengu", 0.0)

            if score > 0.5:
                return WakeWordEvent(
                    timestamp=time.time(),
                    confidence=score,
                    phrase=self.phrase,
                    raw_score=score,
                )
        except Exception as e:
            logger.warning("wake_word_detection_failed", error=str(e))

        return None


class MockWakeWordProvider(WakeWordProvider):
    """Mock wake word provider for testing."""

    def __init__(self, phrase: str = "hello pengu") -> None:
        super().__init__(name="mock", phrase=phrase)
        self._available = True
        self._detection_events: list[WakeWordEvent] = []

    async def health_check(self) -> bool:
        return True

    async def start_listening(self) -> None:
        self._listening = True

    async def stop_listening(self) -> None:
        self._listening = False

    async def detect_once(self, audio_data: Optional[Any] = None) -> Optional[WakeWordEvent]:
        if self._detection_events:
            return self._detection_events.pop(0)
        return None

    def simulate_detection(self, confidence: float = 0.9) -> None:
        """Simulate a wake word detection for testing."""
        self._detection_events.append(WakeWordEvent(
            timestamp=time.time(),
            confidence=confidence,
            phrase=self.phrase,
        ))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_wake_word: Optional[WakeWordProvider] = None


def get_wake_word_provider() -> WakeWordProvider:
    """Get the wake word provider (lazy initialization)."""
    global _wake_word
    if _wake_word is None:
        _wake_word = OpenWakeWordProvider()
    return _wake_word


def reset_wake_word() -> WakeWordProvider:
    """Reset the wake word provider (for testing)."""
    global _wake_word
    _wake_word = OpenWakeWordProvider()
    return _wake_word
