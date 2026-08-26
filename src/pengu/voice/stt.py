"""
Speech-to-text provider for Pengu.

Provides:
  - SpeechToTextProvider: abstract interface
  - FasterWhisperSTT: uses faster-whisper (free, local, CPU-friendly)
  - MockSTT: for testing

All providers are optional. Text mode always works without STT.

Hardware requirements:
  - faster-whisper tiny model: ~1GB RAM
  - faster-whisper base model: ~1.5GB RAM
  - Runs on CPU without GPU
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.voice.stt")


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    language: str
    confidence: float
    duration_ms: float
    segments: list[dict[str, Any]] = None

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


class SpeechToTextProvider(ABC):
    """Abstract base class for speech-to-text."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._available = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if STT is available."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_data: Any,
        language: str = "en",
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio data to text."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...

    def is_available(self) -> bool:
        return self._available


class FasterWhisperSTT(SpeechToTextProvider):
    """
    Speech-to-text using faster-whisper.

    License: MIT
    Cost: Free, runs locally
    Runtime: faster-whisper Python package
    Model: tiny or base (CPU-friendly)

    Hardware:
      tiny: ~1GB RAM, good accuracy
      base: ~1.5GB RAM, better accuracy
    """

    def __init__(self, model_size: str = "tiny") -> None:
        super().__init__(name="faster-whisper")
        self._model_size = model_size
        self._model = None

    async def health_check(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            self._available = True
            return True
        except ImportError:
            self._available = False
            logger.warning("faster_whisper_not_installed")
            return False

    async def transcribe(
        self,
        audio_data: Any,
        language: str = "en",
    ) -> Optional[TranscriptionResult]:
        if not self._available:
            await self.health_check()

        if not self._available:
            return None

        try:
            from faster_whisper import WhisperModel

            if self._model is None:
                logger.info("loading_whisper_model", model=self._model_size)
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info("whisper_model_loaded", model=self._model_size)

            start = time.perf_counter()
            segments, info = self._model.transcribe(
                audio_data,
                language=language,
                beam_size=5,
            )

            text_parts = []
            segment_list = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                })

            duration_ms = (time.perf_counter() - start) * 1000
            full_text = " ".join(text_parts)

            logger.info(
                "transcription_complete",
                text_length=len(full_text),
                language=info.language,
                duration_ms=round(duration_ms, 2),
            )

            return TranscriptionResult(
                text=full_text,
                language=info.language,
                confidence=info.language_probability,
                duration_ms=duration_ms,
                segments=segment_list,
            )

        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return None

    async def close(self) -> None:
        self._model = None


class MockSTT(SpeechToTextProvider):
    """Mock STT for testing."""

    def __init__(self) -> None:
        super().__init__(name="mock")
        self._available = True
        self._transcriptions: list[str] = []

    async def health_check(self) -> bool:
        return True

    async def transcribe(
        self,
        audio_data: Any,
        language: str = "en",
    ) -> Optional[TranscriptionResult]:
        if self._transcriptions:
            text = self._transcriptions.pop(0)
            return TranscriptionResult(
                text=text,
                language=language,
                confidence=0.95,
                duration_ms=100.0,
            )
        return TranscriptionResult(
            text="",
            language=language,
            confidence=0.0,
            duration_ms=0.0,
        )

    async def close(self) -> None:
        pass

    def set_transcription(self, text: str) -> None:
        """Set the next transcription result for testing."""
        self._transcriptions.append(text)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_stt: Optional[SpeechToTextProvider] = None


def get_stt_provider() -> SpeechToTextProvider:
    """Get the STT provider (lazy initialization)."""
    global _stt
    if _stt is None:
        _stt = FasterWhisperSTT()
    return _stt


def reset_stt() -> SpeechToTextProvider:
    """Reset the STT provider (for testing)."""
    global _stt
    _stt = FasterWhisperSTT()
    return _stt
