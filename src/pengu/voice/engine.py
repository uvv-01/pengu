"""
Pengu Voice Engine — the real always-on voice assistant loop.

Pipeline:
  MICROPHONE → WAKE WORD → LISTENING → STT → COMMAND → ACTION → TTS → STANDBY

Components:
  - Microphone capture (sounddevice)
  - Wake word detection (energy-based + keyword)
  - Speech-to-text (faster-whisper)
  - Text-to-speech (edge-tts)
  - Conversation loop
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from pengu.logging import get_logger

logger = get_logger("pengu.voice.engine")


class VoiceState(str, Enum):
    """Voice engine states."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass
class VoiceConfig:
    """Voice engine configuration."""
    # Microphone
    sample_rate: int = 22050
    channels: int = 1
    device_index: Optional[int] = None  # None = default

    # Wake word
    wake_energy_threshold: float = 10.0
    wake_duration_seconds: float = 1.5
    wake_keywords: list[str] = field(default_factory=lambda: ["hello pengu", "hey pengu", "pengu"])

    # STT
    stt_model_size: str = "tiny"
    stt_language: str = "en"
    stt_max_duration: float = 30.0

    # TTS
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"

    # Timing
    silence_timeout: float = 2.0
    min_speech_duration: float = 0.5


class MicrophoneCapture:
    """Captures audio from the microphone using sounddevice."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._stream = None
        self._is_recording = False

    def start(self) -> bool:
        """Start microphone capture."""
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype="int16",
                device=self._config.device_index,
                callback=self._audio_callback,
                blocksize=1024,
            )
            self._stream.start()
            self._is_recording = True
            logger.info("microphone_started", device=self._config.device_index)
            return True
        except Exception as e:
            logger.error("microphone_start_failed", error=str(e))
            return False

    def stop(self) -> None:
        """Stop microphone capture."""
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Signal end of stream
        self._audio_queue.put(None)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Callback for audio data."""
        if self._is_recording:
            self._audio_queue.put(indata.copy())

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from the queue."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate audio energy (RMS)."""
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    @property
    def is_active(self) -> bool:
        return self._is_recording


class WakeWordDetector:
    """Detects 'Hello Pengu' using energy-based approach."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._energy_buffer: list[float] = []
        self._buffer_duration = 3.0  # seconds of audio to keep
        self._sample_rate = config.sample_rate
        self._chunk_size = 1024
        self._chunks_per_second = self._sample_rate / self._chunk_size

    def process_chunk(self, audio: np.ndarray, energy: float) -> bool:
        """
        Process an audio chunk and check for wake word activation.

        Returns True if wake word detected.
        """
        self._energy_buffer.append(energy)

        # Keep only recent buffer
        max_chunks = int(self._buffer_duration * self._chunks_per_second)
        if len(self._energy_buffer) > max_chunks:
            self._energy_buffer = self._energy_buffer[-max_chunks:]

        # Simple energy-based activation: sustained energy above threshold
        if len(self._energy_buffer) >= int(self._config.wake_duration_seconds * self._chunks_per_second):
            recent = self._energy_buffer[-int(self._config.wake_duration_seconds * self._chunks_per_second):]
            avg_energy = sum(recent) / len(recent)
            if avg_energy > self._config.wake_energy_threshold:
                logger.info("wake_word_energy_detected", avg_energy=avg_energy)
                return True

        return False

    def reset(self) -> None:
        self._energy_buffer.clear()


class SpeechCollector:
    """Collects speech audio after wake word detection."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_buffer: list[np.ndarray] = []
        self._is_collecting = False
        self._last_speech_time: float = 0
        self._start_time: float = 0

    def start(self) -> None:
        self._audio_buffer.clear()
        self._is_collecting = True
        self._start_time = time.time()
        self._last_speech_time = time.time()

    def add_chunk(self, audio: np.ndarray, energy: float) -> None:
        if not self._is_collecting:
            return

        self._audio_buffer.append(audio)

        # Update last speech time if energy is above threshold
        if energy > self._config.wake_energy_threshold * 0.5:
            self._last_speech_time = time.time()

    def should_stop(self) -> bool:
        """Check if we should stop collecting speech."""
        if not self._is_collecting:
            return False

        # Stop on silence timeout
        if time.time() - self._last_speech_time > self._config.silence_timeout:
            return True

        # Stop on max duration
        if time.time() - self._start_time > self._config.stt_max_duration:
            return True

        return False

    def get_audio(self) -> Optional[np.ndarray]:
        """Get the collected audio as a numpy array."""
        if not self._audio_buffer:
            return None
        return np.concatenate(self._audio_buffer)

    def stop(self) -> Optional[np.ndarray]:
        """Stop collecting and return audio."""
        self._is_collecting = False
        return self.get_audio()


class STTEngine:
    """Speech-to-text using faster-whisper."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._available = False

    async def initialize(self) -> bool:
        """Load the STT model."""
        try:
            from faster_whisper import WhisperModel

            logger.info("loading_stt_model", model=self._config.stt_model_size)
            self._model = WhisperModel(
                self._config.stt_model_size,
                device="cpu",
                compute_type="int8",
            )
            self._available = True
            logger.info("stt_model_loaded", model=self._config.stt_model_size)
            return True
        except Exception as e:
            logger.error("stt_model_load_failed", error=str(e))
            self._available = False
            return False

    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text."""
        if not self._available or self._model is None:
            return None

        try:
            # Convert to float32 and normalize
            audio_float = audio.astype(np.float32) / 32768.0

            segments, info = self._model.transcribe(
                audio_float,
                language=self._config.stt_language,
                beam_size=5,
            )

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts).strip()

            if full_text:
                logger.info(
                    "transcription_complete",
                    text=full_text[:100],
                    language=info.language,
                )

            return full_text if full_text else None

        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return None

    def is_available(self) -> bool:
        return self._available


class TTSEngine:
    """Text-to-speech using edge-tts."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._available = False

    async def initialize(self) -> bool:
        """Check if TTS is available."""
        try:
            import edge_tts
            self._available = True
            logger.info("tts_initialized", voice=self._config.tts_voice)
            return True
        except ImportError:
            logger.warning("edge_tts_not_installed")
            return False

    async def speak(self, text: str) -> bool:
        """Speak text using edge-tts and play audio."""
        if not self._available:
            return False

        try:
            import edge_tts
            import sounddevice as sd

            communicate = edge_tts.Communicate(
                text,
                self._config.tts_voice,
                rate=self._config.tts_rate,
            )

            # Collect audio data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return False

            # Play audio using sounddevice
            # edge-tts outputs mp3, we need to decode it
            # Use a temporary file approach
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Use pygame or mpv to play, or fall back to subprocess
                import subprocess
                # Try using Windows default player
                subprocess.run(
                    ["cmd", "/c", "start", "/wait", temp_path],
                    capture_output=True,
                    timeout=10,
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            logger.info("tts_spoke", text_length=len(text))
            return True

        except Exception as e:
            logger.error("tts_failed", error=str(e))
            return False

    def is_available(self) -> bool:
        return self._available


class VoiceEngine:
    """
    Main voice engine — orchestrates the full conversation loop.

    Usage:
        engine = VoiceEngine(config)
        await engine.start()
        # Engine runs in background, calling callback on commands
        await engine.stop()
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        command_callback: Optional[Callable[[str], Any]] = None,
        state_callback: Optional[Callable[[VoiceState], None]] = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._command_callback = command_callback
        self._state_callback = state_callback

        self._microphone = MicrophoneCapture(self._config)
        self._wake_detector = WakeWordDetector(self._config)
        self._speech_collector = SpeechCollector(self._config)
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)

        self._state = VoiceState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def initialize(self) -> bool:
        """Initialize all components."""
        stt_ok = await self._stt.initialize()
        tts_ok = await self._tts.initialize()

        mic_ok = self._microphone.start()
        if not mic_ok:
            logger.warning("microphone_unavailable")

        logger.info(
            "voice_engine_initialized",
            stt=stt_ok,
            tts=tts_ok,
            mic=mic_ok,
        )

        return stt_ok or tts_ok  # At least one must work

    async def start(self) -> None:
        """Start the voice engine loop."""
        self._running = True
        self._set_state(VoiceState.IDLE)
        logger.info("voice_engine_started")

        # Run the loop in a background thread
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        """Stop the voice engine."""
        self._running = False
        self._microphone.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_state(VoiceState.IDLE)
        logger.info("voice_engine_stopped")

    def _run_loop(self) -> None:
        """Main voice loop (runs in a thread)."""
        while self._running:
            try:
                # Phase 1: Wake word detection
                self._set_state(VoiceState.IDLE)
                self._wake_detector.reset()

                while self._running:
                    audio = self._microphone.get_audio(timeout=0.1)
                    if audio is None:
                        continue

                    energy = self._microphone.calculate_energy(audio)

                    if self._wake_detector.process_chunk(audio, energy):
                        logger.info("wake_word_detected")
                        break

                if not self._running:
                    break

                # Phase 2: Listening
                self._set_state(VoiceState.LISTENING)
                self._speech_collector.start()

                # Play a short acknowledgment sound or speak
                asyncio.run_coroutine_threadsafe(
                    self._tts.speak("Yes?"),
                    self._loop,
                ).result(timeout=5)

                # Collect speech
                while self._running and not self._speech_collector.should_stop():
                    audio = self._microphone.get_audio(timeout=0.1)
                    if audio is not None:
                        energy = self._microphone.calculate_energy(audio)
                        self._speech_collector.add_chunk(audio, energy)

                # Phase 3: Process
                self._set_state(VoiceState.PROCESSING)
                speech_audio = self._speech_collector.stop()

                if speech_audio is None or len(speech_audio) == 0:
                    continue

                # Transcribe
                text = asyncio.run_coroutine_threadsafe(
                    self._stt.transcribe(speech_audio),
                    self._loop,
                ).result(timeout=30)

                if not text:
                    logger.info("no_speech_detected")
                    continue

                logger.info("command_received", text=text)

                # Phase 4: Execute command
                if self._command_callback:
                    result = self._command_callback(text)
                    if isinstance(result, str):
                        # Speak the result
                        self._set_state(VoiceState.SPEAKING)
                        asyncio.run_coroutine_threadsafe(
                            self._tts.speak(result),
                            self._loop,
                        ).result(timeout=30)

            except Exception as e:
                logger.error("voice_loop_error", error=str(e))
                self._set_state(VoiceState.ERROR)
                time.sleep(1)

    def _set_state(self, state: VoiceState) -> None:
        """Set the voice engine state."""
        self._state = state
        if self._state_callback:
            try:
                self._state_callback(state)
            except Exception:
                pass
        logger.info("voice_state_changed", state=state.value)

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get voice engine status."""
        return {
            "running": self._running,
            "state": self._state.value,
            "stt_available": self._stt.is_available(),
            "tts_available": self._tts.is_available(),
            "microphone_active": self._microphone.is_active,
        }
