"""
Pengu Voice Engine — production-grade voice assistant pipeline.

Architecture:
  STANDBY → WAKE → LISTENING → TRANSCRIBING → THINKING → EXECUTING → SPEAKING → STANDBY

Features:
  - Energy-based wake word detection with debounce
  - TTS echo protection (mic muted during speech)
  - VAD-based command recording with silence timeout
  - Structured state machine
  - Error recovery
"""

from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from pengu.logging import get_logger

logger = get_logger("pengu.voice.engine")


class VoiceState(str, Enum):
    """Voice engine states."""
    STANDBY = "STANDBY"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


@dataclass
class VoiceConfig:
    """Voice engine configuration."""
    # Microphone
    sample_rate: int = 16000
    channels: int = 1
    device_index: Optional[int] = None

    # Wake word detection
    wake_energy_threshold: float = 15.0
    wake_min_speech_duration: float = 0.3
    wake_max_speech_duration: float = 3.0
    wake_debounce_seconds: float = 2.0

    # Command recording (VAD)
    command_silence_timeout: float = 1.2
    command_min_duration: float = 0.3
    command_max_duration: float = 15.0
    command_energy_threshold: float = 10.0

    # STT
    stt_model_size: str = "tiny"
    stt_language: str = "en"

    # TTS
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"


class MicrophoneCapture:
    """Captures audio from the microphone with mute control."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._stream = None
        self._is_recording = False
        self._is_muted = False
        self._lock = threading.Lock()

    def start(self) -> bool:
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
            logger.info("microphone_started")
            return True
        except Exception as e:
            logger.error("microphone_start_failed", error=str(e))
            return False

    def stop(self) -> None:
        self._is_recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if self._is_recording and not self._is_muted:
            self._audio_queue.put(indata.copy())

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def calculate_energy(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

    def mute(self) -> None:
        """Mute microphone (for TTS echo protection)."""
        with self._lock:
            self._is_muted = True
            self.flush()

    def unmute(self) -> None:
        """Unmute microphone."""
        with self._lock:
            self._is_muted = False

    def flush(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_active(self) -> bool:
        return self._is_recording

    @property
    def is_muted(self) -> bool:
        return self._is_muted


class WakeWordDetector:
    """
    Wake word detection using energy-based approach with debounce.

    Detects sustained speech followed by silence, with cooldown period.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._speech_start: float = 0
        self._speech_detected = False
        self._last_wake_time: float = 0

    def process_chunk(self, audio: np.ndarray, energy: float) -> bool:
        now = time.time()

        # Debounce: don't trigger too frequently
        if now - self._last_wake_time < self._config.wake_debounce_seconds:
            return False

        if not self._speech_detected:
            # Looking for speech onset
            if energy > self._config.wake_energy_threshold:
                self._speech_detected = True
                self._speech_start = now
        else:
            # Speech was detected, looking for pause (end of wake phrase)
            speech_duration = now - self._speech_start

            if energy < self._config.command_energy_threshold:
                # Silence after speech — wake word pattern detected
                if self._config.wake_min_speech_duration <= speech_duration <= self._config.wake_max_speech_duration:
                    logger.info("wake_word_detected", speech_duration=speech_duration)
                    self._speech_detected = False
                    self._last_wake_time = now
                    return True
                # Reset if speech too long or too short
                self._speech_detected = False
            elif speech_duration > self._config.wake_max_speech_duration:
                # Speech too long, reset
                self._speech_detected = False

        return False

    def reset(self) -> None:
        self._speech_detected = False
        self._speech_start = 0


class CommandRecorder:
    """Records command audio with VAD-based silence detection."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._last_speech_time: float = 0
        self._start_time: float = 0

    def start(self) -> None:
        self._audio_buffer.clear()
        self._is_recording = True
        self._start_time = time.time()
        self._last_speech_time = time.time() - self._config.command_silence_timeout  # Start in silence

    def add_chunk(self, audio: np.ndarray, energy: float) -> None:
        if not self._is_recording:
            return
        self._audio_buffer.append(audio)
        if energy > self._config.command_energy_threshold:
            self._last_speech_time = time.time()

    def should_stop(self) -> bool:
        if not self._is_recording:
            return False
        now = time.time()
        # Stop on silence timeout (only after some speech was detected)
        if self._last_speech_time > self._start_time:
            if now - self._last_speech_time > self._config.command_silence_timeout:
                return True
        # Stop on max duration
        if now - self._start_time > self._config.command_max_duration:
            return True
        return False

    def stop(self) -> Optional[np.ndarray]:
        self._is_recording = False
        if not self._audio_buffer:
            return None
        audio = np.concatenate(self._audio_buffer)
        duration = len(audio) / self._config.sample_rate
        if duration < self._config.command_min_duration:
            return None
        return audio


class STTEngine:
    """Speech-to-text using faster-whisper."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._available = False

    async def initialize(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            logger.info("loading_stt_model", model=self._config.stt_model_size)
            self._model = WhisperModel(
                self._config.stt_model_size,
                device="cpu",
                compute_type="int8",
            )
            self._available = True
            return True
        except Exception as e:
            logger.error("stt_model_load_failed", error=str(e))
            return False

    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        if not self._available or self._model is None:
            return None
        try:
            audio_float = audio.astype(np.float32) / 32768.0
            start = time.time()
            segments, info = self._model.transcribe(
                audio_float,
                language=self._config.stt_language,
                beam_size=5,
            )
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()
            latency = time.time() - start
            logger.info(
                "stt_complete",
                text=full_text[:100] if full_text else "(empty)",
                latency=f"{latency:.2f}s",
                audio_duration=f"{len(audio)/self._config.sample_rate:.2f}s",
            )
            return full_text if full_text else None
        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return None

    def is_available(self) -> bool:
        return self._available


class TTSEngine:
    """Text-to-speech using edge-tts with playback."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._available = False

    async def initialize(self) -> bool:
        try:
            import edge_tts
            self._available = True
            return True
        except ImportError:
            return False

    async def speak(self, text: str) -> bool:
        if not self._available or not text.strip():
            return False
        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                text,
                self._config.tts_voice,
                rate=self._config.tts_rate,
            )
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            if not audio_data:
                return False

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                import subprocess
                ps_cmd = f'''
                Add-Type -AssemblyName PresentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open([uri]::new("{temp_path}"))
                $player.Play()
                Start-Sleep -Seconds 4
                '''
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
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
    Production voice engine with proper state machine and echo protection.
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        command_callback: Optional[Callable[[str], Optional[str]]] = None,
        state_callback: Optional[Callable[[VoiceState], None]] = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._command_callback = command_callback
        self._state_callback = state_callback

        self._microphone = MicrophoneCapture(self._config)
        self._wake_detector = WakeWordDetector(self._config)
        self._command_recorder = CommandRecorder(self._config)
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)

        self._state = VoiceState.STANDBY
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def initialize(self) -> dict[str, bool]:
        stt_ok = await self._stt.initialize()
        tts_ok = await self._tts.initialize()
        mic_ok = self._microphone.start()
        return {"stt": stt_ok, "tts": tts_ok, "microphone": mic_ok}

    async def start(self) -> None:
        self._running = True
        self._set_state(VoiceState.STANDBY)
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        self._microphone.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_state(VoiceState.STANDBY)

    def _run_loop(self) -> None:
        while self._running:
            try:
                # Phase 1: Wait for wake word
                self._set_state(VoiceState.STANDBY)
                self._wake_detector.reset()
                self._microphone.flush()

                while self._running:
                    audio = self._microphone.get_audio(timeout=0.1)
                    if audio is None:
                        continue
                    energy = self._microphone.calculate_energy(audio)
                    if self._wake_detector.process_chunk(audio, energy):
                        break

                if not self._running:
                    break

                # Phase 2: Acknowledge and listen
                self._set_state(VoiceState.WAKE_DETECTED)

                # Speak acknowledgement WITH MIC MUTED (echo protection)
                self._microphone.mute()
                future = asyncio.run_coroutine_threadsafe(
                    self._tts.speak("Yes?"),
                    self._loop,
                )
                try:
                    future.result(timeout=5)
                except Exception:
                    pass
                # Wait a moment after TTS before unmuting
                time.sleep(0.3)
                self._microphone.unmute()

                # Phase 3: Record command
                self._set_state(VoiceState.LISTENING)
                self._command_recorder.start()
                self._microphone.flush()

                while self._running and not self._command_recorder.should_stop():
                    audio = self._microphone.get_audio(timeout=0.1)
                    if audio is not None:
                        energy = self._microphone.calculate_energy(audio)
                        self._command_recorder.add_chunk(audio, energy)

                speech_audio = self._command_recorder.stop()

                if speech_audio is None:
                    logger.info("no_command_detected")
                    continue

                # Phase 4: Transcribe
                self._set_state(VoiceState.TRANSCRIBING)
                text = asyncio.run_coroutine_threadsafe(
                    self._stt.transcribe(speech_audio),
                    self._loop,
                ).result(timeout=30)

                if not text:
                    logger.info("empty_transcription")
                    self._microphone.mute()
                    asyncio.run_coroutine_threadsafe(
                        self._tts.speak("Sorry, I didn't catch that."),
                        self._loop,
                    ).result(timeout=5)
                    time.sleep(0.3)
                    self._microphone.unmute()
                    continue

                logger.info("command_received", text=text)

                # Phase 5: Process command
                self._set_state(VoiceState.THINKING)
                if self._command_callback:
                    result = self._command_callback(text)
                    if result:
                        # Phase 6: Speak result
                        self._set_state(VoiceState.SPEAKING)
                        self._microphone.mute()
                        asyncio.run_coroutine_threadsafe(
                            self._tts.speak(result),
                            self._loop,
                        ).result(timeout=30)
                        time.sleep(0.3)
                        self._microphone.unmute()

            except Exception as e:
                logger.error("voice_loop_error", error=str(e))
                self._set_state(VoiceState.ERROR)
                time.sleep(2)

    def _set_state(self, state: VoiceState) -> None:
        self._state = state
        if self._state_callback:
            try:
                self._state_callback(state)
            except Exception:
                pass

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "state": self._state.value,
            "stt_available": self._stt.is_available(),
            "tts_available": self._tts.is_available(),
            "microphone_active": self._microphone.is_active,
            "microphone_muted": self._microphone.is_muted,
        }
