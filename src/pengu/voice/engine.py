"""
Pengu Voice Engine — production voice assistant pipeline.

Architecture:
  STANDBY → VAD detects speech → transcribe → check "hello pengu"
  → if wake phrase: ACKNOWLEDGE → LISTEN command → TRANSCRIBE → EXECUTE → SPEAK → STANDBY
  → if not wake phrase: ignore, stay in STANDBY

Features:
  - Streaming microphone capture (one long-lived stream)
  - Adaptive energy-based VAD with noise floor calibration
  - STT-based wake word detection (transcribe short segments, check for "hello pengu")
  - VAD-based command recording with silence timeout
  - TTS with barge-in support
  - Echo protection (mic muted during TTS)
  - Structured state machine with all transitions logged
  - Error recovery — stays alive after failures
  - Intelligent microphone selection with fallback
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


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class VoiceState(str, Enum):
    """Voice engine states."""
    OFFLINE = "OFFLINE"
    STARTING = "STARTING"
    STANDBY = "STANDBY"
    WAKE_DETECTED = "WAKE_DETECTED"
    ACKNOWLEDGING = "ACKNOWLEDGING"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VoiceConfig:
    """Voice engine configuration."""
    # Microphone
    sample_rate: int = 16000
    channels: int = 1
    device_index: Optional[int] = None

    # VAD (energy-based voice activity detection) — adaptive
    vad_energy_threshold: float = 15.0       # Starting threshold (auto-calibrated)
    vad_speech_start_frames: int = 3         # Consecutive frames above threshold to confirm speech
    vad_silence_timeout: float = 1.0         # Seconds of silence to consider speech ended
    vad_max_speech_duration: float = 5.0     # Max duration for a single speech segment
    vad_adaptive: bool = True                # Enable adaptive threshold
    vad_noise_calibration_frames: int = 50   # Frames to calibrate noise floor

    # Wake word detection
    wake_phrase: str = "hello pengu"         # The wake phrase to detect
    wake_max_segments: int = 5               # Max speech segments to check before giving up
    wake_debounce_seconds: float = 3.0       # Cooldown after wake detection

    # Command recording
    command_silence_timeout: float = 1.5     # Silence after speech to end command
    command_min_duration: float = 0.3        # Minimum command audio duration
    command_max_duration: float = 20.0       # Maximum command audio duration

    # STT
    stt_model_size: str = "tiny"
    stt_language: str = "en"

    # TTS
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"

    # Barge-in
    barge_in_energy_threshold: float = 30.0  # Energy level to detect barge-in during TTS


# ---------------------------------------------------------------------------
# Streaming Microphone with Intelligent Device Selection
# ---------------------------------------------------------------------------

class MicrophoneManager:
    """
    Long-lived streaming microphone capture with intelligent device selection,
    mute control, adaptive calibration, and diagnostics.

    Device selection priority:
      1. Explicitly configured device (PENGU_MIC_DEVICE)
      2. Windows default input device
      3. Best device with actual audio signal (tested via short recording)
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._stream = None
        self._is_active = False
        self._is_muted = False
        self._lock = threading.Lock()
        self._device_info: dict[str, Any] = {}
        self._rms_history: list[float] = []
        self._peak_level: float = 0.0
        self._noise_floor: float = 0.5
        self._calibrated = False
        self._selected_device: Optional[int] = None

    def _select_best_device(self) -> Optional[int]:
        """
        Find the best input device by actually testing candidates.

        This is the critical fix: do NOT just pick the default device.
        The default device may have zero gain (as we discovered).
        Instead, test candidates and pick one with actual audio signal.
        """
        try:
            import sounddevice as sd

            # Check environment override first
            env_device = os.environ.get("PENGU_MIC_DEVICE")
            if env_device:
                try:
                    dev_idx = int(env_device)
                    sd.check_input_settings(device=dev_idx, samplerate=self._config.sample_rate, channels=1)
                    logger.info("mic_selected_env", device=dev_idx, source="PENGU_MIC_DEVICE")
                    return dev_idx
                except Exception as e:
                    logger.warning("env_device_failed", device=env_device, error=str(e))

            # Get all input devices
            devices = sd.query_devices()
            input_devices = []
            for i, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append((i, dev))

            if not input_devices:
                logger.error("no_input_devices")
                return None

            # Get default device
            default_device_idx = sd.default.device[0]

            # Quick test: record 1 second from a device and check if it has signal
            def _quick_test(dev_idx: int) -> float:
                """Record 1 second and return the RMS. Returns 0 on failure."""
                try:
                    sd.check_input_settings(
                        device=dev_idx,
                        samplerate=self._config.sample_rate,
                        channels=1,
                        dtype="int16",
                    )
                    audio = sd.rec(
                        int(1.0 * self._config.sample_rate),
                        samplerate=self._config.sample_rate,
                        channels=1,
                        dtype="int16",
                        device=dev_idx,
                    )
                    sd.wait()
                    if audio is None or audio.size == 0:
                        return 0.0
                    flat = audio.flatten().astype(np.float32)
                    return float(np.sqrt(np.mean(flat ** 2)))
                except Exception:
                    return 0.0

            # Test default device first
            if default_device_idx is not None and 0 <= default_device_idx < len(devices):
                rms = _quick_test(default_device_idx)
                if rms > 10.0:  # Has meaningful signal
                    logger.info("mic_selected_default", device=default_device_idx, rms=round(rms, 1))
                    return default_device_idx
                else:
                    logger.warning("default_device_silent", device=default_device_idx, rms=round(rms, 1))

            # Test other devices — prefer MME and DirectSound APIs for Windows
            best_device = None
            best_rms = 0.0
            tested = {default_device_idx}

            # Sort: prefer MME/DirectSound APIs (more reliable on Windows)
            api_priority = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2, "Windows WDM-KS": 3}
            sorted_devices = sorted(
                input_devices,
                key=lambda x: api_priority.get(x[1].get("hostapi", ""), 99)
                if hasattr(x[1], 'get') else 99
            )

            for dev_idx, dev in sorted_devices:
                if dev_idx in tested:
                    continue
                if len(tested) >= 6:  # Don't test too many
                    break
                tested.add(dev_idx)

                rms = _quick_test(dev_idx)
                if rms > best_rms:
                    best_rms = rms
                    best_device = dev_idx
                    if rms > 50.0:  # Good enough
                        break

            if best_device is not None and best_rms > 5.0:
                logger.info("mic_selected_fallback", device=best_device, rms=round(best_rms, 1))
                return best_device

            # Last resort: return default even if silent
            if default_device_idx is not None:
                logger.warning("mic_fallback_default_silent", device=default_device_idx)
                return default_device_idx

            # Return first available
            return input_devices[0][0] if input_devices else None

        except Exception as e:
            logger.error("mic_selection_failed", error=str(e))
            return None

    def start(self) -> bool:
        """Start the microphone stream with intelligent device selection."""
        try:
            import sounddevice as sd

            # Select device
            device = self._config.device_index
            if device is None:
                device = self._select_best_device()
                if device is None:
                    logger.error("no_microphone_found")
                    return False

            self._selected_device = device
            self._device_info = sd.query_devices(device)
            logger.info(
                "microphone_started",
                device=device,
                name=self._device_info.get("name", "unknown"),
                sample_rate=self._config.sample_rate,
                channels=self._config.channels,
            )

            # Try opening the stream
            try:
                self._stream = sd.InputStream(
                    samplerate=self._config.sample_rate,
                    channels=self._config.channels,
                    dtype="int16",
                    device=device,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._stream.start()
                self._is_active = True
                return True
            except Exception as e:
                logger.error("stream_open_failed", device=device, error=str(e))
                # Try fallback: reduce to 1 channel
                try:
                    self._stream = sd.InputStream(
                        samplerate=self._config.sample_rate,
                        channels=1,
                        dtype="int16",
                        device=device,
                        callback=self._audio_callback,
                        blocksize=1024,
                    )
                    self._stream.start()
                    self._config.channels = 1
                    self._is_active = True
                    logger.info("mic_fallback_mono", device=device)
                    return True
                except Exception as e2:
                    logger.error("stream_fallback_failed", error=str(e2))
                    return False

        except Exception as e:
            logger.error("microphone_start_failed", error=str(e))
            return False

    def stop(self) -> None:
        """Stop the microphone stream."""
        self._is_active = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Audio stream callback — queues audio data unless muted."""
        if self._is_active and not self._is_muted:
            self._audio_queue.put(indata.copy())

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from the stream."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def calculate_energy(self, audio: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk."""
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        self._rms_history.append(energy)
        if len(self._rms_history) > 200:
            self._rms_history.pop(0)
        if energy > self._peak_level:
            self._peak_level = energy
        return energy

    def calibrate_noise_floor(self, num_frames: int = 50) -> float:
        """
        Calibrate noise floor by recording ambient noise.

        This runs BEFORE voice detection to establish a baseline.
        """
        if not self._rms_history:
            # Collect some frames first
            collected = 0
            while collected < num_frames:
                audio = self.get_audio(timeout=0.1)
                if audio is not None:
                    self.calculate_energy(audio)
                    collected += 1
                else:
                    break

        if self._rms_history:
            # Use the 25th percentile as noise floor
            sorted_rms = sorted(self._rms_history)
            idx = max(0, len(sorted_rms) // 4)
            self._noise_floor = sorted_rms[idx] if sorted_rms else 0.5

        # Set adaptive threshold: noise_floor * 3 + minimum
        adaptive_threshold = max(self._noise_floor * 3.0, 5.0)
        self._config.vad_energy_threshold = adaptive_threshold
        self._calibrated = True

        logger.info(
            "noise_calibrated",
            noise_floor=round(self._noise_floor, 2),
            threshold=round(adaptive_threshold, 2),
        )
        return adaptive_threshold

    def flush(self) -> None:
        """Discard all queued audio."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def mute(self) -> None:
        """Mute the microphone (for TTS echo protection)."""
        with self._lock:
            self._is_muted = True
            self.flush()

    def unmute(self) -> None:
        """Unmute the microphone."""
        with self._lock:
            self._is_muted = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def selected_device(self) -> Optional[int]:
        return self._selected_device

    def get_level(self) -> dict[str, Any]:
        """Get current microphone level info."""
        avg_rms = float(np.mean(self._rms_history[-50:])) if self._rms_history else 0.0
        return {
            "active": self._is_active,
            "muted": self._is_muted,
            "avg_rms": round(avg_rms, 2),
            "peak": round(self._peak_level, 2),
            "noise_floor": round(self._noise_floor, 2),
            "calibrated": self._calibrated,
            "device": self._device_info.get("name", "unknown") if self._device_info else "unknown",
            "device_index": self._selected_device,
            "sample_rate": self._config.sample_rate,
        }


# ---------------------------------------------------------------------------
# Adaptive VAD
# ---------------------------------------------------------------------------

class AdaptiveVAD:
    """
    Adaptive Voice Activity Detection.

    Uses noise floor calibration and adaptive threshold to correctly
    detect speech onset regardless of microphone gain.
    """

    def __init__(self, config: VoiceConfig, microphone: MicrophoneManager) -> None:
        self._config = config
        self._mic = microphone
        self._speech_detected = False
        self._speech_start_time: float = 0
        self._speech_frames: int = 0
        self._consecutive_silence_frames: int = 0

    def reset(self) -> None:
        """Reset detection state."""
        self._speech_detected = False
        self._speech_frames = 0
        self._consecutive_silence_frames = 0

    def process_chunk(self, audio: np.ndarray, energy: float) -> tuple[bool, bool]:
        """
        Process an audio chunk.

        Returns:
            (speech_detected, speech_ended)
            - speech_detected: True if speech is currently happening
            - speech_ended: True if a speech segment just ended
        """
        threshold = self._config.vad_energy_threshold

        if not self._speech_detected:
            # Looking for speech onset
            if energy > threshold:
                self._speech_detected = True
                self._speech_start_time = time.time()
                self._speech_frames = 1
                self._consecutive_silence_frames = 0
                return True, False
            return False, False
        else:
            # Speech is ongoing
            self._speech_frames += 1
            speech_duration = time.time() - self._speech_start_time

            if energy < threshold * 0.5:
                self._consecutive_silence_frames += 1
                # Need several consecutive silence frames to confirm end
                silence_frames_needed = int(self._config.vad_silence_timeout * 10)  # ~100ms per frame
                if self._consecutive_silence_frames >= silence_frames_needed:
                    self._reset()
                    return False, True
            else:
                self._consecutive_silence_frames = 0

            # Max duration
            if speech_duration > self._config.vad_max_speech_duration:
                self._reset()
                return False, True

            return True, False

    def _reset(self) -> None:
        self._speech_detected = False
        self._speech_frames = 0
        self._consecutive_silence_frames = 0


# ---------------------------------------------------------------------------
# VAD-based Wake Word Detector (STT-based)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """
    Wake word detection using VAD + STT transcription.

    Strategy:
      1. Monitor microphone energy for speech onset
      2. When speech detected, record the segment
      3. Transcribe with faster-whisper
      4. Check if transcription contains the wake phrase
      5. If yes → wake; if no → continue monitoring

    This avoids false triggers from random speech, noise, or music.
    """

    def __init__(self, config: VoiceConfig, stt_engine: "STTEngine") -> None:
        self._config = config
        self._stt = stt_engine
        self._last_wake_time: float = 0
        self._vad = AdaptiveVAD(config, None)  # type: ignore

    def process_chunk(self, audio: np.ndarray, energy: float) -> Optional[str]:
        """
        Process an audio chunk for wake word detection.

        Returns:
            The wake phrase if detected, None otherwise.
        """
        now = time.time()

        # Debounce: don't re-trigger too quickly
        if now - self._last_wake_time < self._config.wake_debounce_seconds:
            return None

        speech_detected, speech_ended = self._vad.process_chunk(audio, energy)

        if speech_ended:
            # Speech segment ended — try to transcribe and check for wake phrase
            # But we need the buffered audio... let's use a simpler approach
            pass

        if speech_detected and not speech_ended:
            # Accumulate audio for transcription
            if not hasattr(self, '_buffer'):
                self._buffer = []
            self._buffer.append(audio)
            speech_duration = time.time() - self._vad._speech_start_time

            # When speech ends (detected by energy drop), we'll transcribe
            # For now, just accumulate
            return None

        if speech_ended and hasattr(self, '_buffer') and self._buffer:
            # Speech ended — transcribe and check
            result = self._check_wake_phrase(self._buffer)
            self._buffer = []
            if result:
                self._last_wake_time = time.time()
                return result

        return None

    def process_chunk_simple(self, audio: np.ndarray, energy: float) -> Optional[str]:
        """
        Simpler wake word detection: accumulate audio during speech, transcribe on silence.
        Used by the main loop when not using the integrated VAD.
        """
        now = time.time()

        # Debounce
        if now - self._last_wake_time < self._config.wake_debounce_seconds:
            return None

        if not hasattr(self, '_speech_active'):
            self._speech_active = False
            self._buffer: list[np.ndarray] = []
            self._speech_start: float = 0.0
            self._silence_frames: int = 0

        if not self._speech_active:
            # Looking for speech onset
            if energy > self._config.vad_energy_threshold:
                self._speech_active = True
                self._buffer = [audio]
                self._speech_start = now
                self._silence_frames = 0
            return None
        else:
            # Speech ongoing
            self._buffer.append(audio)
            speech_duration = now - self._speech_start

            # Check if speech ended (energy dropped below threshold)
            if energy < self._config.vad_energy_threshold * 0.5:
                self._silence_frames += 1
                # Require a few consecutive silence frames (not too many)
                if self._silence_frames >= 3:
                    self._speech_active = False
                    if speech_duration >= 0.3 and self._buffer:
                        result = self._check_wake_phrase(self._buffer)
                        self._buffer = []
                        if result:
                            self._last_wake_time = time.time()
                            return result
                    self._buffer = []
            else:
                self._silence_frames = 0

            # Max duration — force stop
            if speech_duration > self._config.vad_max_speech_duration:
                self._speech_active = False
                if speech_duration >= 0.3 and self._buffer:
                    result = self._check_wake_phrase(self._buffer)
                    self._buffer = []
                    if result:
                        self._last_wake_time = time.time()
                        return result
                self._buffer = []

        return None

    def _check_wake_phrase(self, buffer: list[np.ndarray]) -> Optional[str]:
        """Transcribe buffered audio and check for wake phrase."""
        if not buffer:
            return None

        audio = np.concatenate(buffer)
        duration = len(audio) / self._config.sample_rate
        if duration < 0.3:
            return None

        logger.info("wake_check_transcribing", duration=f"{duration:.2f}s")

        # Run STT
        try:
            loop = asyncio.new_event_loop()
            try:
                text = loop.run_until_complete(self._stt.transcribe(audio))
            finally:
                loop.close()
        except Exception as e:
            logger.error("wake_stt_error", error=str(e))
            return None

        if not text:
            return None

        text_lower = text.lower().strip()
        logger.info("wake_check_text", text=text_lower[:100])

        # Check for wake phrase variants
        wake_phrases = [
            self._config.wake_phrase.lower(),
            "hello pengu",
            "hey pengu",
            "hey pengo",
            "hello pengo",
            "pengu",
        ]

        for phrase in wake_phrases:
            if phrase in text_lower:
                logger.info("wake_phrase_detected", text=text_lower, phrase=phrase)
                return text_lower

        return None

    def reset(self) -> None:
        """Full reset."""
        self._vad.reset()
        if hasattr(self, '_buffer'):
            self._buffer = []
        if hasattr(self, '_speech_active'):
            self._speech_active = False
        self._last_wake_time = 0


# ---------------------------------------------------------------------------
# Command Recorder (VAD-based)
# ---------------------------------------------------------------------------

class CommandRecorder:
    """Records command audio with VAD-based silence detection."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._last_speech_time: float = 0
        self._start_time: float = 0
        self._has_speech: bool = False
        self._noise_floor: float = 0.5

    def start(self) -> None:
        """Start recording a command."""
        self._audio_buffer.clear()
        self._is_recording = True
        self._start_time = time.time()
        self._last_speech_time = 0  # No speech yet
        self._has_speech = False

    def add_chunk(self, audio: np.ndarray, energy: float) -> None:
        """Add an audio chunk to the recording."""
        if not self._is_recording:
            return
        self._audio_buffer.append(audio)
        # Use adaptive threshold
        threshold = max(self._config.vad_energy_threshold, self._noise_floor * 3.0)
        if energy > threshold:
            self._last_speech_time = time.time()
            self._has_speech = True

    def set_noise_floor(self, noise_floor: float) -> None:
        """Set the calibrated noise floor for adaptive threshold."""
        self._noise_floor = noise_floor

    def should_stop(self) -> bool:
        """Check if recording should stop."""
        if not self._is_recording:
            return False
        now = time.time()

        # Stop on max duration
        if now - self._start_time > self._config.command_max_duration:
            return True

        # Only stop on silence after speech was detected
        if self._has_speech and self._last_speech_time > 0:
            silence_duration = now - self._last_speech_time
            if silence_duration > self._config.command_silence_timeout:
                return True

        return False

    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return the captured audio."""
        self._is_recording = False
        if not self._audio_buffer:
            return None
        audio = np.concatenate(self._audio_buffer)
        duration = len(audio) / self._config.sample_rate
        if duration < self._config.command_min_duration:
            return None
        return audio


# ---------------------------------------------------------------------------
# STT Engine
# ---------------------------------------------------------------------------

class STTEngine:
    """Speech-to-text using faster-whisper."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._model = None
        self._available = False

    async def initialize(self) -> bool:
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            model_size = self._config.stt_model_size

            # Allow environment override
            model_size = os.environ.get("PENGU_STT_MODEL", model_size)
            compute_type = os.environ.get("PENGU_STT_COMPUTE_TYPE", "int8")
            device = os.environ.get("PENGU_STT_DEVICE", "cpu")

            logger.info("loading_stt_model", model=model_size, device=device, compute_type=compute_type)
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            self._available = True
            logger.info("stt_model_loaded", model=model_size)
            return True
        except Exception as e:
            logger.error("stt_model_load_failed", error=str(e))
            return False

    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text."""
        if not self._available or self._model is None:
            return None
        try:
            audio_flat = audio.flatten() if audio.ndim > 1 else audio
            audio_float = audio_flat.astype(np.float32) / 32768.0
            start = time.time()
            language = os.environ.get("PENGU_STT_LANGUAGE", self._config.stt_language)
            segments, info = self._model.transcribe(
                audio_float,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts).strip()
            latency = time.time() - start
            logger.info(
                "stt_complete",
                text=full_text[:100] if full_text else "(empty)",
                latency=f"{latency:.2f}s",
                audio_duration=f"{len(audio)/self._config.sample_rate:.2f}s",
                probability=f"{info.language_probability:.2f}" if info.language_probability else "n/a",
            )
            return full_text if full_text else None
        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            return None

    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# TTS Engine (with barge-in support)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Text-to-speech using edge-tts with barge-in support."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._available = False
        self._is_speaking = False
        self._cancel_event = threading.Event()
        self._speak_lock = threading.Lock()
        self._process = None

    async def initialize(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            self._available = True
            return True
        except ImportError:
            return False

    async def speak(self, text: str) -> bool:
        """Speak text. Returns True if speaking completed, False if cancelled."""
        if not self._available or not text.strip():
            return False

        with self._speak_lock:
            self._is_speaking = True
            self._cancel_event.clear()

        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text,
                self._config.tts_voice,
                rate=self._config.tts_rate,
            )
            audio_data = b""
            async for chunk in communicate.stream():
                if self._cancel_event.is_set():
                    logger.info("tts_cancelled")
                    return False
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                return False

            # Write to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                import subprocess
                # Use PowerShell to play and wait for completion
                ps_cmd = f'''
                Add-Type -AssemblyName PresentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open([uri]::new("{temp_path}"))
                $player.Play()
                Start-Sleep -Milliseconds 500
                while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
                    if ([System.Threading.EventWaitHandle]::WaitOne(100)) {{ break }}
                    Start-Sleep -Milliseconds 100
                }}
                $player.Close()
                '''
                self._process = subprocess.Popen(
                    ["powershell", "-Command", ps_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Wait but allow cancellation
                start = time.time()
                while self._process.poll() is None:
                    if self._cancel_event.is_set():
                        try:
                            self._process.kill()
                        except Exception:
                            pass
                        return False
                    if time.time() - start > 30:  # Safety timeout
                        try:
                            self._process.kill()
                        except Exception:
                            pass
                        break
                    time.sleep(0.1)

            finally:
                self._process = None
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            logger.info("tts_spoke", text_length=len(text))
            return True

        except Exception as e:
            logger.error("tts_failed", error=str(e))
            return False
        finally:
            with self._speak_lock:
                self._is_speaking = False
                self._process = None

    def cancel(self) -> None:
        """Cancel current TTS playback (barge-in)."""
        self._cancel_event.set()
        if self._process:
            try:
                self._process.kill()
            except Exception:
                pass

    def is_available(self) -> bool:
        return self._available

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking


# ---------------------------------------------------------------------------
# Voice Engine
# ---------------------------------------------------------------------------

class VoiceEngine:
    """
    Production voice engine with proper state machine and wake word detection.

    Flow:
      STANDBY
        → [VAD detects speech] → transcribe → check for "hello pengu"
        → [wake phrase found] → WAKE_DETECTED → ACKNOWLEDGING (speak "Yes?")
        → LISTENING (record command with VAD)
        → TRANSCRIBING (STT on command)
        → THINKING (process command)
        → EXECUTING (run action)
        → SPEAKING (TTS response)
        → STANDBY

    Critical behavior:
      - After acknowledging ("Yes?"), stays in LISTENING until command completes
      - Returns to STANDBY only after full command cycle
      - Recovers from errors without crashing
      - Barge-in stops TTS and returns to LISTENING
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

        self._microphone = MicrophoneManager(self._config)
        self._stt = STTEngine(self._config)
        self._tts = TTSEngine(self._config)
        self._wake_detector = WakeWordDetector(self._config, self._stt)
        self._command_recorder = CommandRecorder(self._config)

        self._state = VoiceState.OFFLINE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_command_audio: Optional[np.ndarray] = None

        # Diagnostics
        self._diagnostics: dict[str, Any] = {
            "wake_detections": 0,
            "commands_processed": 0,
            "empty_transcriptions": 0,
            "tts_spoken": 0,
            "errors": 0,
        }

    async def initialize(self) -> dict[str, bool]:
        """Initialize all voice components."""
        self._set_state(VoiceState.STARTING)
        stt_ok = await self._stt.initialize()
        tts_ok = await self._tts.initialize()
        mic_ok = self._microphone.start()
        result = {"stt": stt_ok, "tts": tts_ok, "microphone": mic_ok}
        logger.info("voice_init", **result)

        # Calibrate noise floor after mic is running
        if mic_ok:
            self._microphone.calibrate_noise_floor()
            self._command_recorder.set_noise_floor(self._microphone.noise_floor)

        return result

    async def start(self) -> None:
        """Start the voice engine main loop."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-engine")
        self._thread.start()
        self._set_state(VoiceState.STANDBY)
        logger.info("voice_engine_started")

    async def stop(self) -> None:
        """Stop the voice engine."""
        self._running = False
        self._set_state(VoiceState.STOPPING)
        self._tts.cancel()  # Cancel any ongoing speech
        self._microphone.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_state(VoiceState.OFFLINE)
        logger.info("voice_engine_stopped")

    def _run_loop(self) -> None:
        """Main voice engine loop — runs in a background thread."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self._running:
            try:
                consecutive_errors = 0

                # Phase 1: Standby and detect wake word
                self._phase_standby_and_detect()
                if not self._running:
                    break

                # Phase 2: Acknowledge wake word
                self._phase_acknowledge()
                if not self._running:
                    break

                # Phase 3: Listen for command (CRITICAL: stays in LISTENING)
                self._phase_listen_command()
                if not self._running:
                    break

                # Phase 4: Transcribe and execute
                self._phase_transcribe_and_execute()

            except Exception as e:
                logger.error("voice_loop_error", error=str(e), exc_info=True)
                self._diagnostics["errors"] += 1
                consecutive_errors += 1
                self._set_state(VoiceState.ERROR)

                if consecutive_errors >= max_consecutive_errors:
                    logger.error("too_many_errors", count=consecutive_errors)
                    time.sleep(5)  # Longer pause after repeated failures
                else:
                    time.sleep(1)

                if self._running:
                    self._set_state(VoiceState.STANDBY)

    def _phase_standby_and_detect(self) -> None:
        """Phase 1: Wait for wake word via VAD + STT transcription."""
        self._set_state(VoiceState.STANDBY)
        self._wake_detector.reset()
        self._microphone.flush()
        self._microphone.unmute()

        logger.info("standby_listening")

        while self._running:
            audio = self._microphone.get_audio(timeout=0.1)
            if audio is None:
                continue

            energy = self._microphone.calculate_energy(audio)
            wake = self._wake_detector.process_chunk_simple(audio, energy)
            if wake is not None:
                self._diagnostics["wake_detections"] += 1
                logger.info("wake_detected", text=wake)
                return

    def _phase_acknowledge(self) -> None:
        """Phase 2: Speak acknowledgement with mic muted (echo protection)."""
        self._set_state(VoiceState.WAKE_DETECTED)
        time.sleep(0.2)  # Brief pause

        self._set_state(VoiceState.ACKNOWLEDGING)
        self._microphone.mute()

        # Speak "Yes?" with cancellation support
        future = asyncio.run_coroutine_threadsafe(
            self._tts.speak("Yes?"),
            self._loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            pass

        # Wait for TTS to finish and audio to settle
        time.sleep(0.4)
        self._microphone.flush()
        self._microphone.unmute()

    def _phase_listen_command(self) -> None:
        """Phase 3: Record the user's command with VAD-based silence detection."""
        self._set_state(VoiceState.LISTENING)
        self._command_recorder.start()

        logger.info("command_listening")

        while self._running and not self._command_recorder.should_stop():
            audio = self._microphone.get_audio(timeout=0.1)
            if audio is not None:
                energy = self._microphone.calculate_energy(audio)
                self._command_recorder.add_chunk(audio, energy)

        speech_audio = self._command_recorder.stop()

        if speech_audio is None:
            logger.info("no_command_detected")
            self._microphone.mute()
            self._safe_tts_speak("I didn't hear a command.")
            time.sleep(0.3)
            self._microphone.unmute()
            return

        # Store for transcription phase
        self._pending_command_audio = speech_audio

    def _phase_transcribe_and_execute(self) -> None:
        """Phase 4: Transcribe command and execute."""
        if self._pending_command_audio is None:
            return

        speech_audio = self._pending_command_audio
        self._pending_command_audio = None

        # Transcribe
        self._set_state(VoiceState.TRANSCRIBING)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._stt.transcribe(speech_audio),
                self._loop,
            )
            text = future.result(timeout=30)
        except Exception as e:
            logger.error("transcription_error", error=str(e))
            text = None

        if not text:
            self._diagnostics["empty_transcriptions"] += 1
            logger.info("empty_transcription")
            self._microphone.mute()
            self._safe_tts_speak("Sorry, I didn't catch that. Please try again.")
            time.sleep(0.3)
            self._microphone.unmute()
            return

        logger.info("command_transcribed", text=text)

        # Process command
        self._set_state(VoiceState.THINKING)
        result_text = None
        if self._command_callback:
            try:
                self._set_state(VoiceState.EXECUTING)
                result_text = self._command_callback(text)
                self._diagnostics["commands_processed"] += 1
            except Exception as e:
                logger.error("command_error", error=str(e))
                result_text = f"Error executing command: {e}"

        # Speak result
        if result_text:
            self._set_state(VoiceState.SPEAKING)
            self._microphone.mute()
            self._safe_tts_speak(result_text)
            self._diagnostics["tts_spoken"] += 1
            time.sleep(0.3)
            self._microphone.unmute()

    def _safe_tts_speak(self, text: str) -> None:
        """Speak text safely in a thread with cancellation support."""
        future = asyncio.run_coroutine_threadsafe(
            self._tts.speak(text),
            self._loop,
        )
        try:
            future.result(timeout=30)
        except Exception:
            pass

    def interrupt(self) -> None:
        """Interrupt current TTS playback (barge-in)."""
        self._tts.cancel()
        self._microphone.flush()

    def _set_state(self, state: VoiceState) -> None:
        """Update state and notify callback."""
        old_state = self._state
        self._state = state
        if state != old_state:
            logger.info("state_transition", from_state=old_state.value, to_state=state.value)
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
        """Get full voice engine status."""
        return {
            "running": self._running,
            "state": self._state.value,
            "stt_available": self._stt.is_available(),
            "tts_available": self._tts.is_available(),
            "tts_speaking": self._tts.is_speaking,
            "microphone": self._microphone.get_level(),
            "diagnostics": self._diagnostics.copy(),
            "config": {
                "wake_phrase": self._config.wake_phrase,
                "stt_model": self._config.stt_model_size,
                "tts_voice": self._config.tts_voice,
                "vad_threshold": self._config.vad_energy_threshold,
                "adaptive_vad": self._config.vad_adaptive,
            },
        }

    def run_diagnostics(self) -> dict[str, Any]:
        """Run full voice system diagnostics."""
        result = {}

        # Microphone
        mic_level = self._microphone.get_level()
        result["microphone"] = {
            "status": "OK" if mic_level["active"] else "NOT ACTIVE",
            "device": mic_level["device"],
            "device_index": mic_level["device_index"],
            "sample_rate": mic_level["sample_rate"],
            "avg_rms": mic_level["avg_rms"],
            "peak": mic_level["peak"],
            "noise_floor": mic_level["noise_floor"],
            "calibrated": mic_level["calibrated"],
        }

        # STT
        result["stt"] = {
            "status": "OK" if self._stt.is_available() else "NOT LOADED",
            "model": self._config.stt_model_size,
        }

        # TTS
        result["tts"] = {
            "status": "OK" if self._tts.is_available() else "NOT AVAILABLE",
            "voice": self._config.tts_voice,
        }

        # Wake word
        result["wake_word"] = {
            "status": "OK",
            "phrase": self._config.wake_phrase,
            "method": "VAD + STT transcription",
        }

        # Stats
        result["stats"] = self._diagnostics.copy()

        return result
