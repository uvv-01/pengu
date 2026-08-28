"""
Pengu Voice Engine — production voice assistant pipeline.

Architecture:
  STANDBY → VAD detects speech → transcribe → check "hello pengu"
  → if wake phrase: ACKNOWLEDGE → LISTEN command → TRANSCRIBE → EXECUTE → SPEAK → STANDBY
  → if not wake phrase: ignore, stay in STANDBY

Features:
  - Streaming microphone capture (one long-lived stream)
  - Quality-based device selection (tests ALL candidates, picks strongest signal)
  - Channel downmix (multi-channel → mono)
  - Resample to 16kHz for STT
  - Adaptive energy-based VAD with noise floor calibration
  - STT-based wake word detection
  - VAD-based command recording with silence timeout
  - TTS with barge-in support
  - Echo protection (mic muted during TTS)
  - Audio quality gate before entering voice mode
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

# Target audio format for STT / VAD / wake word
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MIN_SIGNAL_RMS = 50.0  # Minimum RMS for a device to be considered usable


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
    sample_rate: int = TARGET_SAMPLE_RATE
    channels: int = TARGET_CHANNELS
    device_index: Optional[int] = None

    # VAD (energy-based voice activity detection) — adaptive
    vad_energy_threshold: float = 15.0       # Starting threshold (auto-calibrated)
    vad_speech_start_frames: int = 3
    vad_silence_timeout: float = 1.0
    vad_max_speech_duration: float = 5.0
    vad_adaptive: bool = True
    vad_noise_calibration_frames: int = 50

    # Wake word detection
    wake_phrase: str = "hello pengu"
    wake_max_segments: int = 5
    wake_debounce_seconds: float = 3.0

    # Command recording
    command_silence_timeout: float = 1.5
    command_min_duration: float = 0.3
    command_max_duration: float = 20.0

    # STT
    stt_model_size: str = "tiny"
    stt_language: str = "en"

    # TTS
    tts_voice: str = "en-US-GuyNeural"
    tts_rate: str = "+0%"

    # Barge-in
    barge_in_energy_threshold: float = 30.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Downmix multi-channel audio to mono by averaging all channels.
    Input shape: (num_samples, num_channels) or (num_samples,)
    Output shape: (num_samples,)
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio.astype(np.float32), axis=1).astype(audio.dtype)
    return audio.flatten()


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio from orig_rate to target_rate.
    Uses linear interpolation (good enough for voice).
    """
    if orig_rate == target_rate:
        return audio
    duration = len(audio) / orig_rate
    num_samples = int(duration * target_rate)
    x_orig = np.linspace(0, duration, len(audio), endpoint=False)
    x_target = np.linspace(0, duration, num_samples, endpoint=False)
    return np.interp(x_target, x_orig, audio.astype(np.float64)).astype(np.int16)


def _measure_rms(audio: np.ndarray) -> float:
    """Measure RMS of audio (int16 or float32)."""
    flat = audio.flatten().astype(np.float32)
    return float(np.sqrt(np.mean(flat ** 2)))


def _measure_peak(audio: np.ndarray) -> float:
    """Measure peak amplitude."""
    flat = audio.flatten().astype(np.float32)
    return float(np.max(np.abs(flat)))


# ---------------------------------------------------------------------------
# Device Probe — tests a device with a short real capture
# ---------------------------------------------------------------------------

@dataclass
class DeviceProbe:
    """Result of probing a device with a short capture."""
    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float
    # Test results
    can_open: bool = False
    rms_mono_16k: float = 0.0
    peak_mono_16k: float = 0.0
    tested_channels: int = 0
    tested_sample_rate: int = 0
    error: str = ""
    # Scoring
    score: float = 0.0


def _probe_device(
    dev_idx: int,
    dev_info: dict,
    host_api_name: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    capture_duration: float = 2.0,
) -> DeviceProbe:
    """
    Probe a device: try to open at 16kHz mono, record, measure RMS.

    If 16kHz mono fails, try 1 channel at native sample rate and note it.
    """
    probe = DeviceProbe(
        device_index=dev_idx,
        device_name=dev_info.get("name", f"Device {dev_idx}"),
        host_api=host_api_name,
        max_channels=dev_info.get("max_input_channels", 0),
        native_sample_rate=dev_info.get("default_samplerate", 0),
    )

    try:
        import sounddevice as sd
    except ImportError:
        probe.error = "sounddevice not installed"
        return probe

    # Try 16kHz mono first
    try:
        sd.check_input_settings(device=dev_idx, samplerate=target_sr, channels=1, dtype="int16")
        audio = sd.rec(
            int(capture_duration * target_sr),
            samplerate=target_sr,
            channels=1,
            dtype="int16",
            device=dev_idx,
        )
        sd.wait()
        if audio is not None and audio.size > 0:
            probe.can_open = True
            probe.rms_mono_16k = _measure_rms(audio)
            probe.peak_mono_16k = _measure_peak(audio)
            probe.tested_channels = 1
            probe.tested_sample_rate = target_sr
            return probe
    except Exception:
        pass

    # Fallback: try native sample rate, 1 channel
    native_sr = int(dev_info.get("default_samplerate", target_sr))
    try:
        sd.check_input_settings(device=dev_idx, samplerate=native_sr, channels=1, dtype="int16")
        audio = sd.rec(
            int(capture_duration * native_sr),
            samplerate=native_sr,
            channels=1,
            dtype="int16",
            device=dev_idx,
        )
        sd.wait()
        if audio is not None and audio.size > 0:
            probe.can_open = True
            probe.rms_mono_16k = _measure_rms(audio)
            probe.peak_mono_16k = _measure_peak(audio)
            probe.tested_channels = 1
            probe.tested_sample_rate = native_sr
            return probe
    except Exception:
        pass

    probe.error = "Failed to open at 16kHz or native rate"
    return probe


def _is_real_microphone(name: str) -> bool:
    """
    Determine if a device name refers to a real microphone input.

    Returns False for system audio capture (Stereo Mix, PC Speaker),
    mapper endpoints (Microsoft Sound Mapper), and output devices.
    """
    name_lower = name.lower()

    # System audio capture — NOT a microphone
    non_mic_patterns = [
        "stereo mix",
        "pc speaker",
        "sound mapper",
        "primary sound capture",
    ]
    for pattern in non_mic_patterns:
        if pattern in name_lower:
            return False

    return True


def _score_device(probe: DeviceProbe) -> float:
    """
    Score a device probe. Higher is better.

    Scoring factors:
      - Measured RMS (primary signal quality metric)
      - Penalty for mapper/abstraction endpoints
      - Penalty for non-microphone devices (Stereo Mix, PC Speaker)
      - Bonus for native 16kHz support
      - Bonus for DirectSound / WDM-KS API
    """
    if not probe.can_open or probe.rms_mono_16k < 1.0:
        return 0.0

    # CRITICAL: Exclude non-microphone endpoints entirely
    if not _is_real_microphone(probe.device_name):
        return 0.0

    score = probe.rms_mono_16k

    # Bonus for native 16kHz (avoids resampling)
    if probe.tested_sample_rate == TARGET_SAMPLE_RATE:
        score *= 1.2

    # Bonus for DirectSound (good reliability on Windows)
    if "DirectSound" in probe.host_api:
        score *= 1.1

    # Bonus for WDM-KS (lowest latency, often best quality)
    if "WDM-KS" in probe.host_api:
        score *= 1.15

    # Penalty for MME (higher latency, often worse quality)
    if probe.host_api == "MME":
        score *= 0.8

    return score


# ---------------------------------------------------------------------------
# Streaming Microphone with Quality-Based Device Selection
# ---------------------------------------------------------------------------

class MicrophoneManager:
    """
    Long-lived streaming microphone capture with quality-based device selection,
    channel downmix, resampling, and diagnostics.

    Selection algorithm:
      1. Enumerate all input devices with host API info
      2. Probe each device with a short real capture at 16kHz mono
      3. Score by measured RMS (strongest clean signal wins)
      4. Penalize mapper/abstraction endpoints
      5. Prefer native 16kHz
      6. If explicit PENGU_MIC_DEVICE is set, use that (after validation)
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
        self._selected_probe: Optional[DeviceProbe] = None
        self._device_native_sr: int = TARGET_SAMPLE_RATE
        self._device_native_channels: int = 1
        self._all_probes: list[DeviceProbe] = []

    def _select_best_device(self) -> Optional[int]:
        """
        Find the best input device by probing ALL candidates and ranking by
        measured audio quality.

        Algorithm:
          1. Check PENGU_MIC_DEVICE env override
          2. Enumerate all input devices with host API info
          3. Probe each: 2-second capture at 16kHz mono
          4. Score by RMS (strongest clean signal wins)
          5. Penalize mapper endpoints
          6. Prefer native 16kHz / DirectSound / WDM-KS
          7. Return the device with highest score
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice_not_installed")
            return None

        # 1. Environment override
        env_device = os.environ.get("PENGU_MIC_DEVICE")
        if env_device:
            try:
                dev_idx = int(env_device)
                sd.check_input_settings(device=dev_idx, samplerate=TARGET_SAMPLE_RATE, channels=1)
                logger.info("mic_selected_env", device=dev_idx)
                return dev_idx
            except Exception as e:
                logger.warning("env_device_failed", device=env_device, error=str(e))

        # 2. Enumerate all input devices with host API names
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        input_devices = []
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                api_idx = dev.get("hostapi", 0)
                api_name = host_apis[api_idx]["name"] if 0 <= api_idx < len(host_apis) else "Unknown"
                input_devices.append((i, dev, api_name))

        if not input_devices:
            logger.error("no_input_devices")
            return None

        logger.info(
            "mic_enumerated",
            count=len(input_devices),
            devices=[(i, d.get("name", "?"), api) for i, d, api in input_devices],
        )

        # 3. Probe each device with a short real capture
        probes: list[DeviceProbe] = []
        for dev_idx, dev_info, api_name in input_devices:
            probe = _probe_device(dev_idx, dev_info, api_name, target_sr=TARGET_SAMPLE_RATE, capture_duration=2.0)
            probe.score = _score_device(probe)
            probes.append(probe)
            if probe.can_open:
                logger.info(
                    "mic_probed",
                    device=dev_idx,
                    name=probe.device_name,
                    api=probe.host_api,
                    rms=round(probe.rms_mono_16k, 1),
                    score=round(probe.score, 1),
                    sr=probe.tested_sample_rate,
                )
            else:
                logger.debug("mic_probe_failed", device=dev_idx, name=probe.device_name, error=probe.error)

        self._all_probes = probes

        # 4. Filter to devices that can open and have meaningful signal
        usable = [p for p in probes if p.can_open and p.score > 0]

        if not usable:
            logger.error("no_usable_microphone")
            # Last resort: return first device that can open at all
            any_open = [p for p in probes if p.can_open]
            if any_open:
                return any_open[0].device_index
            return None

        # 5. Pick the best device by score
        usable.sort(key=lambda p: p.score, reverse=True)
        best = usable[0]

        # Store probe info for quality gate and streaming config
        self._selected_probe = best
        self._device_native_sr = best.tested_sample_rate
        self._device_native_channels = 1  # We captured mono

        logger.info(
            "mic_selected",
            device=best.device_index,
            name=best.device_name,
            api=best.host_api,
            rms=round(best.rms_mono_16k, 1),
            score=round(best.score, 1),
            sr=best.tested_sample_rate,
            alternatives=[(p.device_index, p.device_name, round(p.score, 1)) for p in usable[1:5]],
        )

        return best.device_index

    def check_audio_quality(self) -> tuple[bool, str]:
        """
        Check if the selected microphone produces acceptable audio.
        Called before entering STANDBY to prevent operating with a broken mic.

        Returns:
            (is_good, message)
        """
        if not self._is_active:
            return False, "Microphone is not active."

        if self._selected_probe is None:
            return False, "No device probe data available."

        probe = self._selected_probe

        # Print the mic check report
        report_lines = [
            "MIC CHECK",
            f"  Device:       {probe.device_name}",
            f"  API:          {probe.host_api}",
            f"  Sample Rate:  {probe.tested_sample_rate} Hz",
            f"  Channels:     {probe.tested_channels}",
            f"  RMS:          {probe.rms_mono_16k:.1f}",
            f"  Peak:         {probe.peak_mono_16k:.1f}",
            f"  Score:        {probe.score:.1f}",
        ]

        # Check quality thresholds
        if probe.rms_mono_16k < MIN_SIGNAL_RMS:
            # Find the best alternative
            best_alt = None
            for p in self._all_probes:
                if p.can_open and p.device_index != probe.device_index and p.rms_mono_16k > MIN_SIGNAL_RMS:
                    if best_alt is None or p.score > best_alt.score:
                        best_alt = p

            status_msg = "MICROPHONE QUALITY TOO LOW"
            detail = f"Selected device is producing insufficient signal (RMS={probe.rms_mono_16k:.1f}, need>{MIN_SIGNAL_RMS:.0f})."
            if best_alt:
                detail += f"\nRecommended device: {best_alt.device_name} (Device {best_alt.device_index}, RMS={best_alt.rms_mono_16k:.1f}, API={best_alt.host_api})"
            else:
                detail += "\nNo better microphone found on this system."

            report_lines.append(f"  Status:       {status_msg}")
            report_lines.append(f"  {detail}")

            full_report = "\n".join(report_lines)
            logger.warning("mic_quality_low", report=full_report)
            return False, full_report

        report_lines.append("  Status:       GOOD")
        full_report = "\n".join(report_lines)
        logger.info("mic_quality_ok", report=full_report)
        return True, full_report

    def start(self) -> bool:
        """Start the microphone stream with quality-based device selection."""
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

            # Determine actual stream parameters
            # Always try to capture at 16kHz mono first (what STT needs)
            stream_sr = TARGET_SAMPLE_RATE
            stream_ch = TARGET_CHANNELS

            # If the device doesn't support 16kHz, use native rate and resample
            if self._device_native_sr != TARGET_SAMPLE_RATE:
                stream_sr = self._device_native_sr
                logger.info("mic_using_native_sr", device=device, sr=stream_sr, will_resample=True)

            logger.info(
                "microphone_opening",
                device=device,
                name=self._device_info.get("name", "unknown"),
                stream_sr=stream_sr,
                stream_ch=stream_ch,
                target_sr=TARGET_SAMPLE_RATE,
            )

            # Open the stream
            try:
                sd.check_input_settings(device=device, samplerate=stream_sr, channels=stream_ch, dtype="int16")
                self._stream = sd.InputStream(
                    samplerate=stream_sr,
                    channels=stream_ch,
                    dtype="int16",
                    device=device,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._stream.start()
                self._is_active = True
                self._device_native_sr = stream_sr
                self._device_native_channels = stream_ch
                logger.info("microphone_started", device=device, sr=stream_sr, ch=stream_ch)
                return True
            except Exception as e:
                logger.error("stream_open_failed_16k", device=device, error=str(e))

            # Fallback: try native sample rate
            native_sr = int(self._device_info.get("default_samplerate", 44100))
            try:
                sd.check_input_settings(device=device, samplerate=native_sr, channels=1, dtype="int16")
                self._stream = sd.InputStream(
                    samplerate=native_sr,
                    channels=1,
                    dtype="int16",
                    device=device,
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self._stream.start()
                self._is_active = True
                self._device_native_sr = native_sr
                self._device_native_channels = 1
                logger.info("microphone_started_fallback", device=device, sr=native_sr)
                return True
            except Exception as e2:
                logger.error("stream_open_failed_native", device=device, error=str(e2))

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
        """
        Audio stream callback — processes and queues audio data.

        Pipeline:
          1. Copy raw audio
          2. Downmix to mono (if multi-channel)
          3. Resample to 16kHz (if needed)
          4. Queue for processing
        """
        if not self._is_active or self._is_muted:
            return

        audio = indata.copy()

        # Step 1: Downmix to mono
        audio = _downmix_to_mono(audio)

        # Step 2: Resample to 16kHz if needed
        if self._device_native_sr != TARGET_SAMPLE_RATE:
            audio = _resample(audio, self._device_native_sr, TARGET_SAMPLE_RATE)

        self._audio_queue.put(audio)

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from the stream (always 16kHz mono int16)."""
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
        """Calibrate noise floor by recording ambient noise."""
        if not self._rms_history:
            collected = 0
            while collected < num_frames:
                audio = self.get_audio(timeout=0.1)
                if audio is not None:
                    self.calculate_energy(audio)
                    collected += 1
                else:
                    break

        if self._rms_history:
            sorted_rms = sorted(self._rms_history)
            idx = max(0, len(sorted_rms) // 4)
            self._noise_floor = sorted_rms[idx] if sorted_rms else 0.5

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
        probe = self._selected_probe
        return {
            "active": self._is_active,
            "muted": self._is_muted,
            "avg_rms": round(avg_rms, 2),
            "peak": round(self._peak_level, 2),
            "noise_floor": round(self._noise_floor, 2),
            "calibrated": self._calibrated,
            "device": self._device_info.get("name", "unknown") if self._device_info else "unknown",
            "device_index": self._selected_device,
            "host_api": probe.host_api if probe else "unknown",
            "probe_rms": round(probe.rms_mono_16k, 1) if probe else 0,
            "probe_score": round(probe.score, 1) if probe else 0,
            "sample_rate": self._device_native_sr,
            "target_sample_rate": TARGET_SAMPLE_RATE,
        }


# ---------------------------------------------------------------------------
# Adaptive VAD
# ---------------------------------------------------------------------------

class AdaptiveVAD:
    """Adaptive Voice Activity Detection."""

    def __init__(self, config: VoiceConfig, microphone: MicrophoneManager) -> None:
        self._config = config
        self._mic = microphone
        self._speech_detected = False
        self._speech_start_time: float = 0
        self._speech_frames: int = 0
        self._consecutive_silence_frames: int = 0

    def reset(self) -> None:
        self._speech_detected = False
        self._speech_frames = 0
        self._consecutive_silence_frames = 0

    def process_chunk(self, audio: np.ndarray, energy: float) -> tuple[bool, bool]:
        threshold = self._config.vad_energy_threshold

        if not self._speech_detected:
            if energy > threshold:
                self._speech_detected = True
                self._speech_start_time = time.time()
                self._speech_frames = 1
                self._consecutive_silence_frames = 0
                return True, False
            return False, False
        else:
            self._speech_frames += 1
            speech_duration = time.time() - self._speech_start_time

            if energy < threshold * 0.5:
                self._consecutive_silence_frames += 1
                silence_frames_needed = int(self._config.vad_silence_timeout * 10)
                if self._consecutive_silence_frames >= silence_frames_needed:
                    self._reset()
                    return False, True
            else:
                self._consecutive_silence_frames = 0

            if speech_duration > self._config.vad_max_speech_duration:
                self._reset()
                return False, True

            return True, False

    def _reset(self) -> None:
        self._speech_detected = False
        self._speech_frames = 0
        self._consecutive_silence_frames = 0


# ---------------------------------------------------------------------------
# Wake Word Detector (STT-based)
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Wake word detection using VAD + STT transcription."""

    def __init__(self, config: VoiceConfig, stt_engine: "STTEngine") -> None:
        self._config = config
        self._stt = stt_engine
        self._last_wake_time: float = 0
        self._vad = AdaptiveVAD(config, None)  # type: ignore

    def process_chunk_simple(self, audio: np.ndarray, energy: float) -> Optional[str]:
        """Accumulate audio during speech, transcribe on silence."""
        now = time.time()

        if now - self._last_wake_time < self._config.wake_debounce_seconds:
            return None

        if not hasattr(self, '_speech_active'):
            self._speech_active = False
            self._buffer: list[np.ndarray] = []
            self._speech_start: float = 0.0
            self._silence_frames: int = 0

        if not self._speech_active:
            if energy > self._config.vad_energy_threshold:
                self._speech_active = True
                self._buffer = [audio]
                self._speech_start = now
                self._silence_frames = 0
            return None
        else:
            self._buffer.append(audio)
            speech_duration = now - self._speech_start

            if energy < self._config.vad_energy_threshold * 0.5:
                self._silence_frames += 1
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
        if not buffer:
            return None

        audio = np.concatenate(buffer)
        duration = len(audio) / self._config.sample_rate
        if duration < 0.3:
            return None

        logger.info("wake_check_transcribing", duration=f"{duration:.2f}s")

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
        self._vad.reset()
        if hasattr(self, '_buffer'):
            self._buffer = []
        if hasattr(self, '_speech_active'):
            self._speech_active = False
        self._last_wake_time = 0


# ---------------------------------------------------------------------------
# Command Recorder
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
        self._audio_buffer.clear()
        self._is_recording = True
        self._start_time = time.time()
        self._last_speech_time = 0
        self._has_speech = False

    def add_chunk(self, audio: np.ndarray, energy: float) -> None:
        if not self._is_recording:
            return
        self._audio_buffer.append(audio)
        threshold = max(self._config.vad_energy_threshold, self._noise_floor * 3.0)
        if energy > threshold:
            self._last_speech_time = time.time()
            self._has_speech = True

    def set_noise_floor(self, noise_floor: float) -> None:
        self._noise_floor = noise_floor

    def should_stop(self) -> bool:
        if not self._is_recording:
            return False
        now = time.time()
        if now - self._start_time > self._config.command_max_duration:
            return True
        if self._has_speech and self._last_speech_time > 0:
            silence_duration = now - self._last_speech_time
            if silence_duration > self._config.command_silence_timeout:
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
        try:
            from faster_whisper import WhisperModel
            model_size = os.environ.get("PENGU_STT_MODEL", self._config.stt_model_size)
            compute_type = os.environ.get("PENGU_STT_COMPUTE_TYPE", "int8")
            device = os.environ.get("PENGU_STT_DEVICE", "cpu")
            logger.info("loading_stt_model", model=model_size, device=device, compute_type=compute_type)
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._available = True
            logger.info("stt_model_loaded", model=model_size)
            return True
        except Exception as e:
            logger.error("stt_model_load_failed", error=str(e))
            return False

    async def transcribe(self, audio: np.ndarray) -> Optional[str]:
        if not self._available or self._model is None:
            return None
        try:
            audio_flat = audio.flatten() if audio.ndim > 1 else audio
            audio_float = audio_flat.astype(np.float32) / 32768.0
            start = time.time()
            language = os.environ.get("PENGU_STT_LANGUAGE", self._config.stt_language)
            segments, info = self._model.transcribe(audio_float, language=language, beam_size=5, vad_filter=True)
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


# ---------------------------------------------------------------------------
# TTS Engine
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
        if not self._available or not text.strip():
            return False

        with self._speak_lock:
            self._is_speaking = True
            self._cancel_event.clear()

        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self._config.tts_voice, rate=self._config.tts_rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if self._cancel_event.is_set():
                    return False
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
                start = time.time()
                while self._process.poll() is None:
                    if self._cancel_event.is_set():
                        try:
                            self._process.kill()
                        except Exception:
                            pass
                        return False
                    if time.time() - start > 30:
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
    Production voice engine with quality-based mic selection, adaptive VAD,
    and proper state machine flow.

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

    Audio pipeline:
      MIC → downmix to mono → resample to 16kHz → VAD → STT/Wake
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
        self._mic_quality_ok = False

        self._diagnostics: dict[str, Any] = {
            "wake_detections": 0,
            "commands_processed": 0,
            "empty_transcriptions": 0,
            "tts_spoken": 0,
            "errors": 0,
        }

    async def initialize(self) -> dict[str, bool]:
        """Initialize all voice components and check mic quality."""
        self._set_state(VoiceState.STARTING)
        stt_ok = await self._stt.initialize()
        tts_ok = await self._tts.initialize()
        mic_ok = self._microphone.start()

        result = {"stt": stt_ok, "tts": tts_ok, "microphone": mic_ok}
        logger.info("voice_init", **result)

        # Quality gate: check mic before proceeding
        if mic_ok:
            self._mic_quality_ok, quality_report = self._microphone.check_audio_quality()
            result["mic_quality"] = self._mic_quality_ok
            if not self._mic_quality_ok:
                logger.warning("mic_quality_gate_failed", report=quality_report)
            else:
                logger.info("mic_quality_gate_passed", report=quality_report)

            # Calibrate noise floor
            self._microphone.calibrate_noise_floor()
            self._command_recorder.set_noise_floor(self._microphone.noise_floor)

        return result

    async def start(self) -> None:
        self._running = True
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="voice-engine")
        self._thread.start()
        self._set_state(VoiceState.STANDBY)
        logger.info("voice_engine_started")

    async def stop(self) -> None:
        self._running = False
        self._set_state(VoiceState.STOPPING)
        self._tts.cancel()
        self._microphone.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._set_state(VoiceState.OFFLINE)
        logger.info("voice_engine_stopped")

    def _run_loop(self) -> None:
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self._running:
            try:
                consecutive_errors = 0
                self._phase_standby_and_detect()
                if not self._running:
                    break
                self._phase_acknowledge()
                if not self._running:
                    break
                self._phase_listen_command()
                if not self._running:
                    break
                self._phase_transcribe_and_execute()
            except Exception as e:
                logger.error("voice_loop_error", error=str(e), exc_info=True)
                self._diagnostics["errors"] += 1
                consecutive_errors += 1
                self._set_state(VoiceState.ERROR)
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("too_many_errors", count=consecutive_errors)
                    time.sleep(5)
                else:
                    time.sleep(1)
                if self._running:
                    self._set_state(VoiceState.STANDBY)

    def _phase_standby_and_detect(self) -> None:
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
        self._set_state(VoiceState.WAKE_DETECTED)
        time.sleep(0.2)
        self._set_state(VoiceState.ACKNOWLEDGING)
        self._microphone.mute()
        future = asyncio.run_coroutine_threadsafe(self._tts.speak("Yes?"), self._loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass
        time.sleep(0.4)
        self._microphone.flush()
        self._microphone.unmute()

    def _phase_listen_command(self) -> None:
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
        self._pending_command_audio = speech_audio

    def _phase_transcribe_and_execute(self) -> None:
        if self._pending_command_audio is None:
            return

        speech_audio = self._pending_command_audio
        self._pending_command_audio = None

        self._set_state(VoiceState.TRANSCRIBING)
        try:
            future = asyncio.run_coroutine_threadsafe(self._stt.transcribe(speech_audio), self._loop)
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

        if result_text:
            self._set_state(VoiceState.SPEAKING)
            self._microphone.mute()
            self._safe_tts_speak(result_text)
            self._diagnostics["tts_spoken"] += 1
            time.sleep(0.3)
            self._microphone.unmute()

    def _safe_tts_speak(self, text: str) -> None:
        future = asyncio.run_coroutine_threadsafe(self._tts.speak(text), self._loop)
        try:
            future.result(timeout=30)
        except Exception:
            pass

    def interrupt(self) -> None:
        self._tts.cancel()
        self._microphone.flush()

    def _set_state(self, state: VoiceState) -> None:
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
        return {
            "running": self._running,
            "state": self._state.value,
            "stt_available": self._stt.is_available(),
            "tts_available": self._tts.is_available(),
            "tts_speaking": self._tts.is_speaking,
            "microphone": self._microphone.get_level(),
            "mic_quality_ok": self._mic_quality_ok,
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
        result = {}
        mic_level = self._microphone.get_level()
        result["microphone"] = {
            "status": "OK" if mic_level["active"] else "NOT ACTIVE",
            "device": mic_level["device"],
            "device_index": mic_level["device_index"],
            "host_api": mic_level.get("host_api", "unknown"),
            "probe_rms": mic_level.get("probe_rms", 0),
            "probe_score": mic_level.get("probe_score", 0),
            "sample_rate": mic_level["sample_rate"],
            "avg_rms": mic_level["avg_rms"],
            "peak": mic_level["peak"],
            "noise_floor": mic_level["noise_floor"],
            "calibrated": mic_level["calibrated"],
            "quality_ok": self._mic_quality_ok,
        }
        result["stt"] = {"status": "OK" if self._stt.is_available() else "NOT LOADED", "model": self._config.stt_model_size}
        result["tts"] = {"status": "OK" if self._tts.is_available() else "NOT AVAILABLE", "voice": self._config.tts_voice}
        result["wake_word"] = {"status": "OK", "phrase": self._config.wake_phrase, "method": "VAD + STT transcription"}
        result["stats"] = self._diagnostics.copy()
        return result
