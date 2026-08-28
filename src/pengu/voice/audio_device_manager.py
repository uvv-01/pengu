"""
AudioDeviceManager — single source of truth for microphone device selection.

Both --mic-test and VoiceEngine.initialize() MUST use this class.

Architecture:
    Phase 1: Silent baseline capture -> noise floor
    Phase 2: User speech capture   -> speech RMS, SNR

SNR:
    SNR_dB = 20 * log10(speech_rms / noise_floor)

Noise floor:
    Estimated from the 75th percentile of silence-frame RMS values.

The manager:
    - Enumerates input devices
    - Filters obvious non-microphones
    - Probes microphones using silence + speech
    - Performs multiple rounds
    - Uses median statistics across rounds
    - Scores devices consistently
    - Selects the best voice-ready microphone
    - Supports PENGU_MIC_DEVICE override
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from pengu.logging import get_logger


logger = get_logger("pengu.voice.audio_device_manager")


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

MIN_SNR_DB = 6.0
MIN_SPEECH_RMS = 100.0
MIN_NOISE_FLOOR = 0.1

DEFAULT_NUM_ROUNDS = 3
DEFAULT_SILENCE_DURATION = 1.0
DEFAULT_SPEECH_DURATION = 2.0

CLIPPING_THRESHOLD = 32000.0


# ============================================================================
# QUALITY
# ============================================================================


class AudioQuality(str, Enum):
    """Objective microphone quality states."""

    UNUSABLE = "UNUSABLE"
    POOR = "POOR"
    NO_SPEECH = "NO_SPEECH"
    ACCEPTABLE = "ACCEPTABLE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"

    @property
    def is_voice_ready(self) -> bool:
        """Voice-ready means the device has demonstrated speech capture."""
        return self in (
            AudioQuality.ACCEPTABLE,
            AudioQuality.GOOD,
            AudioQuality.EXCELLENT,
        )

    @property
    def is_hardware_ok(self) -> bool:
        """Hardware is OK even if no speech was observed during probe."""
        return self in (
            AudioQuality.NO_SPEECH,
            AudioQuality.ACCEPTABLE,
            AudioQuality.GOOD,
            AudioQuality.EXCELLENT,
        )


# ============================================================================
# DEVICE SELECTION
# ============================================================================


@dataclass
class DeviceSelection:
    """Complete result of microphone selection."""

    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float

    capture_sample_rate: int = TARGET_SAMPLE_RATE
    capture_channels: int = TARGET_CHANNELS

    # Silence / noise
    noise_floor: float = 0.0
    noise_rms: float = 0.0

    # Speech
    speech_rms: float = 0.0
    speech_peak: float = 0.0

    # General
    rms: float = 0.0
    peak: float = 0.0

    # Derived
    snr_db: float = 0.0
    clipping_percent: float = 0.0

    # Quality
    quality: AudioQuality = AudioQuality.UNUSABLE
    quality_score: float = 0.0

    selection_reason: str = ""

    # Status
    voice_ready: bool = False
    speech_detected: bool = False

    # All probes
    all_probes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert selection into a JSON-friendly dictionary."""

        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "host_api": self.host_api,
            "capture_sample_rate": self.capture_sample_rate,
            "capture_channels": self.capture_channels,
            "noise_floor": round(self.noise_floor, 1),
            "noise_rms": round(self.noise_rms, 1),
            "speech_rms": round(self.speech_rms, 1),
            "speech_peak": round(self.speech_peak, 1),
            "rms": round(self.rms, 1),
            "peak": round(self.peak, 1),
            "snr_db": round(self.snr_db, 1),
            "clipping_percent": round(self.clipping_percent, 4),
            "quality": self.quality.value,
            "quality_score": round(self.quality_score, 1),
            "voice_ready": self.voice_ready,
            "speech_detected": self.speech_detected,
            "selection_reason": self.selection_reason,
            "all_probes": self.all_probes,
        }


# ============================================================================
# DEVICE FILTERING
# ============================================================================


NON_MICROPHONE_PATTERNS = [
    "stereo mix",
    "loopback",
    "wave out",
    "what u hear",
    "monitor",
]

NON_MICROPHONE_DEVICE_NAMES = [
    "pc speaker",
    "sound mapper",
]


def _is_real_microphone(name: str) -> bool:
    """
    Return True when a device looks like a physical microphone.

    Some Windows endpoints are abstractions, so we intentionally filter
    obvious non-microphone names rather than relying only on host API.
    """

    name_lower = name.lower()

    for pattern in NON_MICROPHONE_PATTERNS:
        if pattern in name_lower:
            return False

    for pattern in NON_MICROPHONE_DEVICE_NAMES:
        if pattern in name_lower:
            return False

    return True


# ============================================================================
# AUDIO UTILITIES
# ============================================================================


def _measure_frame_rms(
    audio: np.ndarray,
    frame_size: int = 1024,
) -> list[float]:
    """
    Measure RMS for every complete frame.

    DC offset is removed before calculating RMS.
    """

    flat = audio.flatten().astype(np.float32)

    values: list[float] = []

    for start in range(
        0,
        len(flat) - frame_size + 1,
        frame_size,
    ):
        frame = flat[start:start + frame_size]

        if frame.size == 0:
            continue

        frame = frame - np.mean(frame)

        rms = float(
            np.sqrt(
                np.mean(frame ** 2)
            )
        )

        if np.isfinite(rms):
            values.append(rms)

    return values


def _estimate_noise_floor(
    frame_rms: list[float],
) -> float:
    """
    Estimate normal background noise.

    Uses P75 of silence frame RMS values.

    This avoids:
        - noisy-frame mean inflation
        - unrealistic near-zero values caused by gating
    """

    if not frame_rms:
        return MIN_NOISE_FLOOR

    values = np.asarray(
        frame_rms,
        dtype=np.float32,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return MIN_NOISE_FLOOR

    noise_floor = float(
        np.percentile(values, 50)
    )

    return max(
        min(noise_floor, 500.0),
        MIN_NOISE_FLOOR,
    )


def _calculate_rms(
    audio: np.ndarray,
) -> float:
    """Calculate RMS after removing DC offset."""

    flat = audio.flatten().astype(np.float32)

    if flat.size == 0:
        return 0.0

    centered = flat - np.mean(flat)

    value = float(
        np.sqrt(
            np.mean(centered ** 2)
        )
    )

    if not np.isfinite(value):
        return 0.0

    return value


def _calculate_peak(
    audio: np.ndarray,
) -> float:
    """Calculate absolute peak after DC removal."""

    flat = audio.flatten().astype(np.float32)

    if flat.size == 0:
        return 0.0

    centered = flat - np.mean(flat)

    peak = float(
        np.max(
            np.abs(centered)
        )
    )

    if not np.isfinite(peak):
        return 0.0

    return peak


def _measure_clipping_percent(
    audio: np.ndarray,
    threshold: float = CLIPPING_THRESHOLD,
) -> float:
    """Calculate percentage of samples near int16 clipping."""

    flat = audio.flatten().astype(np.float32)

    if flat.size == 0:
        return 0.0

    clipped = np.sum(
        np.abs(flat) >= threshold
    )

    return float(
        clipped / len(flat) * 100.0
    )


def _calculate_snr_db(
    speech_rms: float,
    noise_floor: float,
) -> float:
    """
    Calculate:

        20 * log10(speech_rms / noise_floor)

    Invalid measurements return 0.
    """

    if noise_floor < MIN_NOISE_FLOOR:
        return 0.0

    if speech_rms <= 0:
        return 0.0

    ratio = speech_rms / noise_floor

    if ratio <= 0:
        return 0.0

    value = float(
        20.0 * np.log10(ratio)
    )

    if not np.isfinite(value):
        return 0.0

    return value


# ============================================================================
# QUALITY CLASSIFICATION
# ============================================================================


def _classify_quality(
    snr_db: float,
    speech_rms: float,
    clipping_pct: float,
    noise_floor: float,
    speech_detected: bool,
    can_open: bool = True,
) -> AudioQuality:
    """Classify microphone quality.

    NO_SPEECH means the device opened successfully but no speech was
    observed during the probe. This is NOT the same as POOR, which
    means the device has actual quality issues.
    """

    # Severe clipping.
    if clipping_pct > 1.0:
        return AudioQuality.UNUSABLE

    # No speech observed — device may still be excellent.
    if not speech_detected:
        if can_open and noise_floor < 500.0:
            return AudioQuality.NO_SPEECH
        return AudioQuality.POOR

    # Degenerate signal.
    if (
        speech_rms < 1.0
        and noise_floor < 1.0
    ):
        return AudioQuality.UNUSABLE

    # Minimum requirements.
    if snr_db < MIN_SNR_DB:
        return AudioQuality.POOR

    if speech_rms < MIN_SPEECH_RMS:
        return AudioQuality.POOR

    # Excellent.
    if (
        snr_db >= 20.0
        and speech_rms >= 500.0
    ):
        return AudioQuality.EXCELLENT

    # Good.
    if (
        snr_db >= 12.0
        and speech_rms >= 200.0
    ):
        return AudioQuality.GOOD

    # Minimum acceptable.
    return AudioQuality.ACCEPTABLE


# ============================================================================
# DEVICE SCORING
# ============================================================================


def _score_device(
    snr_db: float,
    speech_rms: float,
    noise_floor: float,
    clipping_pct: float,
    quality: AudioQuality,
    host_api: str,
    capture_sr: int,
    device_name: str,
) -> float:
    """
    Calculate composite microphone score.

    Higher = better.
    0 = unusable.
    """


    if quality == AudioQuality.UNUSABLE:
        return 0.0

    # No speech observed: score based on hardware potential.
    if quality == AudioQuality.NO_SPEECH:
        score = 100.0
        # Prefer lower noise floor
        if noise_floor < 1.0:
            score *= 2.0
        elif noise_floor < 10.0:
            score *= 1.5
        # Prefer 16 kHz native
        if capture_sr == TARGET_SAMPLE_RATE:
            score *= 1.2
        # Prefer WDM-KS (lowest latency)
        if "WDM-KS" in host_api:
            score *= 1.15
        elif "DirectSound" in host_api:
            score *= 1.10
        elif host_api == "MME":
            score *= 0.80
        return float(score)

    if not _is_real_microphone(device_name):
        return 0.0

    score = float(speech_rms)

    # SNR weighting.
    if snr_db >= 20.0:
        score *= 2.0
    elif snr_db >= 12.0:
        score *= 1.5
    elif snr_db >= MIN_SNR_DB:
        score *= 1.0
    else:
        score *= 0.3

    # Clipping penalty.
    if clipping_pct > 0.1:
        score *= 0.3
    elif clipping_pct > 0.01:
        score *= 0.7

    # Host API preference.
    if "DirectSound" in host_api:
        score *= 1.10

    elif "WDM-KS" in host_api:
        score *= 1.15

    elif host_api == "MME":
        score *= 0.80

    # Prefer 16 kHz speech-native capture.
    if capture_sr == TARGET_SAMPLE_RATE:
        score *= 1.20

    # Quality multiplier.
    quality_multiplier = {
        AudioQuality.EXCELLENT: 1.50,
        AudioQuality.GOOD: 1.20,
        AudioQuality.ACCEPTABLE: 1.00,
        AudioQuality.POOR: 0.40,
        AudioQuality.UNUSABLE: 0.00,
    }

    score *= quality_multiplier.get(
        quality,
        0.0,
    )

    return float(score)


# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================


def _downmix_to_mono(
    audio: np.ndarray,
) -> np.ndarray:
    """Convert multi-channel audio to mono."""

    if audio.ndim == 1:
        return audio

    if audio.shape[1] == 1:
        return audio.flatten()

    return np.mean(
        audio,
        axis=1,
    ).astype(audio.dtype)


def _resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio using linear interpolation."""

    if orig_sr == target_sr:
        return audio

    if orig_sr <= 0 or target_sr <= 0:
        return audio

    flat = audio.flatten().astype(np.float32)

    if flat.size == 0:
        return audio

    duration = len(flat) / orig_sr

    new_length = int(
        duration * target_sr
    )

    if new_length <= 0:
        return audio

    indices = np.linspace(
        0,
        len(flat) - 1,
        new_length,
    )

    resampled = np.interp(
        indices,
        np.arange(len(flat)),
        flat,
    )

    return resampled.astype(
        audio.dtype
    )


# ============================================================================
# PROBE RESULT
# ============================================================================


@dataclass
class ProbeResult:
    """Result of microphone probing."""

    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float

    capture_sample_rate: int = 0
    capture_channels: int = 0

    # Silence
    noise_floor: float = 0.0
    noise_rms: float = 0.0

    # Speech
    speech_rms: float = 0.0
    speech_peak: float = 0.0

    # Overall
    overall_rms: float = 0.0
    overall_peak: float = 0.0

    # Derived
    snr_db: float = 0.0
    clipping_percent: float = 0.0

    # Status
    speech_detected: bool = False
    can_open: bool = False
    error: str = ""

    # Quality
    quality: AudioQuality = AudioQuality.UNUSABLE
    quality_score: float = 0.0

    # Multi-round measurements
    round_noise_floor_values: list[float] = field(
        default_factory=list
    )

    round_noise_rms_values: list[float] = field(
        default_factory=list
    )

    round_speech_rms_values: list[float] = field(
        default_factory=list
    )

    round_speech_peak_values: list[float] = field(
        default_factory=list
    )

    round_snr_values: list[float] = field(
        default_factory=list
    )

    round_clipping_values: list[float] = field(
        default_factory=list
    )


# ============================================================================
# TWO-PHASE MICROPHONE PROBE
# ============================================================================


def _probe_device_two_phase(
    dev_idx: int,
    dev_info: dict,
    host_api_name: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    speech_duration: float = DEFAULT_SPEECH_DURATION,
    prompt_user: bool = False,
) -> ProbeResult:
    """
    Probe one microphone using silence + speech phases.

    Each round:
        1. Record silence.
        2. Estimate noise floor.
        3. Record speech.
        4. Calculate RMS / peak / SNR / clipping.

    Final measurements use medians across successful rounds.
    """

    import sounddevice as sd

    result = ProbeResult(
        device_index=dev_idx,
        device_name=dev_info.get(
            "name",
            f"Device {dev_idx}",
        ),
        host_api=host_api_name,
        max_channels=dev_info.get(
            "max_input_channels",
            0,
        ),
        native_sample_rate=dev_info.get(
            "default_samplerate",
            0,
        ),
    )

    # ----------------------------------------------------------------------
    # Determine capture sample rate
    # ----------------------------------------------------------------------

    capture_sr = int(target_sr)

    try:
        sd.check_input_settings(
            device=dev_idx,
            samplerate=capture_sr,
            channels=1,
            dtype="int16",
        )

    except Exception:
        native_sr = int(
            dev_info.get(
                "default_samplerate",
                target_sr,
            )
        )

        try:
            sd.check_input_settings(
                device=dev_idx,
                samplerate=native_sr,
                channels=1,
                dtype="int16",
            )

            capture_sr = native_sr

        except Exception as exc:
            result.error = (
                f"Cannot open device: {exc}"
            )
            return result

    result.capture_sample_rate = capture_sr
    result.capture_channels = 1

    # Protect against invalid round values.
    num_rounds = max(
        1,
        int(num_rounds),
    )

    # ----------------------------------------------------------------------
    # Per-round measurements
    # ----------------------------------------------------------------------

    for round_num in range(num_rounds):

        # ==============================================================
        # PHASE 1 — SILENCE
        # ==============================================================

        if prompt_user:
            print()
            print(
                f"  Round {round_num + 1}/{num_rounds}: "
                f"Stay silent for "
                f"{silence_duration:.1f} seconds..."
            )

        try:
            silence_audio = sd.rec(
                int(
                    silence_duration
                    * capture_sr
                ),
                samplerate=capture_sr,
                channels=1,
                dtype="int16",
                device=dev_idx,
            )

            sd.wait()

            if (
                silence_audio is None
                or silence_audio.size == 0
            ):
                raise RuntimeError(
                    "Empty silence capture"
                )

            frame_rms = _measure_frame_rms(
                silence_audio
            )

            if not frame_rms:
                raise RuntimeError(
                    "No silence frames captured"
                )

            noise_floor = _estimate_noise_floor(
                frame_rms
            )

            noise_rms = _calculate_rms(
                silence_audio
            )

        except Exception as exc:
            logger.debug(
                "probe_silence_failed",
                device=dev_idx,
                round=round_num,
                error=str(exc),
            )
            continue

        # ==============================================================
        # PHASE 2 — SPEECH
        # ==============================================================

        if prompt_user:
            sys.stdout.write(
                f"  Speak normally for "
                f"{speech_duration:.1f} seconds "
                f"(say 'Pengu microphone test')..."
            )
            sys.stdout.flush()

            # Give the user a tiny gap after the prompt.
            time.sleep(0.3)

        try:
            speech_audio = sd.rec(
                int(
                    speech_duration
                    * capture_sr
                ),
                samplerate=capture_sr,
                channels=1,
                dtype="int16",
                device=dev_idx,
            )

            sd.wait()

            if (
                speech_audio is None
                or speech_audio.size == 0
            ):
                raise RuntimeError(
                    "Empty speech capture"
                )

            speech_rms = _calculate_rms(
                speech_audio
            )

            speech_peak = _calculate_peak(
                speech_audio
            )

            clipping = _measure_clipping_percent(
                speech_audio
            )

            # ----------------------------------------------------------
            # Speech detection
            # ----------------------------------------------------------

            speech_detected = (
                speech_rms >= MIN_SPEECH_RMS
                and speech_rms
                >= noise_floor * 3.0
            )

            # ----------------------------------------------------------
            # SNR
            # ----------------------------------------------------------

            if speech_detected:
                snr_db = _calculate_snr_db(
                    speech_rms,
                    noise_floor,
                )
            else:
                snr_db = 0.0

            # ----------------------------------------------------------
            # Store round
            # ----------------------------------------------------------

            result.round_noise_floor_values.append(
                noise_floor
            )

            result.round_noise_rms_values.append(
                noise_rms
            )

            result.round_speech_rms_values.append(
                speech_rms
            )

            result.round_speech_peak_values.append(
                speech_peak
            )

            result.round_snr_values.append(
                snr_db
            )

            result.round_clipping_values.append(
                clipping
            )

            if prompt_user:
                print()
                print(
                    f"    Noise Floor : "
                    f"{noise_floor:.1f}"
                )
                print(
                    f"    Noise RMS   : "
                    f"{noise_rms:.1f}"
                )
                print(
                    f"    Speech RMS  : "
                    f"{speech_rms:.1f}"
                )
                print(
                    f"    Speech Peak : "
                    f"{speech_peak:.1f}"
                )
                print(
                    f"    SNR         : "
                    f"{snr_db:.1f} dB"
                )
                print(
                    f"    Clipping    : "
                    f"{clipping:.4f}%"
                )
                print(
                    f"    Speech      : "
                    f"{'YES' if speech_detected else 'NO'}"
                )

        except Exception as exc:
            logger.debug(
                "probe_speech_failed",
                device=dev_idx,
                round=round_num,
                error=str(exc),
            )
            continue

    # ----------------------------------------------------------------------
    # No successful rounds
    # ----------------------------------------------------------------------

    if not result.round_speech_rms_values:
        result.error = "All probe rounds failed"
        return result

    result.can_open = True

    # ----------------------------------------------------------------------
    # Robust median statistics
    # ----------------------------------------------------------------------

    result.noise_floor = float(
        np.median(
            result.round_noise_floor_values
        )
    )

    result.noise_rms = float(
        np.median(
            result.round_noise_rms_values
        )
    )

    result.speech_rms = float(
        np.median(
            result.round_speech_rms_values
        )
    )

    result.speech_peak = float(
        np.median(
            result.round_speech_peak_values
        )
    )

    result.snr_db = float(
        np.median(
            result.round_snr_values
        )
    )

    result.clipping_percent = float(
        np.median(
            result.round_clipping_values
        )
    )

    # Overall measurements are now consistent with the final
    # median speech measurement.
    result.overall_rms = result.speech_rms
    result.overall_peak = result.speech_peak

    # Speech is considered detected if at least one successful
    # round contained a valid speech signal.
    result.speech_detected = any(
        rms >= MIN_SPEECH_RMS
        and snr >= MIN_SNR_DB
        for rms, snr in zip(
            result.round_speech_rms_values,
            result.round_snr_values,
        )
    )

    # ----------------------------------------------------------------------
    # Quality
    # ----------------------------------------------------------------------

    result.quality = _classify_quality(
        snr_db=result.snr_db,
        speech_rms=result.speech_rms,
        clipping_pct=result.clipping_percent,
        noise_floor=result.noise_floor,
        speech_detected=result.speech_detected,
        can_open=result.can_open,
    )

    # ----------------------------------------------------------------------
    # Score
    # ----------------------------------------------------------------------

    result.quality_score = _score_device(
        snr_db=result.snr_db,
        speech_rms=result.speech_rms,
        noise_floor=result.noise_floor,
        clipping_pct=result.clipping_percent,
        quality=result.quality,
        host_api=host_api_name,
        capture_sr=capture_sr,
        device_name=result.device_name,
    )

    return result


# ============================================================================
# AUDIO DEVICE MANAGER
# ============================================================================


class AudioDeviceManager:
    """
    Single source of truth for microphone device selection.

    Both --mic-test and VoiceEngine MUST use this class.
    """

    def __init__(
        self,
        configured_device: Optional[int] = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        num_rounds: int = DEFAULT_NUM_ROUNDS,
        prompt_user: bool = False,

        # Backwards compatibility:
        # Older mic_diagnostics.py may pass nums_rounds.
        nums_rounds: Optional[int] = None,
    ) -> None:

        self._configured_device = (
            configured_device
        )

        self._target_sr = int(
            target_sample_rate
        )

        # Support both spellings:
        #
        #     num_rounds
        #     nums_rounds
        #
        # This directly fixes:
        #
        # TypeError:
        # AudioDeviceManager.__init__()
        # got an unexpected keyword argument 'nums_rounds'
        if nums_rounds is not None:
            self._num_rounds = max(
                1,
                int(nums_rounds),
            )
        else:
            self._num_rounds = max(
                1,
                int(num_rounds),
            )

        self._prompt_user = bool(
            prompt_user
        )

        self._all_probes: list[ProbeResult] = []

    # ======================================================================
    # PROPERTIES
    # ======================================================================

    @property
    def all_probes(self) -> list[ProbeResult]:
        """Return all probe results."""

        return self._all_probes

    @property
    def selected_device(self) -> Optional[DeviceSelection]:
        """
        Convenience property.

        Returns the highest-scoring successful probe,
        or None if no microphone is available.
        """

        candidates = [
            probe
            for probe in self._all_probes
            if (
                probe.can_open
                and probe.quality_score > 0
            )
        ]

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda probe: probe.quality_score,
        )

        return self._probe_to_selection(
            best,
            (
                f"Score {best.quality_score:.1f}, "
                f"SNR={best.snr_db:.1f}dB"
            ),
        )

    # ======================================================================
    # ENUMERATE DEVICES
    # ======================================================================

    def enumerate_devices(self) -> list[dict]:
        """Enumerate all input devices."""

        try:
            import sounddevice as sd
        except ImportError:
            logger.error(
                "sounddevice_not_installed"
            )
            return []

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:
            logger.error(
                "audio_device_enumeration_failed",
                error=str(exc),
            )
            return []

        result: list[dict] = []

        for index, dev in enumerate(devices):

            max_input_channels = int(
                dev.get(
                    "max_input_channels",
                    0,
                )
            )

            if max_input_channels <= 0:
                continue

            api_idx = int(
                dev.get(
                    "hostapi",
                    0,
                )
            )

            if (
                0 <= api_idx
                < len(host_apis)
            ):
                api_name = host_apis[
                    api_idx
                ]["name"]
            else:
                api_name = "Unknown"

            name = dev.get(
                "name",
                f"Device {index}",
            )

            result.append(
                {
                    "index": index,
                    "name": name,
                    "host_api": api_name,
                    "max_channels": max_input_channels,
                    "native_sample_rate": dev.get(
                        "default_samplerate",
                        0,
                    ),
                    "is_real_mic": (
                        _is_real_microphone(name)
                    ),
                }
            )

        return result

    # ======================================================================
    # PROBE ONE DEVICE
    # ======================================================================

    def probe_device(
        self,
        device_index: int,
        prompt_user: bool = False,
    ) -> ProbeResult:
        """Probe one microphone."""

        try:
            import sounddevice as sd
        except ImportError:
            return ProbeResult(
                device_index=device_index,
                device_name="unknown",
                host_api="unknown",
                max_channels=0,
                native_sample_rate=0,
                error="sounddevice not installed",
            )

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:
            return ProbeResult(
                device_index=device_index,
                device_name="unknown",
                host_api="unknown",
                max_channels=0,
                native_sample_rate=0,
                error=str(exc),
            )

        if (
            device_index < 0
            or device_index >= len(devices)
        ):
            return ProbeResult(
                device_index=device_index,
                device_name="unknown",
                host_api="unknown",
                max_channels=0,
                native_sample_rate=0,
                error=(
                    f"Device {device_index} "
                    f"out of range"
                ),
            )

        dev = devices[device_index]

        api_idx = int(
            dev.get(
                "hostapi",
                0,
            )
        )

        if (
            0 <= api_idx
            < len(host_apis)
        ):
            api_name = host_apis[
                api_idx
            ]["name"]
        else:
            api_name = "Unknown"

        return _probe_device_two_phase(
            dev_idx=device_index,
            dev_info=dev,
            host_api_name=api_name,
            target_sr=self._target_sr,
            num_rounds=self._num_rounds,
            prompt_user=(
                prompt_user
                or self._prompt_user
            ),
        )

    # ======================================================================
    # SELECT BEST DEVICE
    # ======================================================================

    def select_best_device(
        self,
    ) -> Optional[DeviceSelection]:
        """
        Select the best microphone.

        Priority:

        1. PENGU_MIC_DEVICE override
        2. Real microphones
        3. Voice-ready devices
        4. Highest score
        """

        # ------------------------------------------------------------------
        # Explicit environment override
        # ------------------------------------------------------------------

        env_device = os.environ.get(
            "PENGU_MIC_DEVICE"
        )

        if env_device:
            try:
                dev_idx = int(
                    env_device
                )

                probe = self.probe_device(
                    dev_idx,
                    prompt_user=(
                        self._prompt_user
                    ),
                )

                if probe.can_open:
                    self._all_probes = [
                        probe
                    ]

                    logger.info(
                        "mic_selected_override",
                        device=probe.device_index,
                        name=probe.device_name,
                        api=probe.host_api,
                    )

                    return self._probe_to_selection(
                        probe,
                        (
                            "PENGU_MIC_DEVICE="
                            f"{env_device}"
                        ),
                    )

                logger.warning(
                    "env_device_unusable",
                    device=env_device,
                    error=probe.error,
                )

            except Exception as exc:
                logger.warning(
                    "env_device_failed",
                    device=env_device,
                    error=str(exc),
                )

        # ------------------------------------------------------------------
        # Configured device
        # ------------------------------------------------------------------

        if self._configured_device is not None:

            try:
                probe = self.probe_device(
                    int(
                        self._configured_device
                    ),
                    prompt_user=(
                        self._prompt_user
                    ),
                )

                if probe.can_open:
                    self._all_probes = [
                        probe
                    ]

                    logger.info(
                        "mic_selected_configured",
                        device=probe.device_index,
                        name=probe.device_name,
                        api=probe.host_api,
                    )

                    return self._probe_to_selection(
                        probe,
                        (
                            "configured_device="
                            f"{self._configured_device}"
                        ),
                    )

            except Exception as exc:
                logger.warning(
                    "configured_device_failed",
                    device=self._configured_device,
                    error=str(exc),
                )

        # ------------------------------------------------------------------
        # Enumerate
        # ------------------------------------------------------------------

        devices = self.enumerate_devices()

        if not devices:
            logger.error(
                "no_input_devices"
            )
            return None

        logger.info(
            "mic_enumerated",
            count=len(devices),
        )

        # ------------------------------------------------------------------
        # Interactive information
        # ------------------------------------------------------------------

        if self._prompt_user:
            print()
            print(
                f"  Probing {len(devices)} input devices..."
            )
            print(
                "  Each device: silence baseline + speech capture"
            )
            print(
                "  Speak only when prompted."
            )
            print()

        # ------------------------------------------------------------------
        # Probe real microphones
        # ------------------------------------------------------------------

        probes: list[ProbeResult] = []

        for dev in devices:

            if not dev["is_real_mic"]:
                logger.debug(
                    "skipping_non_microphone",
                    device=dev["index"],
                    name=dev["name"],
                )
                continue

            probe = self.probe_device(
                dev["index"],
                prompt_user=self._prompt_user,
            )

            probes.append(probe)

            if probe.can_open:

                logger.info(
                    "mic_probed",
                    device=probe.device_index,
                    name=probe.device_name,
                    api=probe.host_api,
                    noise=round(
                        probe.noise_floor,
                        1,
                    ),
                    speech=round(
                        probe.speech_rms,
                        1,
                    ),
                    snr=round(
                        probe.snr_db,
                        1,
                    ),
                    quality=probe.quality.value,
                    score=round(
                        probe.quality_score,
                        1,
                    ),
                    speech_detected=(
                        probe.speech_detected
                    ),
                )

            else:

                logger.debug(
                    "mic_probe_failed",
                    device=probe.device_index,
                    error=probe.error,
                )

        self._all_probes = probes

        # ------------------------------------------------------------------
        # Voice-ready microphones
        # ------------------------------------------------------------------

        # Prefer devices with speech evidence
        usable = [
            probe
            for probe in probes
            if (
                probe.can_open
                and probe.quality.is_voice_ready
                and probe.quality_score > 0
                and probe.speech_detected
            )
        ]

        # Fallback: devices with hardware OK but no speech observed
        if not usable:
            usable = [
                probe
                for probe in probes
                if (
                    probe.can_open
                    and probe.quality.is_hardware_ok
                    and probe.quality_score > 0
                )
            ]

        # Last resort: any device with positive score
        if not usable:
            usable = [
                probe
                for probe in probes
                if (
                    probe.can_open
                    and probe.quality_score > 0
                )
            ]

        # ------------------------------------------------------------------
        # Nothing usable
        # ------------------------------------------------------------------

        if not usable:

            logger.error(
                "no_usable_microphone"
            )

            return None

        # ------------------------------------------------------------------
        # Highest score
        # ------------------------------------------------------------------

        usable.sort(
            key=lambda probe: (
                probe.quality_score,
                probe.snr_db,
                probe.speech_rms,
            ),
            reverse=True,
        )

        best = usable[0]

        reason = (
            f"Score {best.quality_score:.1f}, "
            f"SNR={best.snr_db:.1f}dB"
        )

        if best.speech_detected:
            reason += ", speech detected"
        else:
            reason += ", NO SPEECH DETECTED"

        logger.info(
            "mic_selected",
            device=best.device_index,
            name=best.device_name,
            api=best.host_api,
            noise=round(
                best.noise_floor,
                1,
            ),
            speech=round(
                best.speech_rms,
                1,
            ),
            snr=round(
                best.snr_db,
                1,
            ),
            quality=best.quality.value,
            score=round(
                best.quality_score,
                1,
            ),
        )

        return self._probe_to_selection(
            best,
            reason,
        )

    # ======================================================================
    # PROBE -> SELECTION
    # ======================================================================

    def _probe_to_selection(
        self,
        probe: ProbeResult,
        reason: str,
    ) -> DeviceSelection:
        """Convert ProbeResult into DeviceSelection."""

        return DeviceSelection(
            device_index=probe.device_index,
            device_name=probe.device_name,
            host_api=probe.host_api,
            max_channels=probe.max_channels,
            native_sample_rate=probe.native_sample_rate,

            capture_sample_rate=(
                probe.capture_sample_rate
            ),

            capture_channels=(
                probe.capture_channels
            ),

            noise_floor=(
                probe.noise_floor
            ),

            noise_rms=(
                probe.noise_rms
            ),

            speech_rms=(
                probe.speech_rms
            ),

            speech_peak=(
                probe.speech_peak
            ),

            rms=(
                probe.overall_rms
            ),

            peak=(
                probe.overall_peak
            ),

            snr_db=(
                probe.snr_db
            ),

            clipping_percent=(
                probe.clipping_percent
            ),

            quality=(
                probe.quality
            ),

            quality_score=(
                probe.quality_score
            ),

            selection_reason=reason,

            voice_ready=(
                probe.quality.is_voice_ready
            ),

            speech_detected=(
                probe.speech_detected
            ),

            all_probes=[
                {
                    "device": p.device_index,
                    "name": p.device_name,
                    "api": p.host_api,
                    "sample_rate": (
                        p.capture_sample_rate
                    ),
                    "channels": (
                        p.capture_channels
                    ),
                    "noise": round(
                        p.noise_floor,
                        1,
                    ),
                    "noise_rms": round(
                        p.noise_rms,
                        1,
                    ),
                    "speech": round(
                        p.speech_rms,
                        1,
                    ),
                    "speech_peak": round(
                        p.speech_peak,
                        1,
                    ),
                    "snr": round(
                        p.snr_db,
                        1,
                    ),
                    "clipping": round(
                        p.clipping_percent,
                        4,
                    ),
                    "score": round(
                        p.quality_score,
                        1,
                    ),
                    "quality": (
                        p.quality.value
                    ),
                    "speech_detected": (
                        p.speech_detected
                    ),
                    "can_open": (
                        p.can_open
                    ),
                    "error": (
                        p.error
                    ),
                    "round_noise": [
                        round(v, 1)
                        for v in (
                            p.round_noise_floor_values
                        )
                    ],
                    "round_speech": [
                        round(v, 1)
                        for v in (
                            p.round_speech_rms_values
                        )
                    ],
                    "round_snr": [
                        round(v, 1)
                        for v in (
                            p.round_snr_values
                        )
                    ],
                }
                for p in self._all_probes
                if p.can_open
            ],
        )


# ============================================================================
# PUBLIC HELPER
# ============================================================================


def get_audio_device_manager(
    configured_device: Optional[int] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    prompt_user: bool = False,
) -> AudioDeviceManager:
    """
    Convenience factory.

    Keeps construction consistent across the project.
    """

    return AudioDeviceManager(
        configured_device=configured_device,
        target_sample_rate=target_sample_rate,
        num_rounds=num_rounds,
        prompt_user=prompt_user,
    )