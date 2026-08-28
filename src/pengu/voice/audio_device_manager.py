"""
AudioDeviceManager — single source of truth for microphone device selection.

Both:
  python -m pengu --mic-test
  VoiceEngine.initialize()

MUST call the same AudioDeviceManager for device selection.

This eliminates the architectural inconsistency where diagnostics and
production used separate selection algorithms producing different results.

Selection algorithm:
  1. Enumerate ALL input devices with host API info
  2. Filter out non-microphone endpoints (Stereo Mix, PC Speaker, etc.)
  3. Probe each device with a REAL 3-second capture
  4. Score using multi-factor system:
     - Speech signal strength (median frame RMS, not just total RMS)
     - Signal-to-noise ratio
     - Noise floor
     - Clipping percentage
     - Native 16kHz support
     - API reliability (DirectSound > WDM-KS > MME)
     - Physical microphone likelihood
  5. Return the device with the highest composite score
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from pengu.logging import get_logger

logger = get_logger("pengu.voice.audio_device_manager")

# Target audio format for STT / VAD / wake word
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------

class AudioQuality(str, Enum):
    """Classification of audio signal quality."""
    UNUSABLE_SILENCE = "UNUSABLE_SILENCE"
    UNUSABLE_NOISE = "UNUSABLE_NOISE"
    UNUSABLE_CLIPPING = "UNUSABLE_CLIPPING"
    WEAK = "WEAK"
    NOISY = "NOISY"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"

    def describe(self) -> str:
        descriptions = {
            AudioQuality.UNUSABLE_SILENCE: "Audio is essentially silent.",
            AudioQuality.UNUSABLE_NOISE: "Audio contains only noise.",
            AudioQuality.UNUSABLE_CLIPPING: "Audio is severely clipped.",
            AudioQuality.WEAK: "Audio signal is very weak.",
            AudioQuality.NOISY: "Audio signal is usable but noisy.",
            AudioQuality.GOOD: "Audio signal is good for speech recognition.",
            AudioQuality.EXCELLENT: "Audio signal is excellent for speech recognition.",
        }
        return descriptions.get(self, "Unknown quality")


# ---------------------------------------------------------------------------
# Structured device result
# ---------------------------------------------------------------------------

@dataclass
class DeviceSelection:
    """
    Complete result of device selection.
    This is the single structured object that represents a selected microphone.
    """
    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float
    # Capture configuration
    capture_sample_rate: int = TARGET_SAMPLE_RATE
    capture_channels: int = TARGET_CHANNELS
    # Quality measurements
    rms: float = 0.0
    peak: float = 0.0
    noise_floor: float = 0.0
    speech_rms: float = 0.0
    snr: float = 0.0
    clipping_percent: float = 0.0
    # Scoring
    quality: AudioQuality = AudioQuality.UNUSABLE_SILENCE
    quality_score: float = 0.0
    selection_reason: str = ""
    # Probe details
    probe_duration_seconds: float = 0.0
    probe_success: bool = False
    probe_error: str = ""
    # All probed devices (for comparison)
    all_probes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "host_api": self.host_api,
            "max_channels": self.max_channels,
            "native_sample_rate": self.native_sample_rate,
            "capture_sample_rate": self.capture_sample_rate,
            "capture_channels": self.capture_channels,
            "rms": round(self.rms, 1),
            "peak": round(self.peak, 1),
            "noise_floor": round(self.noise_floor, 1),
            "speech_rms": round(self.speech_rms, 1),
            "snr": round(self.snr, 1),
            "clipping_percent": round(self.clipping_percent, 4),
            "quality": self.quality.value,
            "quality_score": round(self.quality_score, 1),
            "selection_reason": self.selection_reason,
        }


# ---------------------------------------------------------------------------
# Non-microphone endpoint filtering
# ---------------------------------------------------------------------------

# Device name patterns that are NOT real microphone inputs
NON_MICROPHONE_PATTERNS = [
    "stereo mix",
    "pc speaker",
    "sound mapper",
    "primary sound capture",
    "loopback",
    "wave out",
    "what u hear",
    "monitor",
    "output",
]


def _is_real_microphone(name: str) -> bool:
    """
    Determine if a device name refers to a real microphone input.

    Returns False for:
      - System audio capture (Stereo Mix, PC Speaker)
      - Mapper/abstraction endpoints (Microsoft Sound Mapper)
      - Loopback devices
      - Output monitoring devices
    """
    name_lower = name.lower()
    for pattern in NON_MICROPHONE_PATTERNS:
        if pattern in name_lower:
            return False
    return True


# ---------------------------------------------------------------------------
# Audio measurement utilities
# ---------------------------------------------------------------------------

def _measure_frame_rms(audio: np.ndarray, frame_size: int = 1024) -> list[float]:
    """Calculate RMS per frame for robust statistics."""
    flat = audio.flatten().astype(np.float32)
    rms_values = []
    for i in range(0, len(flat) - frame_size, frame_size):
        frame = flat[i:i + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)
    return rms_values


def _measure_clipping_percent(audio: np.ndarray, threshold: float = 32000) -> float:
    """Calculate percentage of samples at or near max amplitude."""
    flat = audio.flatten().astype(np.float32)
    clipped = np.sum(np.abs(flat) >= threshold)
    return float(clipped / len(flat) * 100) if len(flat) > 0 else 0.0


def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono by averaging."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio.astype(np.float32), axis=1).astype(np.int16)
    return audio.flatten()


def _resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if orig_rate == target_rate:
        return audio
    duration = len(audio) / orig_rate
    num_samples = int(duration * target_rate)
    x_orig = np.linspace(0, duration, len(audio), endpoint=False)
    x_target = np.linspace(0, duration, num_samples, endpoint=False)
    return np.interp(x_target, x_orig, audio.astype(np.float64)).astype(np.int16)


# ---------------------------------------------------------------------------
# Device probing
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of probing a single device with multiple capture windows."""
    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float
    # Capture config used
    capture_sample_rate: int = 0
    capture_channels: int = 0
    # Measurements from the BEST window
    rms: float = 0.0
    peak: float = 0.0
    noise_floor: float = 0.0
    speech_rms: float = 0.0
    snr: float = 0.0
    clipping_percent: float = 0.0
    # Multiple window results for robustness
    window_rms_values: list[float] = field(default_factory=list)
    # Quality
    quality: AudioQuality = AudioQuality.UNUSABLE_SILENCE
    quality_score: float = 0.0
    # Status
    can_open: bool = False
    error: str = ""
    capture_time_seconds: float = 0.0


def _probe_device_robust(
    dev_idx: int,
    dev_info: dict,
    host_api_name: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    num_windows: int = 3,
    window_duration: float = 1.0,
) -> ProbeResult:
    """
    Probe a device with MULTIPLE short captures for robust statistics.

    Uses multiple windows to avoid selecting a device based on a single
    transient peak. Calculates median RMS across windows.

    Pipeline per window:
      1. Open device at 16kHz mono
      2. Capture 1 second
      3. Calculate frame-level RMS
      4. Calculate noise floor (10th percentile of frame RMS)
      5. Calculate speech RMS (90th percentile of frame RMS)
      6. Calculate SNR
      7. Check clipping
    """
    import sounddevice as sd

    result = ProbeResult(
        device_index=dev_idx,
        device_name=dev_info.get("name", f"Device {dev_idx}"),
        host_api=host_api_name,
        max_channels=dev_info.get("max_input_channels", 0),
        native_sample_rate=dev_info.get("default_samplerate", 0),
    )

    # Determine capture parameters
    capture_sr = target_sr
    capture_ch = 1

    # Try 16kHz mono first
    try:
        sd.check_input_settings(device=dev_idx, samplerate=target_sr, channels=1, dtype="int16")
    except Exception:
        # Fallback to native sample rate
        native_sr = int(dev_info.get("default_samplerate", target_sr))
        try:
            sd.check_input_settings(device=dev_idx, samplerate=native_sr, channels=1, dtype="int16")
            capture_sr = native_sr
        except Exception as e:
            result.error = f"Cannot open device: {e}"
            return result

    result.capture_sample_rate = capture_sr
    result.capture_channels = capture_ch

    # Capture multiple windows
    all_window_rms = []
    all_window_snr = []
    all_window_noise = []
    all_window_speech = []
    all_window_peak = []
    all_window_clipping = []
    total_capture_time = 0.0

    for w in range(num_windows):
        try:
            start = time.time()
            audio = sd.rec(
                int(window_duration * capture_sr),
                samplerate=capture_sr,
                channels=capture_ch,
                dtype="int16",
                device=dev_idx,
            )
            sd.wait()
            capture_time = time.time() - start
            total_capture_time += capture_time

            if audio is None or audio.size == 0:
                continue

            flat = audio.flatten().astype(np.float32)

            # Frame-level analysis
            frame_rms = _measure_frame_rms(audio, frame_size=1024)

            if not frame_rms:
                continue

            # Robust statistics using percentiles
            sorted_rms = sorted(frame_rms)
            noise_count = max(1, len(sorted_rms) // 10)  # Bottom 10%
            speech_count = max(1, len(sorted_rms) // 3)  # Top 33%

            noise_floor = float(np.mean(sorted_rms[:noise_count]))
            speech_rms = float(np.mean(sorted_rms[-speech_count:]))
            rms = float(np.sqrt(np.mean(flat ** 2)))
            peak = float(np.max(np.abs(flat)))

            # SNR
            snr = 0.0
            if noise_floor > 0.5:
                snr = 20 * np.log10(speech_rms / noise_floor) if speech_rms > 0 else 0.0

            # Clipping
            clipping = _measure_clipping_percent(audio)

            all_window_rms.append(rms)
            all_window_snr.append(snr)
            all_window_noise.append(noise_floor)
            all_window_speech.append(speech_rms)
            all_window_peak.append(peak)
            all_window_clipping.append(clipping)

        except Exception as e:
            logger.debug("probe_window_failed", device=dev_idx, window=w, error=str(e))
            continue

    if not all_window_rms:
        result.error = "All capture windows failed"
        return result

    result.can_open = True
    result.capture_time_seconds = total_capture_time

    # Use MEDIAN across windows for robustness (not max, not mean)
    result.rms = float(np.median(all_window_rms))
    result.peak = float(np.median(all_window_peak))
    result.noise_floor = float(np.median(all_window_noise))
    result.speech_rms = float(np.median(all_window_speech))
    result.snr = float(np.median(all_window_snr))
    result.clipping_percent = float(np.median(all_window_clipping))
    result.window_rms_values = all_window_rms

    # Classify quality
    result.quality = _classify_quality(result)

    # Score
    result.quality_score = _score_probe(result)

    return result


def _classify_quality(probe: ProbeResult) -> AudioQuality:
    """Classify audio quality based on probe measurements."""
    if probe.clipping_percent > 1.0:
        return AudioQuality.UNUSABLE_CLIPPING
    if probe.rms < 1.0 and probe.peak < 5.0:
        return AudioQuality.UNUSABLE_SILENCE
    # Check for pure noise: high RMS but no speech variation
    # Only flag as noise if noise floor is significant (not quiet room)
    if probe.noise_floor > 2.0 and probe.speech_rms < probe.noise_floor * 2:
        return AudioQuality.UNUSABLE_NOISE
    if probe.rms < 50:
        return AudioQuality.WEAK
    if probe.snr > 15 and probe.rms > 200:
        return AudioQuality.EXCELLENT
    if probe.snr > 8 and probe.rms > 100:
        return AudioQuality.GOOD
    if probe.rms > 30:
        return AudioQuality.NOISY
    return AudioQuality.WEAK


def _score_probe(probe: ProbeResult) -> float:
    """
    Multi-factor scoring system for device quality.

    Factors:
      1. Speech signal strength (median frame RMS of loudest frames)
      2. SNR (signal-to-noise ratio)
      3. Noise floor (lower is better)
      4. Clipping (none is best)
      5. Native 16kHz support
      6. API reliability
      7. Physical microphone likelihood
      8. Stable capture (multiple windows should agree)

    Returns: composite score (higher is better, 0 = unusable)
    """
    if not probe.can_open:
        return 0.0

    # Exclude non-microphone endpoints
    if not _is_real_microphone(probe.device_name):
        return 0.0

    # Exclude unusable quality
    if probe.quality in (
        AudioQuality.UNUSABLE_SILENCE,
        AudioQuality.UNUSABLE_NOISE,
        AudioQuality.UNUSABLE_CLIPPING,
    ):
        return 0.0

    # Reject devices where all frame RMS values are identical across windows
    # (indicates a single spike, not continuous audio capture)
    if len(probe.window_rms_values) > 1:
        rms_std = float(np.std(probe.window_rms_values))
        rms_mean = float(np.mean(probe.window_rms_values))
        if rms_mean > 100 and rms_std / rms_mean < 0.01:
            return 0.0

    # Base score: speech RMS (how loud is the actual speech signal)
    score = probe.speech_rms

    # SNR bonus: higher SNR means cleaner signal
    if probe.snr > 20:
        score *= 1.5
    elif probe.snr > 10:
        score *= 1.2
    elif probe.snr > 5:
        score *= 1.0
    else:
        score *= 0.7

    # Noise floor penalty: lower is better (but must be > 0)
    if probe.noise_floor > 50:
        score *= 0.6
    elif probe.noise_floor > 20:
        score *= 0.8

    # Clipping penalty
    if probe.clipping_percent > 0.1:
        score *= 0.5
    elif probe.clipping_percent > 0.01:
        score *= 0.8

    # Native 16kHz bonus (avoids resampling)
    if probe.capture_sample_rate == TARGET_SAMPLE_RATE:
        score *= 1.2

    # API preference
    api = probe.host_api
    if "DirectSound" in api:
        score *= 1.1
    elif "WDM-KS" in api:
        score *= 1.15
    elif api == "MME":
        score *= 0.8
    elif "WASAPI" in api:
        score *= 1.05

    # WEAK quality penalty
    if probe.quality == AudioQuality.WEAK:
        score *= 0.3
    elif probe.quality == AudioQuality.NOISY:
        score *= 0.7

    return score


# ---------------------------------------------------------------------------
# AudioDeviceManager — the canonical device selector
# ---------------------------------------------------------------------------

class AudioDeviceManager:
    """
    Single source of truth for microphone device selection.

    Usage:
        manager = AudioDeviceManager()
        selection = manager.select_best_device()
        # selection is a DeviceSelection with all measurements

    Both --mic-test and VoiceEngine MUST use this class.
    """

    def __init__(
        self,
        configured_device: Optional[int] = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        probe_duration: float = 1.0,
        num_probe_windows: int = 3,
    ) -> None:
        self._configured_device = configured_device
        self._target_sr = target_sample_rate
        self._probe_duration = probe_duration
        self._num_windows = num_probe_windows
        self._all_probes: list[ProbeResult] = []

    @property
    def all_probes(self) -> list[ProbeResult]:
        """All probe results from the last selection run."""
        return self._all_probes

    def enumerate_devices(self) -> list[dict]:
        """
        Enumerate all input devices with host API info.
        Returns list of dicts with device info.
        """
        try:
            import sounddevice as sd
        except ImportError:
            return []

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        result = []
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                api_idx = dev.get("hostapi", 0)
                api_name = host_apis[api_idx]["name"] if 0 <= api_idx < len(host_apis) else "Unknown"
                result.append({
                    "index": i,
                    "name": dev.get("name", f"Device {i}"),
                    "host_api": api_name,
                    "max_channels": dev.get("max_input_channels", 0),
                    "native_sample_rate": dev.get("default_samplerate", 0),
                    "is_real_mic": _is_real_microphone(dev.get("name", "")),
                })
        return result

    def probe_device(self, device_index: int) -> ProbeResult:
        """
        Probe a single device with multiple capture windows.
        Returns a ProbeResult with robust measurements.
        """
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

        devices = sd.query_devices()
        host_apis = sd.query_hostapis()

        if device_index >= len(devices):
            return ProbeResult(
                device_index=device_index,
                device_name="unknown",
                host_api="unknown",
                max_channels=0,
                native_sample_rate=0,
                error=f"Device index {device_index} out of range",
            )

        dev = devices[device_index]
        api_idx = dev.get("hostapi", 0)
        api_name = host_apis[api_idx]["name"] if 0 <= api_idx < len(host_apis) else "Unknown"

        return _probe_device_robust(
            dev_idx=device_index,
            dev_info=dev,
            host_api_name=api_name,
            target_sr=self._target_sr,
            num_windows=self._num_windows,
            window_duration=self._probe_duration,
        )

    def select_best_device(self) -> Optional[DeviceSelection]:
        """
        Select the best microphone device.

        Algorithm:
          1. Check PENGU_MIC_DEVICE env override
          2. Enumerate all input devices
          3. Probe each with multiple capture windows
          4. Score using multi-factor system
          5. Return the device with highest score

        Returns:
          DeviceSelection with all measurements, or None if no device found.
        """
        # 1. Environment override
        env_device = os.environ.get("PENGU_MIC_DEVICE")
        if env_device:
            try:
                dev_idx = int(env_device)
                probe = self.probe_device(dev_idx)
                if probe.can_open:
                    self._all_probes = [probe]
                    return self._probe_to_selection(probe, f"Selected via PENGU_MIC_DEVICE={env_device}")
            except Exception as e:
                logger.warning("env_device_failed", device=env_device, error=str(e))

        # 2. Enumerate all input devices
        devices = self.enumerate_devices()
        if not devices:
            logger.error("no_input_devices")
            return None

        logger.info("mic_enumerated", count=len(devices))

        # 3. Probe each device
        probes: list[ProbeResult] = []
        for dev in devices:
            probe = self.probe_device(dev["index"])
            probes.append(probe)
            if probe.can_open:
                logger.info(
                    "mic_probed",
                    device=probe.device_index,
                    name=probe.device_name,
                    api=probe.host_api,
                    rms=round(probe.rms, 1),
                    speech_rms=round(probe.speech_rms, 1),
                    snr=round(probe.snr, 1),
                    score=round(probe.quality_score, 1),
                    quality=probe.quality.value,
                )
            else:
                logger.debug("mic_probe_failed", device=probe.device_index, error=probe.error)

        self._all_probes = probes

        # 4. Filter to usable devices
        usable = [p for p in probes if p.can_open and p.quality_score > 0]

        if not usable:
            logger.error("no_usable_microphone")
            return None

        # 5. Pick the best device by score
        usable.sort(key=lambda p: p.quality_score, reverse=True)
        best = usable[0]

        # Log alternatives
        alternatives = [(p.device_index, p.device_name, round(p.quality_score, 1)) for p in usable[1:5]]
        logger.info(
            "mic_selected",
            device=best.device_index,
            name=best.device_name,
            api=best.host_api,
            rms=round(best.rms, 1),
            speech_rms=round(best.speech_rms, 1),
            snr=round(best.snr, 1),
            score=round(best.quality_score, 1),
            quality=best.quality.value,
            sr=best.capture_sample_rate,
            alternatives=alternatives,
        )

        return self._probe_to_selection(
            best,
            f"Highest quality score ({best.quality_score:.1f}) among {len(usable)} usable devices",
        )

    def _probe_to_selection(self, probe: ProbeResult, reason: str) -> DeviceSelection:
        """Convert a ProbeResult to a DeviceSelection."""
        return DeviceSelection(
            device_index=probe.device_index,
            device_name=probe.device_name,
            host_api=probe.host_api,
            max_channels=probe.max_channels,
            native_sample_rate=probe.native_sample_rate,
            capture_sample_rate=probe.capture_sample_rate,
            capture_channels=probe.capture_channels,
            rms=probe.rms,
            peak=probe.peak,
            noise_floor=probe.noise_floor,
            speech_rms=probe.speech_rms,
            snr=probe.snr,
            clipping_percent=probe.clipping_percent,
            quality=probe.quality,
            quality_score=probe.quality_score,
            selection_reason=reason,
            probe_duration_seconds=probe.capture_time_seconds,
            probe_success=probe.can_open,
            probe_error=probe.error,
            all_probes=[
                {
                    "device": p.device_index,
                    "name": p.device_name,
                    "api": p.host_api,
                    "rms": round(p.rms, 1),
                    "speech_rms": round(p.speech_rms, 1),
                    "snr": round(p.snr, 1),
                    "score": round(p.quality_score, 1),
                    "quality": p.quality.value,
                }
                for p in self._all_probes
                if p.can_open
            ],
        )

    def compare_devices(self, device_a: int, device_b: int, num_rounds: int = 5) -> dict:
        """
        Compare two devices across multiple rounds for robustness.
        Returns a comparison report.
        """
        results_a = []
        results_b = []

        for _ in range(num_rounds):
            probe_a = self.probe_device(device_a)
            probe_b = self.probe_device(device_b)
            if probe_a.can_open:
                results_a.append(probe_a)
            if probe_b.can_open:
                results_b.append(probe_b)

        def _summarize(probes: list[ProbeResult]) -> dict:
            if not probes:
                return {"error": "No successful captures"}
            return {
                "successful_captures": len(probes),
                "rms_values": [round(p.rms, 1) for p in probes],
                "rms_median": round(float(np.median([p.rms for p in probes])), 1),
                "speech_rms_values": [round(p.speech_rms, 1) for p in probes],
                "speech_rms_median": round(float(np.median([p.speech_rms for p in probes])), 1),
                "snr_values": [round(p.snr, 1) for p in probes],
                "snr_median": round(float(np.median([p.snr for p in probes])), 1),
                "peak_values": [round(p.peak, 1) for p in probes],
                "peak_median": round(float(np.median([p.peak for p in probes])), 1),
                "noise_floor_values": [round(p.noise_floor, 1) for p in probes],
                "noise_floor_median": round(float(np.median([p.noise_floor for p in probes])), 1),
                "clipping_values": [round(p.clipping_percent, 4) for p in probes],
                "quality": probes[0].quality.value if probes else "unknown",
                "score_values": [round(p.quality_score, 1) for p in probes],
                "score_median": round(float(np.median([p.quality_score for p in probes])), 1),
            }

        return {
            f"device_{device_a}": _summarize(results_a),
            f"device_{device_b}": _summarize(results_b),
            "winner": (
                f"device_{device_a}" if (
                    results_a and results_b and
                    float(np.median([p.quality_score for p in results_a])) >
                    float(np.median([p.quality_score for p in results_b]))
                ) else f"device_{device_b}" if results_b else f"device_{device_a}"
            ),
        }
