"""
AudioDeviceManager — single source of truth for microphone device selection.

Both --mic-test and VoiceEngine.initialize() MUST use this class.

Architecture:
  Phase 1: Silent baseline capture (1s) -> noise floor
  Phase 2: User speech capture (2s) -> speech RMS, SNR
  Phase 3: Post-speech silence (1s) -> verify noise floor returns

Only devices where the user ACTUALLY SPOKE and SNR > 0 are considered READY.

SNR is calculated as:
  SNR_dB = 20 * log10(speech_rms / noise_floor)
  with noise_floor > 0 guard (returns -inf or 0 for degenerate cases)
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

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
MIN_SNR_DB = 6.0         # Minimum SNR for ACCEPTABLE quality
MIN_SPEECH_RMS = 100.0    # Minimum speech RMS for usable voice capture
MIN_NOISE_FLOOR = 0.1     # Minimum noise floor to calculate SNR


# ---------------------------------------------------------------------------
# Quality levels — objective, threshold-based
# ---------------------------------------------------------------------------

class AudioQuality(str, Enum):
    """
    Objective quality states based on measured data.

    UNUSABLE:  Device cannot capture audio or signal is absent
    POOR:      Device captures audio but quality is too low for voice recognition
    ACCEPTABLE: Device captures voice with marginal quality
    GOOD:      Device captures voice reliably
    EXCELLENT: Device captures voice with high fidelity
    """
    UNUSABLE = "UNUSABLE"
    POOR = "POOR"
    ACCEPTABLE = "ACCEPTABLE"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"

    @property
    def is_voice_ready(self) -> bool:
        return self in (AudioQuality.ACCEPTABLE, AudioQuality.GOOD, AudioQuality.EXCELLENT)


# ---------------------------------------------------------------------------
# Structured device result
# ---------------------------------------------------------------------------

@dataclass
class DeviceSelection:
    """Complete result of device selection — the single structured object."""
    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float
    capture_sample_rate: int = TARGET_SAMPLE_RATE
    capture_channels: int = TARGET_CHANNELS
    # Baseline (silence) measurements
    noise_floor: float = 0.0
    noise_rms: float = 0.0
    # Speech measurements
    speech_rms: float = 0.0
    speech_peak: float = 0.0
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
    # All probed devices for comparison
    all_probes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "device_index": self.device_index,
            "device_name": self.device_name,
            "host_api": self.host_api,
            "capture_sample_rate": self.capture_sample_rate,
            "capture_channels": self.capture_channels,
            "noise_floor": round(self.noise_floor, 1),
            "speech_rms": round(self.speech_rms, 1),
            "snr_db": round(self.snr_db, 1),
            "clipping_percent": round(self.clipping_percent, 4),
            "quality": self.quality.value,
            "quality_score": round(self.quality_score, 1),
            "voice_ready": self.voice_ready,
            "speech_detected": self.speech_detected,
        }


# ---------------------------------------------------------------------------
# Non-microphone filtering
# ---------------------------------------------------------------------------

NON_MICROPHONE_PATTERNS = [
    "stereo mix", "loopback", "wave out",
    "what u hear", "monitor",
]

NON_MICROPHONE_DEVICE_NAMES = [
    "pc speaker",
    "sound mapper",
]


def _is_real_microphone(name: str) -> bool:
    """
    Filter out non-microphone devices.

    Note: 'Primary Sound Capture Driver' is a valid DirectSound endpoint,
    not a speaker output. 'Microsoft Sound Mapper' is an MME abstraction
    that wraps the default device.
    """
    name_lower = name.lower()
    for pattern in NON_MICROPHONE_PATTERNS:
        if pattern in name_lower:
            return False
    for pattern in NON_MICROPHONE_DEVICE_NAMES:
        if pattern in name_lower:
            return False
    return True





# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _measure_frame_rms(audio: np.ndarray, frame_size: int = 1024) -> list[float]:
    flat = audio.flatten().astype(np.float32)
    return [float(np.sqrt(np.mean(flat[i:i+frame_size] ** 2)))
            for i in range(0, len(flat) - frame_size, frame_size)]


def _measure_clipping_percent(audio: np.ndarray, threshold: float = 32000) -> float:
    flat = audio.flatten().astype(np.float32)
    clipped = np.sum(np.abs(flat) >= threshold)
    return float(clipped / len(flat) * 100) if len(flat) > 0 else 0.0


def _calculate_snr_db(speech_rms: float, noise_floor: float) -> float:
    """
    Calculate SNR in dB: 20 * log10(speech_rms / noise_floor)

    Returns 0.0 for degenerate cases (noise_floor <= 0 or speech_rms <= 0).
    """
    if noise_floor < MIN_NOISE_FLOOR or speech_rms <= 0:
        return 0.0
    ratio = speech_rms / noise_floor
    if ratio <= 0:
        return 0.0
    return float(20 * np.log10(ratio))


def _classify_quality(snr_db: float, speech_rms: float, clipping_pct: float,
                      noise_floor: float, speech_detected: bool) -> AudioQuality:
    """
    Classify quality based on objective thresholds.

    A device is only ACCEPTABLE or better if:
      - speech was actually detected (user spoke)
      - SNR > 6 dB
      - speech_rms > 100
      - no excessive clipping
    """
    if clipping_pct > 1.0:
        return AudioQuality.UNUSABLE
    if not speech_detected:
        return AudioQuality.POOR
    if speech_rms < 1.0 and noise_floor < 1.0:
        return AudioQuality.UNUSABLE
    if snr_db < MIN_SNR_DB or speech_rms < MIN_SPEECH_RMS:
        return AudioQuality.POOR
    if snr_db >= 20 and speech_rms >= 500:
        return AudioQuality.EXCELLENT
    if snr_db >= 12 and speech_rms >= 200:
        return AudioQuality.GOOD
    return AudioQuality.ACCEPTABLE


def _score_device(snr_db: float, speech_rms: float, noise_floor: float,
                  clipping_pct: float, quality: AudioQuality,
                  host_api: str, capture_sr: int, device_name: str) -> float:
    """Multi-factor composite score. Higher is better. 0 = unusable."""
    if quality == AudioQuality.UNUSABLE:
        return 0.0
    if not _is_real_microphone(device_name):
        return 0.0

    score = speech_rms

    # SNR is the most important factor
    if snr_db >= 20:
        score *= 2.0
    elif snr_db >= 12:
        score *= 1.5
    elif snr_db >= MIN_SNR_DB:
        score *= 1.0
    else:
        score *= 0.3

    # Clipping penalty
    if clipping_pct > 0.1:
        score *= 0.3
    elif clipping_pct > 0.01:
        score *= 0.7

    # API preference
    if "DirectSound" in host_api:
        score *= 1.1
    elif "WDM-KS" in host_api:
        score *= 1.15
    elif host_api == "MME":
        score *= 0.8

    # Native 16kHz bonus
    if capture_sr == TARGET_SAMPLE_RATE:
        score *= 1.2

    # Quality multiplier
    quality_mult = {
        AudioQuality.EXCELLENT: 1.5,
        AudioQuality.GOOD: 1.2,
        AudioQuality.ACCEPTABLE: 1.0,
        AudioQuality.POOR: 0.4,
        AudioQuality.UNUSABLE: 0.0,
    }
    score *= quality_mult.get(quality, 0.0)

    return score


# ---------------------------------------------------------------------------
# Audio preprocessing utilities (used by engine.py)
# ---------------------------------------------------------------------------

def _downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging all channels."""
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 1:
        return audio.flatten()
    # Average all channels
    return np.mean(audio, axis=1).astype(audio.dtype)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    flat = audio.flatten().astype(np.float32)
    duration = len(flat) / orig_sr
    new_length = int(duration * target_sr)
    if new_length <= 0:
        return audio
    # Linear interpolation
    indices = np.linspace(0, len(flat) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(flat)), flat)
    return resampled.astype(audio.dtype)


# ---------------------------------------------------------------------------
# Two-phase device probe: SILENCE -> SPEECH
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of a two-phase probe (silence + speech)."""
    device_index: int
    device_name: str
    host_api: str
    max_channels: int
    native_sample_rate: float
    capture_sample_rate: int = 0
    capture_channels: int = 0
    # Phase 1: silence baseline
    noise_floor: float = 0.0
    noise_rms: float = 0.0
    # Phase 2: speech
    speech_rms: float = 0.0
    speech_peak: float = 0.0
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
    # Multiple rounds for robustness
    round_snr_values: list[float] = field(default_factory=list)
    round_speech_rms_values: list[float] = field(default_factory=list)


def _probe_device_two_phase(
    dev_idx: int,
    dev_info: dict,
    host_api_name: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    num_rounds: int = 3,
    silence_duration: float = 1.0,
    speech_duration: float = 2.0,
    prompt_user: bool = False,
) -> ProbeResult:
    """
    Two-phase probe: silence baseline -> user speech.

    Phase 1: Capture silence (1s) -> noise floor
    Phase 2: Prompt user to speak (2s) -> speech RMS, SNR

    If prompt_user=False (automated testing), captures whatever is available.
    If prompt_user=True (interactive --mic-test), prompts the user.
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
    try:
        sd.check_input_settings(device=dev_idx, samplerate=target_sr, channels=1, dtype="int16")
    except Exception:
        native_sr = int(dev_info.get("default_samplerate", target_sr))
        try:
            sd.check_input_settings(device=dev_idx, samplerate=native_sr, channels=1, dtype="int16")
            capture_sr = native_sr
        except Exception as e:
            result.error = f"Cannot open device: {e}"
            return result

    result.capture_sample_rate = capture_sr
    result.capture_channels = 1

    round_snrs = []
    round_speech = []

    for round_num in range(num_rounds):
        # Phase 1: Silence baseline
        try:
            silence_audio = sd.rec(
                int(silence_duration * capture_sr),
                samplerate=capture_sr, channels=1, dtype="int16", device=dev_idx,
            )
            sd.wait()
            if silence_audio is None or silence_audio.size == 0:
                continue

            frame_rms = _measure_frame_rms(silence_audio)
            if not frame_rms:
                continue

            sorted_rms = sorted(frame_rms)
            noise_floor = float(np.mean(sorted_rms[max(0, len(sorted_rms)//4):]))  # median of lower half
            noise_rms = float(np.sqrt(np.mean(silence_audio.flatten().astype(np.float32) ** 2)))

        except Exception as e:
            logger.debug("probe_silence_failed", device=dev_idx, round=round_num, error=str(e))
            continue

        # Phase 2: Speech capture
        if prompt_user:
            # Interactive mode — tell user to speak
            sys.stdout.write(f"\r  Round {round_num+1}/{num_rounds}: Speak now (say 'Pengu microphone test')...")
            sys.stdout.flush()
            time.sleep(0.3)  # Brief pause before recording

        try:
            speech_audio = sd.rec(
                int(speech_duration * capture_sr),
                samplerate=capture_sr, channels=1, dtype="int16", device=dev_idx,
            )
            sd.wait()
            if speech_audio is None or speech_audio.size == 0:
                continue

            speech_flat = speech_audio.flatten().astype(np.float32)
            speech_rms = float(np.sqrt(np.mean(speech_flat ** 2)))
            speech_peak = float(np.max(np.abs(speech_flat)))

            # Check if speech was actually detected (RMS significantly above noise floor)
            speech_detected = speech_rms > noise_floor * 3 and speech_rms > MIN_SPEECH_RMS * 0.3

            # SNR calculation
            snr_db = _calculate_snr_db(speech_rms, noise_floor) if speech_detected else 0.0

            # Clipping
            clipping = _measure_clipping_percent(speech_audio)

            round_snrs.append(snr_db)
            round_speech.append(speech_rms)

            if round_num == 0:
                # Store first round as primary values
                result.noise_floor = noise_floor
                result.noise_rms = noise_rms
                result.speech_rms = speech_rms
                result.speech_peak = speech_peak
                result.snr_db = snr_db
                result.clipping_percent = clipping
                result.speech_detected = speech_detected
                result.overall_rms = float(np.sqrt(np.mean(speech_flat ** 2)))
                result.overall_peak = speech_peak

        except Exception as e:
            logger.debug("probe_speech_failed", device=dev_idx, round=round_num, error=str(e))
            continue

        if prompt_user:
            sys.stdout.write(f"\r  Round {round_num+1}/{num_rounds}: SNR={snr_db:.1f}dB  Speech RMS={speech_rms:.0f} {'OK' if speech_detected else 'NO SPEECH'}     \n")
            sys.stdout.flush()

    if not round_speech:
        result.error = "All probe rounds failed"
        return result

    result.can_open = True

    # Robust statistics across rounds
    result.round_snr_values = round_snrs
    result.round_speech_rms_values = round_speech

    if round_snrs:
        result.snr_db = float(np.median(round_snrs))
    if round_speech:
        result.speech_rms = float(np.median(round_speech))
        result.speech_detected = any(s > MIN_SPEECH_RMS * 0.3 for s in round_speech)

    # Classify quality
    result.quality = _classify_quality(
        result.snr_db, result.speech_rms, result.clipping_percent,
        result.noise_floor, result.speech_detected,
    )

    # Score
    result.quality_score = _score_device(
        result.snr_db, result.speech_rms, result.noise_floor,
        result.clipping_percent, result.quality,
        host_api_name, capture_sr, result.device_name,
    )

    return result


# ---------------------------------------------------------------------------
# AudioDeviceManager — the canonical selector
# ---------------------------------------------------------------------------

class AudioDeviceManager:
    """
    Single source of truth for microphone device selection.

    Both --mic-test and VoiceEngine MUST use this class.
    """

    def __init__(
        self,
        configured_device: Optional[int] = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        num_rounds: int = 3,
        prompt_user: bool = False,
    ) -> None:
        self._configured_device = configured_device
        self._target_sr = target_sample_rate
        self._num_rounds = num_rounds
        self._prompt_user = prompt_user
        self._all_probes: list[ProbeResult] = []

    @property
    def all_probes(self) -> list[ProbeResult]:
        return self._all_probes

    def enumerate_devices(self) -> list[dict]:
        """Enumerate all input devices with host API info."""
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

    def probe_device(self, device_index: int, prompt_user: bool = False) -> ProbeResult:
        """Probe a single device with two-phase capture."""
        try:
            import sounddevice as sd
        except ImportError:
            return ProbeResult(
                device_index=device_index, device_name="unknown", host_api="unknown",
                max_channels=0, native_sample_rate=0, error="sounddevice not installed",
            )
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        if device_index >= len(devices):
            return ProbeResult(
                device_index=device_index, device_name="unknown", host_api="unknown",
                max_channels=0, native_sample_rate=0, error=f"Device {device_index} out of range",
            )
        dev = devices[device_index]
        api_idx = dev.get("hostapi", 0)
        api_name = host_apis[api_idx]["name"] if 0 <= api_idx < len(host_apis) else "Unknown"
        return _probe_device_two_phase(
            dev_idx=device_index, dev_info=dev, host_api_name=api_name,
            target_sr=self._target_sr, num_rounds=self._num_rounds,
            prompt_user=prompt_user,
        )

    def select_best_device(self) -> Optional[DeviceSelection]:
        """Select the best microphone device using two-phase probing."""
        # Environment override
        env_device = os.environ.get("PENGU_MIC_DEVICE")
        if env_device:
            try:
                dev_idx = int(env_device)
                probe = self.probe_device(dev_idx)
                if probe.can_open:
                    self._all_probes = [probe]
                    return self._probe_to_selection(probe, f"PENGU_MIC_DEVICE={env_device}")
            except Exception as e:
                logger.warning("env_device_failed", device=env_device, error=str(e))

        # Enumerate
        devices = self.enumerate_devices()
        if not devices:
            logger.error("no_input_devices")
            return None

        logger.info("mic_enumerated", count=len(devices))

        # Probe each device with two-phase capture
        if self._prompt_user:
            print(f"\n  Probing {len(devices)} devices...")
            print(f"  For each device: 1s silence + 2s your speech")
            print(f"  Say 'Pengu microphone test' when prompted.\n")

        probes: list[ProbeResult] = []
        for dev in devices:
            probe = self.probe_device(dev["index"], prompt_user=self._prompt_user)
            probes.append(probe)
            if probe.can_open:
                logger.info(
                    "mic_probed",
                    device=probe.device_index, name=probe.device_name, api=probe.host_api,
                    noise=round(probe.noise_floor, 1), speech=round(probe.speech_rms, 1),
                    snr=round(probe.snr_db, 1), quality=probe.quality.value,
                    score=round(probe.quality_score, 1),
                    speech_detected=probe.speech_detected,
                )
            else:
                logger.debug("mic_probe_failed", device=probe.device_index, error=probe.error)

        self._all_probes = probes

        # Filter to voice-ready devices
        usable = [p for p in probes if p.can_open and p.quality_score > 0 and p.speech_detected]

        if not usable:
            # Try devices with any positive score (even without speech)
            usable = [p for p in probes if p.can_open and p.quality_score > 0]

        if not usable:
            logger.error("no_usable_microphone")
            return None

        # Pick best by score
        usable.sort(key=lambda p: p.quality_score, reverse=True)
        best = usable[0]

        reason = f"Score {best.quality_score:.1f}, SNR={best.snr_db:.1f}dB"
        if best.speech_detected:
            reason += ", speech detected"
        else:
            reason += ", NO SPEECH DETECTED (auto-selection)"

        logger.info(
            "mic_selected",
            device=best.device_index, name=best.device_name, api=best.host_api,
            noise=round(best.noise_floor, 1), speech=round(best.speech_rms, 1),
            snr=round(best.snr_db, 1), quality=best.quality.value,
            score=round(best.quality_score, 1),
        )

        return self._probe_to_selection(best, reason)

    def _probe_to_selection(self, probe: ProbeResult, reason: str) -> DeviceSelection:
        return DeviceSelection(
            device_index=probe.device_index,
            device_name=probe.device_name,
            host_api=probe.host_api,
            max_channels=probe.max_channels,
            native_sample_rate=probe.native_sample_rate,
            capture_sample_rate=probe.capture_sample_rate,
            capture_channels=probe.capture_channels,
            noise_floor=probe.noise_floor,
            noise_rms=probe.noise_rms,
            speech_rms=probe.speech_rms,
            speech_peak=probe.speech_peak,
            rms=probe.overall_rms,
            peak=probe.overall_peak,
            snr_db=probe.snr_db,
            clipping_percent=probe.clipping_percent,
            quality=probe.quality,
            quality_score=probe.quality_score,
            selection_reason=reason,
            voice_ready=probe.quality.is_voice_ready,
            speech_detected=probe.speech_detected,
            all_probes=[
                {
                    "device": p.device_index, "name": p.device_name, "api": p.host_api,
                    "noise": round(p.noise_floor, 1), "speech": round(p.speech_rms, 1),
                    "snr": round(p.snr_db, 1), "score": round(p.quality_score, 1),
                    "quality": p.quality.value, "speech_detected": p.speech_detected,
                }
                for p in self._all_probes if p.can_open
            ],
        )
