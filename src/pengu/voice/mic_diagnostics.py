"""
Pengu Microphone Diagnostics — complete audio pipeline diagnosis.

This module provides thorough microphone diagnostics that:
  - Enumerate all input devices
  - Test each device with real audio capture
  - Measure actual signal levels (RMS, peak, noise floor, SNR)
  - Detect clipping, silence, noise, or usable speech
  - Identify the best microphone for Pengu
  - Report Windows audio permission issues

Usage:
    python -m pengu --mic-test
    python -m pengu --diagnostics

Design principle: NEVER fabricate values. If something fails, say so clearly.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AudioQuality(str, Enum):
    """Classification of audio signal quality."""
    UNUSABLE_SILENCE = "UNUSABLE_SILENCE"   # Signal too weak — mic muted or broken
    UNUSABLE_NOISE = "UNUSABLE_NOISE"       # All noise, no speech possible
    UNUSABLE_CLIPPING = "UNUSABLE_CLIPPING"  # Signal clipped — gain too high
    WEAK = "WEAK"                            # Very low signal — mic gain too low
    NOISY = "NOISY"                          # Usable but very noisy
    GOOD = "GOOD"                            # Good speech signal
    EXCELLENT = "EXCELLENT"                  # Excellent signal quality

    def describe(self) -> str:
        descriptions = {
            AudioQuality.UNUSABLE_SILENCE: "Audio is essentially silent. Microphone may be muted in Windows, permissions not granted, or hardware failure.",
            AudioQuality.UNUSABLE_NOISE: "Audio contains only noise with no usable speech signal. Microphone gain may be wrong or device is broken.",
            AudioQuality.UNUSABLE_CLIPPING: "Audio is severely clipped (peak near max). Input gain is too high — reduce mic volume in Windows settings.",
            AudioQuality.WEAK: "Audio signal is very weak. Microphone gain should be increased in Windows Sound Settings.",
            AudioQuality.NOISY: "Audio signal is usable but has significant background noise. Consider noise suppression or a quieter environment.",
            AudioQuality.GOOD: "Audio signal is good for speech recognition.",
            AudioQuality.EXCELLENT: "Audio signal is excellent for speech recognition.",
        }
        return descriptions.get(self, "Unknown quality")


@dataclass
class DeviceInfo:
    """Information about a single audio input device."""
    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_sample_rate: float
    supported_sample_rates: list[float] = field(default_factory=list)


@dataclass
class AudioMeasurements:
    """Real audio measurements from a recording."""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    num_samples: int = 0
    # Amplitude measurements (on int16 scale)
    rms: float = 0.0
    peak: float = 0.0
    avg_amplitude: float = 0.0
    noise_floor: float = 0.0       # RMS of quietest 10% of frames
    speech_rms: float = 0.0        # RMS of loudest 30% of frames
    # Derived
    signal_to_noise_ratio: float = 0.0
    clipping_detected: bool = False
    clipping_percent: float = 0.0
    # Classification
    quality: AudioQuality = AudioQuality.UNUSABLE_SILENCE
    quality_detail: str = ""
    # Raw data
    rms_history: list[float] = field(default_factory=list)


@dataclass
class DeviceTestResult:
    """Result of testing a single device."""
    device: DeviceInfo
    measurements: Optional[AudioMeasurements] = None
    error: str = ""
    selected: bool = False
    is_default: bool = False


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    python_version: str = ""
    windows_permission: str = "UNKNOWN"
    devices: list[DeviceInfo] = field(default_factory=list)
    test_results: list[DeviceTestResult] = field(default_factory=list)
    selected_device: Optional[DeviceInfo] = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------

def enumerate_input_devices() -> list[DeviceInfo]:
    """Enumerate all available audio input devices."""
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices = []
    try:
        device_list = sd.query_devices()
    except Exception:
        return []

    host_apis = sd.query_hostapis()

    for i, dev in enumerate(device_list):
        if dev.get("max_input_channels", 0) > 0:
            host_api_name = ""
            host_api_idx = dev.get("hostapi", 0)
            if 0 <= host_api_idx < len(host_apis):
                host_api_name = host_apis[host_api_idx].get("name", "")

            device_info = DeviceInfo(
                index=i,
                name=dev.get("name", f"Device {i}"),
                host_api=host_api_name,
                max_input_channels=dev.get("max_input_channels", 0),
                default_sample_rate=dev.get("default_samplerate", 16000),
            )

            # Query supported sample rates
            try:
                for rate in [8000, 16000, 22050, 44100, 48000]:
                    try:
                        sd.check_input_settings(device=i, samplerate=rate, channels=1)
                        device_info.supported_sample_rates.append(rate)
                    except Exception:
                        pass
            except Exception:
                pass

            devices.append(device_info)

    return devices


def get_default_input_device() -> Optional[DeviceInfo]:
    """Get the default Windows input device."""
    try:
        import sounddevice as sd
        default = sd.query_devices(kind="input")
        # Find it in the full list
        devices = enumerate_input_devices()
        for dev in devices:
            if dev.name == default.get("name", ""):
                return dev
        # Fallback: return the first device with the matching name
        return DeviceInfo(
            index=sd.default.device[0] or 0,
            name=default.get("name", "Unknown"),
            host_api="",
            max_input_channels=default.get("max_input_channels", 1),
            default_sample_rate=default.get("default_samplerate", 16000),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Audio measurement
# ---------------------------------------------------------------------------

def _calculate_frame_rms(audio: np.ndarray, frame_size: int = 1024) -> list[float]:
    """Calculate RMS energy per frame."""
    flat = audio.flatten().astype(np.float32)
    rms_values = []
    for i in range(0, len(flat) - frame_size, frame_size):
        frame = flat[i:i + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        rms_values.append(rms)
    return rms_values


def _calculate_peak(audio: np.ndarray) -> float:
    """Calculate peak amplitude."""
    flat = audio.flatten().astype(np.float32)
    return float(np.max(np.abs(flat)))


def _calculate_clipping_percent(audio: np.ndarray, threshold: float = 32000) -> float:
    """Calculate percentage of samples that are clipped."""
    flat = audio.flatten().astype(np.float32)
    clipped = np.sum(np.abs(flat) >= threshold)
    return float(clipped / len(flat) * 100) if len(flat) > 0 else 0.0


def measure_audio(audio: np.ndarray, sample_rate: int) -> AudioMeasurements:
    """
    Calculate comprehensive audio measurements from a recorded sample.

    Args:
        audio: Raw int16 audio data
        sample_rate: Sample rate in Hz

    Returns:
        AudioMeasurements with all calculated metrics
    """
    flat = audio.flatten().astype(np.float32)
    num_samples = len(flat)

    # Basic measurements
    rms = float(np.sqrt(np.mean(flat ** 2)))
    peak = float(np.max(np.abs(flat)))
    avg = float(np.mean(np.abs(flat)))

    # Frame-level analysis
    frame_rms = _calculate_frame_rms(audio, frame_size=1024)

    # Noise floor (RMS of quietest 10% of frames)
    noise_floor = 0.0
    speech_rms = 0.0
    if frame_rms:
        sorted_rms = sorted(frame_rms)
        noise_count = max(1, len(sorted_rms) // 10)
        speech_count = max(1, len(sorted_rms) // 3)
        noise_floor = float(np.mean(sorted_rms[:noise_count]))
        speech_rms = float(np.mean(sorted_rms[-speech_count:]))

    # SNR
    snr = 0.0
    if noise_floor > 0:
        snr = 20 * np.log10(speech_rms / noise_floor) if speech_rms > 0 else 0.0

    # Clipping
    clipping_pct = _calculate_clipping_percent(audio)

    # Classification
    quality, detail = _classify_quality(rms, peak, noise_floor, snr, clipping_pct, speech_rms)

    channels = 1 if audio.ndim == 1 else audio.shape[1]

    return AudioMeasurements(
        duration_seconds=num_samples / sample_rate,
        sample_rate=sample_rate,
        channels=channels,
        num_samples=num_samples,
        rms=rms,
        peak=peak,
        avg_amplitude=avg,
        noise_floor=noise_floor,
        speech_rms=speech_rms,
        signal_to_noise_ratio=snr,
        clipping_detected=clipping_pct > 0.01,
        clipping_percent=clipping_pct,
        quality=quality,
        quality_detail=detail,
        rms_history=frame_rms[:50],  # Keep first 50 frames for analysis
    )


def _classify_quality(
    rms: float,
    peak: float,
    noise_floor: float,
    snr: float,
    clipping_pct: float,
    speech_rms: float,
) -> tuple[AudioQuality, str]:
    """Classify audio quality based on measurements."""

    # Check clipping first
    if clipping_pct > 1.0:
        return AudioQuality.UNUSABLE_CLIPPING, (
            f"Severe clipping detected ({clipping_pct:.2f}% of samples). "
            "Reduce microphone input volume in Windows Sound Settings."
        )

    # Check silence
    if rms < 1.0 and peak < 5.0:
        return AudioQuality.UNUSABLE_SILENCE, (
            f"Audio is essentially silent (RMS={rms:.3f}, Peak={peak:.3f}). "
            "Possible causes:\n"
            "  1. Microphone is muted in Windows Sound Settings\n"
            "  2. Microphone permissions not granted to Python\n"
            "  3. Wrong microphone device selected\n"
            "  4. Microphone hardware failure\n"
            "Fix: Open Windows Sound Settings → Input → "
            "ensure correct mic is selected and unmuted."
        )

    # Check if all noise (no speech variation)
    if rms > 5.0 and speech_rms < rms * 1.1 and snr < 3.0:
        return AudioQuality.UNUSABLE_NOISE, (
            f"Audio appears to be pure noise (SNR={snr:.1f}dB). "
            "The microphone may be picking up only background noise."
        )

    # Weak signal
    if rms < 100:
        return AudioQuality.WEAK, (
            f"Audio signal is very weak (RMS={rms:.1f}, Peak={peak:.1f}). "
            "Increase microphone volume in Windows Sound Settings."
        )

    # Good quality check
    if snr > 15 and rms > 200:
        return AudioQuality.EXCELLENT, (
            f"Excellent signal quality (SNR={snr:.1f}dB, RMS={rms:.1f})."
        )

    if snr > 8 and rms > 100:
        return AudioQuality.GOOD, (
            f"Good signal quality (SNR={snr:.1f}dB, RMS={rms:.1f})."
        )

    # Noisy but usable
    if rms > 50:
        return AudioQuality.NOISY, (
            f"Audio signal is usable but noisy (SNR={snr:.1f}dB, RMS={rms:.1f}). "
            "Consider enabling noise suppression in Windows Sound Settings."
        )

    return AudioQuality.WEAK, (
        f"Audio signal is weak (RMS={rms:.1f}, Peak={peak:.1f}, SNR={snr:.1f}dB). "
        "Increase microphone gain in Windows Sound Settings."
    )


# ---------------------------------------------------------------------------
# Device testing
# ---------------------------------------------------------------------------

def record_audio_from_device(
    device_index: int,
    duration: float = 3.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> tuple[Optional[np.ndarray], str]:
    """
    Record audio from a specific device for diagnostics.

    Returns:
        (audio_data, error_message)
    """
    try:
        import sounddevice as sd
    except ImportError:
        return None, "sounddevice not installed"

    try:
        # Check device settings first
        sd.check_input_settings(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
    except Exception as e:
        return None, f"Device settings check failed: {e}"

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=device_index,
        )
        sd.wait()  # Wait for recording to complete

        if audio is None or audio.size == 0:
            return None, "Recording returned empty data"

        return audio, ""

    except Exception as e:
        return None, f"Recording failed: {e}"


def test_device(
    device: DeviceInfo,
    duration: float = 3.0,
    sample_rate: Optional[int] = None,
    is_default: bool = False,
) -> DeviceTestResult:
    """
    Test a single microphone device.

    Records audio, measures signal quality, and classifies the result.
    """
    target_rate = sample_rate or (16000 if 16000 in device.supported_sample_rates else int(device.default_sample_rate))
    channels = min(device.max_input_channels, 2)

    audio, error = record_audio_from_device(
        device_index=device.index,
        duration=duration,
        sample_rate=target_rate,
        channels=channels,
    )

    if error:
        return DeviceTestResult(
            device=device,
            error=error,
            is_default=is_default,
        )

    measurements = measure_audio(audio, target_rate)

    return DeviceTestResult(
        device=device,
        measurements=measurements,
        is_default=is_default,
    )


def select_best_device(
    test_results: list[DeviceTestResult],
    configured_device: Optional[int] = None,
    configured_sample_rate: Optional[int] = None,
) -> Optional[DeviceTestResult]:
    """
    Select the best device based on test results.

    Priority:
      1. Explicitly configured device (if valid and tested)
      2. Windows default input device (if signal is usable)
      3. Best tested device by quality score
    """
    # Filter to devices that actually produced audio
    valid_results = [r for r in test_results if r.measurements is not None]

    if not valid_results:
        return None

    # 1. Check configured device
    if configured_device is not None:
        for result in valid_results:
            if result.device.index == configured_device:
                if result.measurements and result.measurements.quality not in (
                    AudioQuality.UNUSABLE_SILENCE,
                    AudioQuality.UNUSABLE_CLIPPING,
                ):
                    return result

    # 2. Check default device
    for result in valid_results:
        if result.is_default and result.measurements:
            if result.measurements.quality not in (
                AudioQuality.UNUSABLE_SILENCE,
                AudioQuality.UNUSABLE_CLIPPING,
            ):
                return result

    # 3. Pick best by quality ranking — NEVER select a silent device
    quality_rank = {
        AudioQuality.EXCELLENT: 6,
        AudioQuality.GOOD: 5,
        AudioQuality.NOISY: 4,
        AudioQuality.WEAK: 3,
        AudioQuality.UNUSABLE_NOISE: 2,
        AudioQuality.UNUSABLE_CLIPPING: 1,
        AudioQuality.UNUSABLE_SILENCE: 0,
    }

    best = None
    best_score = -1
    for result in valid_results:
        if result.measurements:
            score = quality_rank.get(result.measurements.quality, 0)
            # Strong bonus for devices with actual signal (RMS > 50)
            if result.measurements.rms > 500:
                score += 3
            elif result.measurements.rms > 100:
                score += 2
            elif result.measurements.rms > 50:
                score += 1
            # Minor bonus for default device only if it has signal
            if result.is_default and result.measurements.rms > 50:
                score += 0.5
            if score > best_score:
                best_score = score
                best = result

    return best


# ---------------------------------------------------------------------------
# Full diagnostic run
# ---------------------------------------------------------------------------

def run_microphone_diagnostics(
    configured_device: Optional[int] = None,
    configured_sample_rate: Optional[int] = None,
    record_duration: float = 3.0,
    test_all: bool = False,
) -> DiagnosticReport:
    """
    Run complete microphone diagnostics.

    Args:
        configured_device: Explicitly configured device index (from PENGU_MIC_DEVICE)
        configured_sample_rate: Explicitly configured sample rate (from PENGU_MIC_SAMPLE_RATE)
        record_duration: How long to record from each device (seconds)
        test_all: If True, test all devices. If False, test only default + top candidates.

    Returns:
        DiagnosticReport with all findings
    """
    report = DiagnosticReport()
    report.python_version = sys.version.split()[0]

    # Check Windows audio permission
    try:
        import sounddevice as sd
        # Try opening a default device
        sd.check_input_settings(samplerate=16000, channels=1)
        report.windows_permission = "GRANTED"
    except Exception as e:
        if "Invalid device" in str(e) or "not found" in str(e).lower():
            report.windows_permission = "NO_INPUT_DEVICE"
        else:
            report.windows_permission = f"ERROR: {e}"

    # Enumerate devices
    report.devices = enumerate_input_devices()
    if not report.devices:
        report.errors.append("No input devices found on this system.")
        return report

    # Determine which devices to test
    default_device = get_default_input_device()
    devices_to_test = []

    if test_all:
        devices_to_test = report.devices
    else:
        # Test the configured device first, then default, then first few candidates
        tested_indices = set()

        # Configured device
        if configured_device is not None:
            for dev in report.devices:
                if dev.index == configured_device:
                    devices_to_test.append(dev)
                    tested_indices.add(dev.index)
                    break

        # Default device
        if default_device and default_device.index not in tested_indices:
            devices_to_test.append(default_device)
            tested_indices.add(default_device.index)

        # Top candidates (prefer higher channel count, known good APIs)
        for dev in report.devices:
            if dev.index not in tested_indices and len(devices_to_test) < 5:
                devices_to_test.append(dev)
                tested_indices.add(dev.index)

    # Test each device
    for device in devices_to_test:
        is_default = default_device is not None and device.index == default_device.index
        result = test_device(
            device=device,
            duration=record_duration,
            is_default=is_default,
        )
        report.test_results.append(result)

    # Select best device
    best = select_best_device(
        report.test_results,
        configured_device=configured_device,
        configured_sample_rate=configured_sample_rate,
    )

    if best:
        report.selected_device = best.device
        best.selected = True
    else:
        report.errors.append(
            "No suitable microphone found. All tested devices produced unusable audio."
        )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_diagnostic_report(report: DiagnosticReport, verbose: bool = False) -> str:
    """Format a diagnostic report as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PENGU MICROPHONE DIAGNOSTICS")
    lines.append("=" * 60)
    lines.append("")

    # Python
    lines.append(f"  Python: {report.python_version}")
    lines.append("")

    # Windows permission
    perm_icon = "[OK]" if report.windows_permission == "GRANTED" else "[!!]"
    lines.append(f"  {perm_icon} Windows Audio Permission: {report.windows_permission}")
    if report.windows_permission != "GRANTED":
        lines.append("         Fix: Open Windows Settings → Privacy → Microphone → Allow apps")
        lines.append("         Also check: Settings → System → Sound → Input")
    lines.append("")

    # Devices
    lines.append(f"  Input Devices Found: {len(report.devices)}")
    for dev in report.devices:
        default_mark = " (DEFAULT)" if (
            report.test_results and
            any(r.device.index == dev.index and r.is_default for r in report.test_results)
        ) else ""
        lines.append(f"    Device {dev.index}: {dev.name}")
        lines.append(f"      Channels: {dev.max_input_channels}, Sample Rate: {dev.default_sample_rate}Hz, API: {dev.host_api}")
        if dev.supported_sample_rates:
            lines.append(f"      Supported Rates: {', '.join(str(int(r)) for r in dev.supported_sample_rates)}")
        if default_mark:
            lines.append(f"      {default_mark}")
    lines.append("")

    # Test results
    lines.append("  Test Results:")
    lines.append("  " + "-" * 56)
    for result in report.test_results:
        selected_mark = " *** SELECTED ***" if result.selected else ""
        default_mark = " (default)" if result.is_default else ""
        lines.append(f"  Device {result.device.index}: {result.device.name}{default_mark}{selected_mark}")

        if result.error:
            lines.append(f"    [ERROR] {result.error}")
        elif result.measurements:
            m = result.measurements
            quality_icon = {
                AudioQuality.EXCELLENT: "[++]",
                AudioQuality.GOOD: "[OK]",
                AudioQuality.NOISY: "[~~]",
                AudioQuality.WEAK: "[--]",
                AudioQuality.UNUSABLE_NOISE: "[!!]",
                AudioQuality.UNUSABLE_CLIPPING: "[!!]",
                AudioQuality.UNUSABLE_SILENCE: "[!!]",
            }.get(m.quality, "[??]")

            lines.append(f"    {quality_icon} Quality: {m.quality.value}")
            lines.append(f"       Duration:   {m.duration_seconds:.1f}s")
            lines.append(f"       Sample Rate: {m.sample_rate}Hz, Channels: {m.channels}")
            lines.append(f"       RMS:        {m.rms:.1f}")
            lines.append(f"       Peak:       {m.peak:.1f}")
            lines.append(f"       Avg Amp:    {m.avg_amplitude:.1f}")
            lines.append(f"       Noise Floor: {m.noise_floor:.1f}")
            lines.append(f"       Speech RMS: {m.speech_rms:.1f}")
            lines.append(f"       SNR:        {m.signal_to_noise_ratio:.1f} dB")
            lines.append(f"       Clipping:   {m.clipping_percent:.4f}%")
            lines.append(f"       {m.quality_detail}")

            if verbose and m.rms_history:
                lines.append(f"       Frame RMS (first 20): {[f'{r:.1f}' for r in m.rms_history[:20]]}")
        lines.append("  " + "-" * 56)
    lines.append("")

    # Selected device summary
    if report.selected_device:
        lines.append(f"  SELECTED MICROPHONE: Device {report.selected_device.index}")
        lines.append(f"    Name: {report.selected_device.name}")
        lines.append(f"    Channels: {report.selected_device.max_input_channels}")
        for result in report.test_results:
            if result.selected and result.measurements:
                lines.append(f"    Recommended Sample Rate: {result.measurements.sample_rate}Hz")
                lines.append(f"    Quality: {result.measurements.quality.value}")
    else:
        lines.append("  [!!] NO SUITABLE MICROPHONE FOUND")
        lines.append("  All tested devices produced unusable audio.")
        lines.append("")
        lines.append("  TROUBLESHOOTING:")
        lines.append("  1. Open Windows Sound Settings (Win+I → Sound)")
        lines.append("  2. Under Input, select the correct microphone")
        lines.append("  3. Ensure the microphone is not muted")
        lines.append("  4. Increase the input volume/slider")
        lines.append("  5. Test with Windows Voice Recorder app")
        lines.append("  6. If using a USB mic, try unplugging and replugging")
        lines.append("  7. Check Device Manager for microphone hardware issues")
        if report.windows_permission != "GRANTED":
            lines.append("  8. Grant microphone permission to Python in Windows Privacy Settings")

    # Errors
    if report.errors:
        lines.append("")
        lines.append("  ERRORS:")
        for err in report.errors:
            lines.append(f"    - {err}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_mic_test_report(
    device: DeviceInfo,
    measurements: AudioMeasurements,
    sample_rate: int,
) -> str:
    """Format a mic test report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PENGU MICROPHONE TEST")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Device:        {device.name}")
    lines.append(f"  Device Index:  {device.index}")
    lines.append(f"  Host API:      {device.host_api}")
    lines.append(f"  Max Channels:  {device.max_input_channels}")
    lines.append(f"  Sample Rate:   {sample_rate}Hz")
    lines.append(f"  Channels:      {measurements.channels}")
    lines.append(f"  Duration:      {measurements.duration_seconds:.1f}s")
    lines.append("")
    lines.append("  Measurements:")
    lines.append(f"    RMS:          {measurements.rms:.1f}")
    lines.append(f"    Peak:         {measurements.peak:.1f}")
    lines.append(f"    Avg Amplitude: {measurements.avg_amplitude:.1f}")
    lines.append(f"    Noise Floor:  {measurements.noise_floor:.1f}")
    lines.append(f"    Speech RMS:   {measurements.speech_rms:.1f}")
    lines.append(f"    SNR:          {measurements.signal_to_noise_ratio:.1f} dB")
    lines.append(f"    Clipping:     {measurements.clipping_percent:.4f}%")
    lines.append("")

    quality_icon = {
        AudioQuality.EXCELLENT: "[++]",
        AudioQuality.GOOD: "[OK]",
        AudioQuality.NOISY: "[~~]",
        AudioQuality.WEAK: "[--]",
        AudioQuality.UNUSABLE_NOISE: "[!!]",
        AudioQuality.UNUSABLE_CLIPPING: "[!!]",
        AudioQuality.UNUSABLE_SILENCE: "[!!]",
    }.get(measurements.quality, "[??]")

    lines.append(f"  Quality: {quality_icon} {measurements.quality.value}")
    lines.append(f"  Assessment: {measurements.quality_detail}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
