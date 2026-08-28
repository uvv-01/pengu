"""
Pengu Microphone Diagnostics — uses AudioDeviceManager as the single source of truth.

All device selection, probing, and scoring is done by AudioDeviceManager.
This module only formats and presents the results.

Usage:
    python -m pengu --mic-test
"""

from __future__ import annotations

import sys
from typing import Optional

from pengu.voice.audio_device_manager import (
    AudioDeviceManager,
    DeviceSelection,
    ProbeResult,
    AudioQuality,
    TARGET_SAMPLE_RATE,
    MIN_SNR_DB,
    MIN_SPEECH_RMS,
)


def run_microphone_diagnostics(
    configured_device: Optional[int] = None,
    record_duration: float = 3.0,
    test_all: bool = False,
) -> dict:
    """
    Run complete microphone diagnostics using AudioDeviceManager.

    Returns a dict with all probe results and the selected device.
    """
    import os

    # Check env override
    if configured_device is None:
        env_device = os.environ.get("PENGU_MIC_DEVICE")
        if env_device:
            try:
                configured_device = int(env_device)
            except ValueError:
                pass

    # Use num_rounds=1 for quick enumeration, num_rounds=3 for best selection
    num_rounds = 3

    manager = AudioDeviceManager(
        configured_device=configured_device,
        target_sample_rate=TARGET_SAMPLE_RATE,
        num_rounds=num_rounds,
    )

    # Enumerate devices
    devices = manager.enumerate_devices()

    # Select best device (this probes all candidates)
    selection = manager.select_best_device()

    return {
        "devices": devices,
        "selection": selection,
        "all_probes": manager.all_probes,
    }


def format_diagnostic_report(result: dict, verbose: bool = False) -> str:
    """Format diagnostic results as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("  PENGU MICROPHONE DIAGNOSTICS")
    lines.append("  (using AudioDeviceManager -- single source of truth)")
    lines.append("=" * 60)
    lines.append("")

    # Devices
    devices = result.get("devices", [])
    lines.append(f"  Input Devices Found: {len(devices)}")
    for dev in devices:
        mic_mark = "" if dev["is_real_mic"] else " [NOT A MICROPHONE]"
        lines.append(f"    Device {dev['index']}: {dev['name']}")
        lines.append(f"      API: {dev['host_api']}, Channels: {dev['max_channels']}, Native SR: {dev['native_sample_rate']:.0f}{mic_mark}")
    lines.append("")

    # Probe results
    all_probes = result.get("all_probes", [])
    lines.append("  Probe Results:")
    lines.append("  " + "-" * 56)

    for probe in all_probes:
        if not probe.can_open:
            lines.append(f"  Device {probe.device_index}: {probe.device_name}")
            lines.append(f"    [ERROR] {probe.error}")
            lines.append("  " + "-" * 56)
            continue

        # Quality icon
        quality_icon = {
            AudioQuality.EXCELLENT: "[++]",
            AudioQuality.GOOD:      "[OK]",
            AudioQuality.ACCEPTABLE:"[~~]",
            AudioQuality.POOR:      "[--]",
            AudioQuality.UNUSABLE:  "[!!]",
        }.get(probe.quality, "[??]")

        # Voice readiness
        voice_status = "VOICE READY" if probe.quality.is_voice_ready else "NOT VOICE READY"

        lines.append(f"  Device {probe.device_index}: {probe.device_name}")
        lines.append(f"    {quality_icon} Quality: {probe.quality.value} ({voice_status})")
        lines.append(f"       API:          {probe.host_api}")
        lines.append(f"       Sample Rate:  {probe.capture_sample_rate} Hz")
        lines.append(f"       Channels:     {probe.capture_channels}")
        lines.append(f"       Noise Floor:  {probe.noise_floor:.1f}")
        lines.append(f"       Speech RMS:   {probe.speech_rms:.1f}")
        lines.append(f"       Overall RMS:  {probe.overall_rms:.1f}")
        lines.append(f"       Peak:         {probe.overall_peak:.1f}")
        lines.append(f"       SNR:          {probe.snr_db:.1f} dB")
        lines.append(f"       Clipping:     {probe.clipping_percent:.4f}%")
        lines.append(f"       Score:        {probe.quality_score:.1f}")
        lines.append(f"       Speech:       {'YES' if probe.speech_detected else 'NO'}")
        if verbose and probe.round_snr_values:
            lines.append(f"       Round SNRs:   {[f'{s:.1f}' for s in probe.round_snr_values]}")
        if verbose and probe.round_speech_rms_values:
            lines.append(f"       Round Speech: {[f'{r:.0f}' for r in probe.round_speech_rms_values]}")
        lines.append("  " + "-" * 56)

    lines.append("")

    # Selected device
    selection = result.get("selection")
    if selection:
        voice_status = "YES" if selection.voice_ready else "NO"
        lines.append(f"  SELECTED MICROPHONE: Device {selection.device_index}")
        lines.append(f"    Name:           {selection.device_name}")
        lines.append(f"    API:            {selection.host_api}")
        lines.append(f"    Sample Rate:    {selection.capture_sample_rate} Hz")
        lines.append(f"    Channels:       {selection.capture_channels}")
        lines.append(f"    Noise Floor:    {selection.noise_floor:.1f}")
        lines.append(f"    Speech RMS:     {selection.speech_rms:.1f}")
        lines.append(f"    Overall RMS:    {selection.rms:.1f}")
        lines.append(f"    Peak:           {selection.peak:.1f}")
        lines.append(f"    SNR:            {selection.snr_db:.1f} dB")
        lines.append(f"    Clipping:       {selection.clipping_percent:.4f}%")
        lines.append(f"    Quality:        {selection.quality.value}")
        lines.append(f"    Score:          {selection.quality_score:.1f}")
        lines.append(f"    Voice Ready:    {voice_status}")
        lines.append(f"    Reason:         {selection.selection_reason}")

        # Requirements
        lines.append("")
        lines.append(f"    Requirements for ACCEPTABLE quality:")
        lines.append(f"      SNR >= {MIN_SNR_DB} dB:     {'MET' if selection.snr_db >= MIN_SNR_DB else 'NOT MET'} ({selection.snr_db:.1f} dB)")
        lines.append(f"      Speech RMS >= {MIN_SPEECH_RMS}:  {'MET' if selection.speech_rms >= MIN_SPEECH_RMS else 'NOT MET'} ({selection.speech_rms:.1f})")
        lines.append(f"      Speech detected:   {'YES' if selection.speech_detected else 'NO'}")
        lines.append(f"      Clipping < 1%:     {'MET' if selection.clipping_percent < 1.0 else 'NOT MET'} ({selection.clipping_percent:.4f}%)")

        if not selection.voice_ready:
            lines.append("")
            lines.append("  WARNING: Selected microphone is NOT voice-ready.")
            lines.append("  Quality must be ACCEPTABLE or better for reliable voice capture.")
            lines.append("  Please speak into the microphone during the probe for accurate measurement.")
    else:
        lines.append("  [!!] NO SUITABLE MICROPHONE FOUND")
        lines.append("  All tested devices produced unusable audio.")
        lines.append("")
        lines.append("  TROUBLESHOOTING:")
        lines.append("  1. Open Windows Sound Settings (Win+I > Sound)")
        lines.append("  2. Under Input, select the correct microphone")
        lines.append("  3. Ensure the microphone is not muted")
        lines.append("  4. Increase the input volume/slider")
        lines.append("  5. Test with Windows Voice Recorder app")
        lines.append("  6. Run: python -m pengu --mic-test")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
