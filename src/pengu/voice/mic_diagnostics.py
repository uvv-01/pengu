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

    manager = AudioDeviceManager(
        configured_device=configured_device,
        target_sample_rate=TARGET_SAMPLE_RATE,
        probe_duration=min(record_duration, 2.0),
        num_probe_windows=3,
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
    lines.append("  (using AudioDeviceManager — single source of truth)")
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

        quality_icon = {
            AudioQuality.EXCELLENT: "[++]",
            AudioQuality.GOOD: "[OK]",
            AudioQuality.NOISY: "[~~]",
            AudioQuality.WEAK: "[--]",
            AudioQuality.UNUSABLE_NOISE: "[!!]",
            AudioQuality.UNUSABLE_CLIPPING: "[!!]",
            AudioQuality.UNUSABLE_SILENCE: "[!!]",
        }.get(probe.quality, "[??]")

        lines.append(f"  Device {probe.device_index}: {probe.device_name}")
        lines.append(f"    {quality_icon} Quality: {probe.quality.value}")
        lines.append(f"       API:          {probe.host_api}")
        lines.append(f"       Sample Rate:  {probe.capture_sample_rate} Hz")
        lines.append(f"       Channels:     {probe.capture_channels}")
        lines.append(f"       RMS:          {probe.rms:.1f}")
        lines.append(f"       Speech RMS:   {probe.speech_rms:.1f}")
        lines.append(f"       Peak:         {probe.peak:.1f}")
        lines.append(f"       Noise Floor:  {probe.noise_floor:.1f}")
        lines.append(f"       SNR:          {probe.snr:.1f} dB")
        lines.append(f"       Clipping:     {probe.clipping_percent:.4f}%")
        lines.append(f"       Score:        {probe.quality_score:.1f}")
        if verbose and probe.window_rms_values:
            lines.append(f"       Windows:      {[f'{r:.0f}' for r in probe.window_rms_values]}")
        lines.append("  " + "-" * 56)

    lines.append("")

    # Selected device
    selection = result.get("selection")
    if selection:
        lines.append(f"  SELECTED MICROPHONE: Device {selection.device_index}")
        lines.append(f"    Name:           {selection.device_name}")
        lines.append(f"    API:            {selection.host_api}")
        lines.append(f"    Sample Rate:    {selection.capture_sample_rate} Hz")
        lines.append(f"    Channels:       {selection.capture_channels}")
        lines.append(f"    RMS:            {selection.rms:.1f}")
        lines.append(f"    Speech RMS:     {selection.speech_rms:.1f}")
        lines.append(f"    Peak:           {selection.peak:.1f}")
        lines.append(f"    Noise Floor:    {selection.noise_floor:.1f}")
        lines.append(f"    SNR:            {selection.snr:.1f} dB")
        lines.append(f"    Clipping:       {selection.clipping_percent:.4f}%")
        lines.append(f"    Quality:        {selection.quality.value}")
        lines.append(f"    Score:          {selection.quality_score:.1f}")
        lines.append(f"    Reason:         {selection.selection_reason}")
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

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
