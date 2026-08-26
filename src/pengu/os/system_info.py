"""
System Info — deterministic system information tool.

Reuses Pengu's existing hardware detection module.
Provides a clean API for the router/pipeline.
"""

from __future__ import annotations

import platform
import time
from typing import Optional

import psutil

from pengu.hardware.detect import detect_hardware, HardwareInfo
from pengu.logging import get_logger

logger = get_logger("pengu.os.system_info")


def get_system_info() -> dict:
    """
    Get comprehensive system information.
    
    Returns a structured dict with OS, CPU, RAM, GPU, disk,
    and tool information.
    """
    hw = detect_hardware()
    
    # Additional runtime info
    uptime_seconds = time.time() - psutil.boot_time()
    uptime_days = int(uptime_seconds // 86400)
    uptime_hours = int((uptime_seconds % 86400) // 3600)
    uptime_mins = int((uptime_seconds % 3600) // 60)

    info = {
        "os": {
            "name": hw.os_name,
            "version": hw.os_version,
            "arch": hw.os_arch,
            "hostname": hw.hostname,
        },
        "cpu": {
            "model": hw.cpu_model,
            "cores_physical": hw.cpu_cores_physical,
            "cores_logical": hw.cpu_cores_logical,
            "freq_max_mhz": hw.cpu_freq_max_mhz,
            "usage_percent": psutil.cpu_percent(interval=0.5),
        },
        "ram": {
            "total_gb": round(hw.ram_total_gb, 1),
            "available_gb": round(hw.ram_available_gb, 1),
            "used_percent": psutil.virtual_memory().percent,
        },
        "gpu": {
            "name": hw.gpu.name,
            "driver": hw.gpu.driver,
            "vram_gb": round(hw.gpu.vram_gb, 1),
            "api": hw.gpu.api,
        },
        "storage": {
            "total_gb": round(hw.disk_total_gb, 1),
            "free_gb": round(hw.disk_free_gb, 1),
            "used_percent": round(
                (1 - hw.disk_free_gb / hw.disk_total_gb) * 100, 1
            ) if hw.disk_total_gb > 0 else 0,
        },
        "uptime": {
            "days": uptime_days,
            "hours": uptime_hours,
            "minutes": uptime_mins,
            "total_seconds": int(uptime_seconds),
        },
        "tier": hw.tier.value,
        "tools": {
            name: {
                "available": info.available,
                "version": info.version,
            }
            for name, info in hw.tools.items()
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
    }

    return info


def get_system_info_summary() -> str:
    """
    Get a human-readable system info summary.
    
    Returns a formatted string suitable for voice/UI output.
    """
    info = get_system_info()

    lines = [
        f"OS: {info['os']['name']} ({info['os']['arch']})",
        f"CPU: {info['cpu']['model']} — {info['cpu']['cores_logical']} cores",
        f"RAM: {info['ram']['total_gb']}GB total, {info['ram']['available_gb']}GB available ({info['ram']['used_percent']}% used)",
        f"GPU: {info['gpu']['name']}" if info['gpu']['name'] != "none" else "GPU: None (integrated/CPU only)",
        f"Storage: {info['storage']['total_gb']}GB total, {info['storage']['free_gb']}GB free",
        f"Tier: {info['tier']}",
        f"Python: {info['python']['version']}",
        f"Uptime: {info['uptime']['days']}d {info['uptime']['hours']}h {info['uptime']['minutes']}m",
    ]

    # Available tools
    available = [name for name, t in info["tools"].items() if t["available"]]
    if available:
        lines.append(f"Tools: {', '.join(available)}")

    return "\n".join(lines)
