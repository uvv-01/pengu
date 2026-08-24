"""
CLI for hardware detection.

Usage:
    python -m pengu.hardware.detect
    pengu-hw
"""

import json
import sys

from pengu.hardware.detect import detect_hardware


def main() -> None:
    info = detect_hardware()

    print("=" * 60)
    print("  PENGU - HARDWARE DETECTION REPORT")
    print("=" * 60)
    print()

    # OS
    print(f"  OS:        {info.os_name} {info.os_version} ({info.os_arch})")
    print(f"  Hostname:  {info.hostname}")
    print()

    # CPU
    print(f"  CPU:       {info.cpu_model}")
    print(f"  Cores:     {info.cpu_cores_physical} physical / {info.cpu_cores_logical} logical")
    if info.cpu_freq_max_mhz:
        print(f"  Freq:      {info.cpu_freq_max_mhz:.0f} MHz max")
    print()

    # RAM
    print(f"  RAM:       {info.ram_total_gb:.1f} GB total / {info.ram_available_gb:.1f} GB available")
    print()

    # GPU
    print(f"  GPU:       {info.gpu.name}")
    if info.gpu.vram_gb:
        print(f"  VRAM:      {info.gpu.vram_gb:.1f} GB")
    if info.gpu.driver:
        print(f"  Driver:    {info.gpu.driver}")
    print()

    # Storage
    print(f"  Storage:   {info.disk_total_gb:.1f} GB total / {info.disk_free_gb:.1f} GB free")
    print()

    # Tools
    print("  Installed Tools:")
    for name, tool in info.tools.items():
        status = "[OK]" if tool.available else "[--]"
        ver = f" ({tool.version})" if tool.version else ""
        print(f"    {status} {name}{ver}")
    print()

    # Tier
    print(f"  +---------------------------------------+")
    print(f"  |  HARDWARE TIER:  {info.tier.value:<20s} |")
    print(f"  +---------------------------------------+")
    print()

    # Recommended models
    print("  Recommended Models:")
    for role, model in info.recommended_models.items():
        model_display = model or "(not available for this tier)"
        print(f"    {role:<12s} -> {model_display}")
    print()

    # JSON output if requested
    if "--json" in sys.argv:
        print("\n--- JSON ---")
        print(json.dumps(info.to_dict(), indent=2))


if __name__ == "__main__":
    main()
