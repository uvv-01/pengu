"""Allow running Pengu as: python -m pengu"""

import asyncio
import sys


def _run_diagnostics():
    """Run system diagnostics without starting the full voice engine."""
    import shutil

    print("=" * 50)
    print("PENGU DIAGNOSTICS")
    print("=" * 50)

    # Python
    print(f"Python:          OK ({sys.version.split()[0]})")

    # Git
    git_path = shutil.which("git")
    print(f"Git:             {'OK' if git_path else 'NOT FOUND'}")

    # Microphone
    try:
        import sounddevice as sd
        devices = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if devices:
            print(f"Microphone:      OK ({len(devices)} devices found)")
            for i, d in enumerate(devices):
                print(f"  Device {i}: {d['name']} ({d['max_input_channels']} ch)")
        else:
            print("Microphone:      NOT FOUND")
    except Exception as e:
        print(f"Microphone:      ERROR ({e})")

    # STT
    try:
        from faster_whisper import WhisperModel
        print("STT:             OK (faster-whisper installed)")
    except ImportError:
        print("STT:             NOT INSTALLED (pip install faster-whisper)")

    # TTS
    try:
        import edge_tts
        print("TTS:             OK (edge-tts installed)")
    except ImportError:
        print("TTS:             NOT INSTALLED (pip install edge-tts)")

    # Torch
    try:
        import torch
        print(f"Torch:           OK ({torch.__version__})")
    except ImportError:
        print("Torch:           NOT INSTALLED")

    # LM Studio
    try:
        import httpx
        resp = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                print(f"LM Studio:       OK ({len(models)} model(s) loaded)")
                for m in models:
                    print(f"  Model: {m.get('id', 'unknown')}")
            else:
                print("LM Studio:       RUNNING (no models loaded)")
        else:
            print(f"LM Studio:       ERROR (HTTP {resp.status_code})")
    except Exception:
        print("LM Studio:       NOT RUNNING")

    # Ollama
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                print(f"Ollama:          OK ({len(models)} model(s))")
                for m in models:
                    print(f"  Model: {m.get('name', 'unknown')}")
            else:
                print("Ollama:          RUNNING (no models)")
        else:
            print(f"Ollama:          ERROR")
    except Exception:
        print("Ollama:          NOT RUNNING")

    # App launcher
    from pengu.os.app_launcher import get_launcher
    launcher = get_launcher()
    apps = launcher.list_apps()
    print(f"App Launcher:    OK ({len(apps)} apps registered)")
    for app in apps:
        found = launcher.find_app(app["name"])
        status = "OK" if found else "NOT FOUND"
        print(f"  {app['name']}: {status}")

    # Hardware
    import platform
    import psutil
    print(f"OS:              {platform.system()} {platform.release()}")
    print(f"CPU:             {platform.processor()}")
    print(f"RAM:             {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB free")

    print("=" * 50)
    print("DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostics":
        _run_diagnostics()
    else:
        from pengu.app import main
        asyncio.run(main())
