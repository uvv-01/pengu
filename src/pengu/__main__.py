"""
Allow running Pengu as: python -m pengu

Commands:
    python -m pengu                   — start Pengu voice assistant
    python -m pengu --diagnostics     — run full system diagnostics (then exit)
    python -m pengu --mic-test        — test microphone (then exit)
    python -m pengu --benchmark-models — benchmark available LLM models
    python -m pengu --hardware        — show hardware detection report
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _fix_encoding():
    """Fix Windows console encoding for Unicode output."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _run_diagnostics() -> None:
    """Run comprehensive system diagnostics."""
    _fix_encoding()
    import shutil
    import platform
    import psutil

    print("=" * 60)
    print("  PENGU SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print()

    print(f"  [OK] Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"  [OK] OS: Windows {platform.release()} ({platform.machine()})")
    print(f"       Version: {platform.version()}")

    cpu_model = platform.processor() or "Unknown"
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    print(f"  [OK] CPU: {cpu_model}")
    print(f"       Cores: {cpu_cores} physical / {cpu_threads} logical")

    mem = psutil.virtual_memory()
    print(f"  [OK] RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"  [OK] GPU: {result.stdout.strip()}")
        else:
            print("  [--] GPU: None detected (CPU-only)")
    except Exception:
        print("  [--] GPU: None detected (CPU-only)")

    git_path = shutil.which("git")
    print(f"  {'[OK]' if git_path else '[!!]'} Git: {'Found' if git_path else 'NOT FOUND'}")

    # Microphone — use AudioDeviceManager
    from pengu.voice.audio_device_manager import AudioDeviceManager
    manager = AudioDeviceManager(target_sample_rate=16000, num_rounds=1)
    devices = manager.enumerate_devices()
    if devices:
        print(f"  [OK] Microphones: {len(devices)} input devices found")
        for dev in devices:
            mic_mark = "" if dev["is_real_mic"] else " [not a mic]"
            print(f"       Device {dev['index']}: {dev['name']} ({dev['max_channels']}ch){mic_mark}")
    else:
        print("  [!!] Microphones: No input devices found")

    try:
        from faster_whisper import WhisperModel  # noqa: F401
        print("  [OK] STT: faster-whisper installed")
    except Exception:
        print("  [!!] STT: NOT INSTALLED")

    try:
        import edge_tts  # noqa: F401
        print("  [OK] TTS: edge-tts installed")
    except ImportError:
        print("  [!!] TTS: NOT INSTALLED")

    try:
        import openwakeword  # noqa: F401
        print("  [OK] Wake Word: openWakeWord installed")
    except ImportError:
        print("  [--] Wake Word: openWakeWord not installed")

    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
    except ImportError:
        print("  [--] PyTorch: NOT INSTALLED")

    # LM Studio
    try:
        import httpx
        resp = httpx.get("http://localhost:1234/v1/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                print(f"  [OK] LM Studio: {len(models)} model(s) loaded")
                for m in models:
                    print(f"       Model: {m.get('id', 'unknown')}")
            else:
                print("  [--] LM Studio: RUNNING (no models loaded)")
        else:
            print(f"  [!!] LM Studio: ERROR (HTTP {resp.status_code})")
    except Exception:
        print("  [--] LM Studio: NOT RUNNING")

    # Ollama
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                print(f"  [OK] Ollama: {len(models)} model(s)")
            else:
                print("  [--] Ollama: RUNNING (no models)")
        else:
            print("  [!!] Ollama: ERROR")
    except Exception:
        print("  [--] Ollama: NOT RUNNING")

    # App launcher
    from pengu.os.app_launcher import get_launcher
    launcher = get_launcher()
    apps = launcher.list_apps()
    found = sum(1 for app in apps if launcher.find_app(app["name"]))
    print(f"  [OK] App Launcher: {found}/{len(apps)} apps found")

    try:
        from playwright.async_api import async_playwright  # noqa: F401
        print("  [OK] Browser: Playwright installed")
    except ImportError:
        print("  [--] Browser: Playwright not installed")

    print("  [OK] Memory: SQLite-based (aiosqlite)")
    print()
    print("=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 60)


def _run_mic_test(duration: float = 3.0, test_all: bool = False) -> None:
    """Run microphone test using AudioDeviceManager — same as VoiceEngine."""
    _fix_encoding()
    from pengu.voice.mic_diagnostics import run_microphone_diagnostics, format_diagnostic_report

    configured_device = None
    if os.environ.get("PENGU_MIC_DEVICE"):
        try:
            configured_device = int(os.environ["PENGU_MIC_DEVICE"])
        except ValueError:
            print(f"  [!!] Invalid PENGU_MIC_DEVICE: {os.environ['PENGU_MIC_DEVICE']}")
            return

    print(f"  Probing all microphone devices ({duration}s each)...")
    print(f"  Using AudioDeviceManager (same selector as VoiceEngine)")
    print()

    result = run_microphone_diagnostics(
        configured_device=configured_device,
        record_duration=duration,
        test_all=test_all,
    )

    print(format_diagnostic_report(result, verbose=True))


def _run_benchmark_models() -> None:
    """Benchmark available LLM models."""
    _fix_encoding()

    async def _benchmark():
        import httpx
        import time

        print("=" * 60)
        print("  PENGU MODEL BENCHMARK")
        print("=" * 60)
        print()

        providers = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:1234/v1/models")
                if resp.status_code == 200:
                    for m in resp.json().get("data", []):
                        providers.append(("LM Studio", "http://localhost:1234/v1", m.get("id", "unknown")))
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    for m in resp.json().get("models", []):
                        providers.append(("Ollama", "http://localhost:11434", m.get("name", "unknown")))
        except Exception:
            pass

        if not providers:
            print("  No model providers found.")
            print("=" * 60)
            return

        print(f"  Found {len(providers)} model(s)")
        for source, url, model in providers:
            print(f"    {source}: {model}")
        print()

        test_messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "user", "content": "Open VS Code"},
        ]

        for source, url, model in providers:
            print(f"  Benchmarking: {model} ({source})")
            total_latency = 0
            successes = 0
            for i, msg in enumerate(test_messages):
                try:
                    start = time.perf_counter()
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        resp = await client.post(f"{url}/chat/completions", json={
                            "model": model,
                            "messages": [{"role": "system", "content": "Be concise."}, msg],
                            "temperature": 0.1, "max_tokens": 100,
                        })
                        latency = (time.perf_counter() - start) * 1000
                        if resp.status_code == 200:
                            data = resp.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            tokens = data.get("usage", {}).get("completion_tokens", 0)
                            tok_per_sec = (tokens / (latency / 1000)) if latency > 0 else 0
                            total_latency += latency
                            successes += 1
                            print(f"    Q{i+1}: {content[:80]}... ({latency:.0f}ms, {tok_per_sec:.1f} tok/s)")
                except Exception as e:
                    print(f"    Q{i+1}: ERROR ({e})")

            avg = total_latency / successes if successes > 0 else 0
            print(f"    Average: {avg:.0f}ms, {successes}/{len(test_messages)} success")
            print()

        print("=" * 60)

    asyncio.run(_benchmark())


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="pengu",
        description="Pengu — local-first voice-first Windows desktop assistant",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Run full system diagnostics and exit")
    parser.add_argument("--mic-test", action="store_true", help="Test microphone with real measurements and exit")
    parser.add_argument("--benchmark-models", action="store_true", help="Benchmark available LLM models and exit")
    parser.add_argument("--hardware", action="store_true", help="Show hardware detection report and exit")
    parser.add_argument("--mic-duration", type=float, default=3.0, help="Duration for mic test recording (seconds)")
    parser.add_argument("--test-all-mics", action="store_true", help="Test all microphone devices")

    args = parser.parse_args()

    if args.diagnostics:
        _run_diagnostics()
    elif args.mic_test:
        _run_mic_test(duration=args.mic_duration, test_all=args.test_all_mics)
    elif args.benchmark_models:
        _run_benchmark_models()
    elif args.hardware:
        from pengu.hardware.cli import main as hw_main
        hw_main()
    else:
        from pengu.app import main as app_main
        asyncio.run(app_main())


if __name__ == "__main__":
    main()
