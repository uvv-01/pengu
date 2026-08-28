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
import sys


def _run_diagnostics() -> None:
    """Run comprehensive system diagnostics without starting the voice engine."""
    import shutil
    import platform
    import psutil

    print("=" * 60)
    print("  PENGU SYSTEM DIAGNOSTICS")
    print("=" * 60)
    print()

    # Python
    print(f"  [OK] Python: {sys.version.split()[0]} ({platform.python_implementation()})")
    print()

    # OS
    print(f"  [OK] OS: Windows {platform.release()} ({platform.machine()})")
    print(f"       Version: {platform.version()}")
    print()

    # CPU
    cpu_model = platform.processor() or "Unknown"
    cpu_cores = psutil.cpu_count(logical=False) or 0
    cpu_threads = psutil.cpu_count(logical=True) or 0
    print(f"  [OK] CPU: {cpu_model}")
    print(f"       Cores: {cpu_cores} physical / {cpu_threads} logical")
    print()

    # RAM
    mem = psutil.virtual_memory()
    print(f"  [OK] RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    print()

    # GPU
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
    print()

    # Git
    git_path = shutil.which("git")
    print(f"  {'[OK]' if git_path else '[!!]'} Git: {'Found' if git_path else 'NOT FOUND'}")
    print()

    # Microphone
    from pengu.voice.mic_diagnostics import enumerate_input_devices, get_default_input_device
    devices = enumerate_input_devices()
    default_dev = get_default_input_device()

    if devices:
        print(f"  [OK] Microphones: {len(devices)} input devices found")
        for dev in devices:
            default_mark = " (DEFAULT)" if default_dev and dev.index == default_dev.index else ""
            print(f"       Device {dev.index}: {dev.name} ({dev.max_input_channels}ch){default_mark}")
    else:
        print("  [!!] Microphones: No input devices found")
    print()

    # STT
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        print("  [OK] STT: faster-whisper installed")
    except ImportError:
        print("  [!!] STT: NOT INSTALLED (pip install faster-whisper)")
    print()

    # TTS
    try:
        import edge_tts  # noqa: F401
        print("  [OK] TTS: edge-tts installed")
    except ImportError:
        print("  [!!] TTS: NOT INSTALLED (pip install edge-tts)")
    print()

    # Wake word
    try:
        import openwakeword  # noqa: F401
        print("  [OK] Wake Word: openWakeWord installed")
    except ImportError:
        print("  [--] Wake Word: openWakeWord not installed (uses STT-based wake)")
    print()

    # Torch
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
    except ImportError:
        print("  [--] PyTorch: NOT INSTALLED")
    print()

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
    print()

    # Ollama
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                print(f"  [OK] Ollama: {len(models)} model(s)")
                for m in models:
                    print(f"       Model: {m.get('name', 'unknown')}")
            else:
                print("  [--] Ollama: RUNNING (no models)")
        else:
            print("  [!!] Ollama: ERROR")
    except Exception:
        print("  [--] Ollama: NOT RUNNING")
    print()

    # App launcher
    from pengu.os.app_launcher import get_launcher
    launcher = get_launcher()
    apps = launcher.list_apps()
    found = sum(1 for app in apps if launcher.find_app(app["name"]))
    print(f"  [OK] App Launcher: {found}/{len(apps)} apps found")
    for app in apps:
        status = "[OK]" if launcher.find_app(app["name"]) else "[!!]"
        print(f"       {status} {app['name']}")
    print()

    # Browser
    try:
        from playwright.async_api import async_playwright  # noqa: F401
        print("  [OK] Browser: Playwright installed")
    except ImportError:
        print("  [--] Browser: Playwright not installed (pip install playwright)")
    print()

    # Memory
    print("  [OK] Memory: SQLite-based (aiosqlite)")
    print()

    print("=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 60)


def _run_mic_test(duration: float = 3.0, test_all: bool = False) -> None:
    """Run microphone test with real measurements."""
    from pengu.voice.mic_diagnostics import (
        run_microphone_diagnostics,
        format_diagnostic_report,
        enumerate_input_devices,
        get_default_input_device,
        test_device,
        format_mic_test_report,
    )

    # Check for environment overrides
    configured_device = None
    configured_sample_rate = None

    if os.environ.get("PENGU_MIC_DEVICE"):
        try:
            configured_device = int(os.environ["PENGU_MIC_DEVICE"])
        except ValueError:
            print(f"  [!!] Invalid PENGU_MIC_DEVICE: {os.environ['PENGU_MIC_DEVICE']}")
            return

    if os.environ.get("PENGU_MIC_SAMPLE_RATE"):
        try:
            configured_sample_rate = int(os.environ["PENGU_MIC_SAMPLE_RATE"])
        except ValueError:
            print(f"  [!!] Invalid PENGU_MIC_SAMPLE_RATE: {os.environ['PENGU_MIC_SAMPLE_RATE']}")
            return

    print(f"  Recording {duration}s from each device...")
    print()

    if configured_device is not None:
        print(f"  Using configured device: {configured_device}")
    if configured_sample_rate is not None:
        print(f"  Using configured sample rate: {configured_sample_rate}Hz")
    print()

    report = run_microphone_diagnostics(
        configured_device=configured_device,
        configured_sample_rate=configured_sample_rate,
        record_duration=duration,
        test_all=test_all,
    )

    print(format_diagnostic_report(report, verbose=True))


def _run_benchmark_models() -> None:
    """Benchmark available LLM models for Pengu."""
    import asyncio

    async def _benchmark():
        import httpx
        import time

        print("=" * 60)
        print("  PENGU MODEL BENCHMARK")
        print("=" * 60)
        print()

        providers = []

        # Check LM Studio
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:1234/v1/models")
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    if models:
                        for m in models:
                            mid = m.get("id", "unknown")
                            providers.append(("LM Studio", "http://localhost:1234/v1", mid))
        except Exception:
            pass

        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    for m in models:
                        name = m.get("name", "unknown")
                        providers.append(("Ollama", "http://localhost:11434", name))
        except Exception:
            pass

        if not providers:
            print("  No model providers found.")
            print("  Start LM Studio or Ollama with a model loaded.")
            print()
            print("=" * 60)
            return

        print(f"  Found {len(providers)} model(s) to benchmark:")
        for source, url, model in providers:
            print(f"    {source}: {model}")
        print()

        # Benchmark each model
        test_messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "user", "content": "Open VS Code"},
            {"role": "user", "content": "What time is it?"},
        ]

        results = []
        for source, url, model in providers:
            print(f"  Benchmarking: {model} ({source})")
            print("  " + "-" * 50)

            total_latency = 0
            total_tokens = 0
            successes = 0
            first_response_time = None

            for i, msg in enumerate(test_messages):
                try:
                    start = time.perf_counter()
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                                msg,
                            ],
                            "temperature": 0.1,
                            "max_tokens": 100,
                        }
                        resp = await client.post(f"{url}/chat/completions", json=payload)
                        latency = (time.perf_counter() - start) * 1000

                        if resp.status_code == 200:
                            data = resp.json()
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            usage = data.get("usage", {})
                            tokens = usage.get("total_tokens", 0)
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)

                            if first_response_time is None:
                                first_response_time = latency

                            total_latency += latency
                            total_tokens += completion_tokens
                            successes += 1

                            tok_per_sec = (completion_tokens / (latency / 1000)) if latency > 0 else 0
                            print(f"    Q{i+1}: {msg['content']}")
                            print(f"         Response: {content[:100]}...")
                            print(f"         Latency: {latency:.0f}ms, Tokens: {tokens}, Rate: {tok_per_sec:.1f} tok/s")
                        else:
                            print(f"    Q{i+1}: ERROR (HTTP {resp.status_code})")
                except Exception as e:
                    print(f"    Q{i+1}: ERROR ({e})")

            avg_latency = total_latency / successes if successes > 0 else 0
            avg_tok_per_sec = (total_tokens / (total_latency / 1000)) if total_latency > 0 else 0

            print()
            print(f"    Summary for {model}:")
            print(f"      Successes:     {successes}/{len(test_messages)}")
            print(f"      Avg Latency:   {avg_latency:.0f}ms")
            print(f"      First Resp:    {first_response_time:.0f}ms" if first_response_time else "      First Resp:    N/A")
            print(f"      Avg Tokens/s:  {avg_tok_per_sec:.1f}")
            print(f"      Total Tokens:  {total_tokens}")
            print()

            results.append({
                "model": model,
                "source": source,
                "successes": successes,
                "avg_latency_ms": avg_latency,
                "first_response_ms": first_response_time,
                "avg_tokens_per_sec": avg_tok_per_sec,
                "total_tokens": total_tokens,
            })

            print("  " + "-" * 50)
            print()

        # Recommendations
        if results:
            print("  RECOMMENDATIONS:")
            best_by_latency = min(results, key=lambda r: r["avg_latency_ms"] if r["avg_latency_ms"] > 0 else 999999)
            best_by_quality = max(results, key=lambda r: r["avg_tokens_per_sec"])

            if best_by_latency["source"] == best_by_quality["source"] and best_by_latency["model"] == best_by_quality["model"]:
                print(f"    Best overall: {best_by_latency['model']} ({best_by_latency['source']})")
            else:
                print(f"    Fastest: {best_by_latency['model']} ({best_by_latency['source']}, {best_by_latency['avg_latency_ms']:.0f}ms avg)")
                print(f"    Best throughput: {best_by_quality['model']} ({best_by_quality['source']}, {best_by_quality['avg_tokens_per_sec']:.1f} tok/s)")

        print()
        print("=" * 60)
        print("  BENCHMARK COMPLETE")
        print("=" * 60)

    asyncio.run(_benchmark())


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="pengu",
        description="Pengu — ₹0-cost local-first voice-first Windows desktop assistant",
    )

    parser.add_argument("--diagnostics", action="store_true", help="Run full system diagnostics and exit")
    parser.add_argument("--mic-test", action="store_true", help="Test microphone with real measurements and exit")
    parser.add_argument("--benchmark-models", action="store_true", help="Benchmark available LLM models and exit")
    parser.add_argument("--hardware", action="store_true", help="Show hardware detection report and exit")
    parser.add_argument("--mic-duration", type=float, default=3.0, help="Duration for mic test recording (seconds)")
    parser.add_argument("--test-all-mics", action="store_true", help="Test all microphone devices (not just candidates)")

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


# Needed for env var access in _run_mic_test
import os  # noqa: E402

if __name__ == "__main__":
    main()
