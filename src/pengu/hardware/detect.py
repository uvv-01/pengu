"""
Hardware detection for Pengu.

Detects:
  - OS and version
  - CPU model, cores, threads
  - RAM total/available
  - GPU (via WMI / nvidia-smi)
  - VRAM
  - Storage
  - Installed tools (Python, Git, Node, VS Code, Ollama, etc.)

Classifies machine into LOW / MEDIUM / HIGH tier.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import psutil

from pengu.config import HardwareTier


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str = "none"
    driver: str = ""
    vram_gb: float = 0.0
    api: str = ""  # "cuda", "directml", "vulkan", ""


@dataclass
class ToolInfo:
    name: str
    path: str
    version: str = ""
    available: bool = True


@dataclass
class HardwareInfo:
    """Complete hardware profile."""

    # OS
    os_name: str = ""
    os_version: str = ""
    os_arch: str = ""
    hostname: str = ""

    # CPU
    cpu_model: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_max_mhz: float = 0.0

    # RAM
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpu: GPUInfo = field(default_factory=GPUInfo)

    # Storage
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0

    # Installed tools
    tools: dict[str, ToolInfo] = field(default_factory=dict)

    # Classification
    tier: HardwareTier = HardwareTier.LOW
    recommended_models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "os": {
                "name": self.os_name,
                "version": self.os_version,
                "arch": self.os_arch,
                "hostname": self.hostname,
            },
            "cpu": {
                "model": self.cpu_model,
                "cores_physical": self.cpu_cores_physical,
                "cores_logical": self.cpu_cores_logical,
                "freq_max_mhz": self.cpu_freq_max_mhz,
            },
            "ram": {
                "total_gb": round(self.ram_total_gb, 1),
                "available_gb": round(self.ram_available_gb, 1),
            },
            "gpu": {
                "name": self.gpu.name,
                "driver": self.gpu.driver,
                "vram_gb": round(self.gpu.vram_gb, 1),
                "api": self.gpu.api,
            },
            "storage": {
                "total_gb": round(self.disk_total_gb, 1),
                "free_gb": round(self.disk_free_gb, 1),
            },
            "tools": {
                name: {"path": t.path, "version": t.version, "available": t.available}
                for name, t in self.tools.items()
            },
            "tier": self.tier.value,
            "recommended_models": self.recommended_models,
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_os() -> tuple[str, str, str]:
    """Return (name, version, arch)."""
    system = platform.system()
    if system == "Windows":
        ver = platform.version()
        release = platform.release()
        arch = platform.machine()
        return f"Windows {release}", ver, arch
    elif system == "Linux":
        ver = platform.release()
        arch = platform.machine()
        return f"Linux", ver, arch
    elif system == "Darwin":
        ver = platform.mac_ver()[0]
        arch = platform.machine()
        return f"macOS", ver, arch
    return system, platform.release(), platform.machine()


def _detect_cpu() -> tuple[str, int, int, float]:
    """Return (model, physical_cores, logical_cores, freq_max_mhz)."""
    model = platform.processor() or "unknown"
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    freq = psutil.cpu_freq()
    freq_max = freq.max if freq and freq.max else (freq.current if freq else 0.0)
    return model, physical, logical, freq_max


def _detect_ram() -> tuple[float, float]:
    """Return (total_gb, available_gb)."""
    mem = psutil.virtual_memory()
    return mem.total / (1024**3), mem.available / (1024**3)


def _detect_gpu() -> GPUInfo:
    """Try to detect GPU via nvidia-smi, then WMI."""
    gpu = GPUInfo()

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                gpu.name = parts[0].strip()
                gpu.driver = parts[1].strip()
                gpu.vram_gb = float(parts[2].strip()) / 1024
                gpu.api = "cuda"
                return gpu
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Try WMI for Windows
    try:
        import wmi as wmi_module  # type: ignore
        c = wmi_module.WMI(namespace="win32")
        for gpu_raw in c.Win32_VideoController():
            name = gpu_raw.Name or ""
            if "Microsoft Basic" not in name and "Microsoft Remote" not in name:
                gpu.name = name
                gpu.driver = gpu_raw.DriverVersion or ""
                # WMI reports adapter ram in bytes
                if gpu_raw.AdapterRAM:
                    gpu.vram_gb = gpu_raw.AdapterRAM / (1024**3)
                gpu.api = "directml"
                return gpu
    except ImportError:
        pass
    except Exception:
        pass

    return gpu


def _find_tool(name: str, command: str, version_args: list[str] | None = None) -> ToolInfo:
    """Find a tool on PATH and get its version."""
    path = shutil.which(command)
    if not path:
        return ToolInfo(name=name, path="", version="", available=False)

    version = ""
    if version_args:
        try:
            result = subprocess.run(
                [command] + version_args,
                capture_output=True, text=True, timeout=5,
            )
            output = (result.stdout + result.stderr).strip()
            # Take first line, truncate
            version = output.split("\n")[0][:100]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

    return ToolInfo(name=name, path=path, version=version, available=True)


def _detect_tools() -> dict[str, ToolInfo]:
    """Detect installed tools relevant to Pengu."""
    tools = {}

    # Python
    tools["python"] = _find_tool("Python", "python", ["--version"])
    tools["python3"] = _find_tool("Python3", "python3", ["--version"])

    # Git
    tools["git"] = _find_tool("Git", "git", ["--version"])

    # Node.js
    tools["node"] = _find_tool("Node.js", "node", ["--version"])
    tools["npm"] = _find_tool("npm", "npm", ["--version"])

    # VS Code
    tools["vscode"] = _find_tool("VS Code", "code", ["--version"])

    # PowerShell
    tools["powershell"] = _find_tool("PowerShell", "powershell", ["-Command", "$PSVersionTable.PSVersion.ToString()"])

    # Ollama
    tools["ollama"] = _find_tool("Ollama", "ollama", ["--version"])

    # WSL
    tools["wsl"] = _find_tool("WSL", "wsl", ["--version"])

    # Windows Terminal
    tools["wt"] = _find_tool("Windows Terminal", "wt", ["--version"])

    # pip
    tools["pip"] = _find_tool("pip", "pip", ["--version"])

    return tools


def _detect_storage() -> tuple[float, float]:
    """Return (total_gb, free_gb) for the system drive."""
    usage = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
    return usage.total / (1024**3), usage.free / (1024**3)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def classify_tier(info: HardwareInfo) -> HardwareTier:
    """
    Classify hardware into LOW / MEDIUM / HIGH.

    Decision logic:
      HIGH  — 32GB+ RAM  OR  16GB+ RAM + dedicated GPU with 4GB+ VRAM
      MEDIUM — 16GB+ RAM  OR  8GB+ RAM + GPU with 2GB+ VRAM
      LOW   — everything else
    """
    ram = info.ram_total_gb
    vram = info.gpu.vram_gb

    if ram >= 32 or (ram >= 16 and vram >= 4):
        return HardwareTier.HIGH
    elif ram >= 16 or (ram >= 8 and vram >= 2):
        return HardwareTier.MEDIUM
    else:
        return HardwareTier.LOW


# ---------------------------------------------------------------------------
# Model recommendations per tier
# ---------------------------------------------------------------------------

TIER_MODELS: dict[HardwareTier, dict[str, str]] = {
    HardwareTier.LOW: {
        "llm": "qwen2.5:1.5b",
        "coding": "qwen2.5-coder:1.5b",
        "vision": "",
        "stt": "distil-small.en",
        "tts": "kokoro",
        "wake_word": "openwakeword",
    },
    HardwareTier.MEDIUM: {
        "llm": "qwen2.5:7b",
        "coding": "qwen2.5-coder:7b",
        "vision": "qwen2-vl:2b",
        "stt": "distil-small.en",
        "tts": "kokoro",
        "wake_word": "openwakeword",
    },
    HardwareTier.HIGH: {
        "llm": "qwen2.5:14b",
        "coding": "qwen2.5-coder:14b",
        "vision": "qwen2-vl:7b",
        "stt": "distil-small.en",
        "tts": "kokoro",
        "wake_word": "openwakeword",
    },
}


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareInfo:
    """Run full hardware detection and classification."""
    info = HardwareInfo()

    # OS
    info.os_name, info.os_version, info.os_arch = _detect_os()
    info.hostname = platform.node()

    # CPU
    info.cpu_model, info.cpu_cores_physical, info.cpu_cores_logical, info.cpu_freq_max_mhz = _detect_cpu()

    # RAM
    info.ram_total_gb, info.ram_available_gb = _detect_ram()

    # GPU
    info.gpu = _detect_gpu()

    # Storage
    info.disk_total_gb, info.disk_free_gb = _detect_storage()

    # Tools
    info.tools = _detect_tools()

    # Classify
    info.tier = classify_tier(info)
    info.recommended_models = TIER_MODELS.get(info.tier, TIER_MODELS[HardwareTier.LOW])

    return info
