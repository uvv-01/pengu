"""
Pengu Configuration System.

Uses Pydantic Settings for environment variable and YAML-based configuration.
Priority: environment variables > YAML file > defaults.

Cost modes:
  FREE_ONLY          — no paid services, local-only (default)
  FREE_PLUS_CLOUD    — local-first, optional free cloud acceleration
"""

from __future__ import annotations

import enum
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CostMode(str, enum.Enum):
    """Controls whether optional cloud providers are allowed."""

    FREE_ONLY = "FREE_ONLY"
    FREE_PLUS_CLOUD = "FREE_PLUS_CLOUD"


class HardwareTier(str, enum.Enum):
    """Hardware classification for model selection."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AssistantState(str, enum.Enum):
    """Top-level state machine states."""

    STANDBY = "STANDBY"
    WAKE_DETECTED = "WAKE_DETECTED"
    ACTIVE = "ACTIVE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"
    COMPLETE = "COMPLETE"


class TaskCategory(str, enum.Enum):
    """Categories the router uses to select tools and models."""

    CHAT = "CHAT"
    SYSTEM_CONTROL = "SYSTEM_CONTROL"
    FILE_OPERATION = "FILE_OPERATION"
    CODING = "CODING"
    TERMINAL = "TERMINAL"
    BROWSER = "BROWSER"
    WEB_SEARCH = "WEB_SEARCH"
    VISION = "VISION"
    GIT = "GIT"
    NETWORK = "NETWORK"
    MEDIA = "MEDIA"
    MEMORY = "MEMORY"
    MISSIONS = "MISSIONS"
    MULTI_STEP_AGENT = "MULTI_STEP_AGENT"


class PermissionLevel(int, enum.Enum):
    """Permission levels for tool execution."""

    SAFE = 0          # read files, list dirs, screen capture, git status, open apps
    LOW_RISK = 1      # create files, edit files, launch programs, browser nav
    HIGH_RISK = 2     # delete files, unknown shell commands, install software, git push
    CRITICAL = 3      # disk ops, credential changes, security config, admin


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class APIConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = "127.0.0.1"
    port: int = 8420
    reload: bool = False
    log_level: str = "info"
    ws_heartbeat: int = 30


class HardwareConfig(BaseModel):
    """Hardware detection overrides (auto-detected if not set)."""

    tier: Optional[HardwareTier] = None
    max_ram_gb: Optional[float] = None
    has_gpu: Optional[bool] = None
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None


class ModelConfig(BaseModel):
    """Model selection — set by hardware detection or user override."""

    # General / routing model
    local_llm_model: str = ""  # auto-selected by hardware
    local_llm_runtime: str = "ollama"

    # Coding model
    local_coding_model: str = ""
    local_coding_runtime: str = "ollama"

    # Vision model
    local_vision_model: str = ""
    local_vision_runtime: str = "ollama"

    # STT model
    stt_model: str = ""  # auto-selected
    stt_runtime: str = "faster-whisper"

    # TTS model
    tts_model: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_runtime: str = "kokoro"

    # Wake word
    wake_word_model: str = ""
    wake_word_runtime: str = "openwakeword"
    wake_word_phrase: str = "hello pengu"


class CloudProviderConfig(BaseModel):
    """Optional cloud provider settings — never required."""

    # Gemini
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    gemini_enabled: bool = False

    # Groq
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = "llama-3.1-8b-instant"
    groq_enabled: bool = False

    # OpenRouter
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = ""
    openrouter_enabled: bool = False


class SecurityConfig(BaseModel):
    """Security and permission settings."""

    require_confirmation_high_risk: bool = True
    require_confirmation_critical: bool = True
    max_shell_command_length: int = 10_000
    blocked_commands: list[str] = Field(default_factory=lambda: [
        "format", "rm -rf /", "del /s", "rd /s", "cipher",
    ])
    auto_approve_safe: bool = True
    auto_approve_low_risk: bool = True


class UIConfig(BaseModel):
    """Desktop UI configuration."""

    theme: str = "dark"
    overlay_opacity: float = 0.92
    overlay_width: int = 400
    overlay_height: int = 600
    always_on_top: bool = True
    start_minimized: bool = False


class VoiceConfig(BaseModel):
    """Voice pipeline configuration."""

    sample_rate: int = 16000
    channels: int = 1
    wake_word_sensitivity: float = 0.5
    stt_language: str = "en"
    tts_speed: float = 1.0
    mic_device_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

class PenguConfig(BaseSettings):
    """Root configuration for the Pengu assistant."""

    model_config = SettingsConfigDict(
        env_prefix="PENGU_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core
    version: str = "0.1.0"
    cost_mode: CostMode = CostMode.FREE_ONLY
    debug: bool = False
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")

    # Sub-configs
    api: APIConfig = Field(default_factory=APIConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cloud: CloudProviderConfig = Field(default_factory=CloudProviderConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # -- Validators --

    @field_validator("cost_mode", mode="before")
    @classmethod
    def validate_cost_mode(cls, v: Any) -> CostMode:
        if isinstance(v, str):
            v = v.upper().strip()
        return CostMode(v)

    # -- Helpers --

    def ensure_dirs(self) -> None:
        """Create data and log directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def is_free_only(self) -> bool:
        """Check if we are in strict ₹0 mode."""
        return self.cost_mode == CostMode.FREE_ONLY

    def cloud_enabled(self) -> bool:
        """Check if any cloud provider is enabled."""
        if self.is_free_only():
            return False
        return any([
            self.cloud.gemini_enabled and self.cloud.gemini_api_key,
            self.cloud.groq_enabled and self.cloud.groq_api_key,
            self.cloud.openrouter_enabled and self.cloud.openrouter_api_key,
        ])

    def summary(self) -> dict[str, Any]:
        """Return a human-readable configuration summary."""
        return {
            "version": self.version,
            "cost_mode": self.cost_mode.value,
            "debug": self.debug,
            "local_llm": self.model.local_llm_model or "(not configured)",
            "local_coding": self.model.local_coding_model or "(not configured)",
            "local_vision": self.model.local_vision_model or "(not configured)",
            "stt": self.model.stt_model or "(not configured)",
            "tts": self.model.tts_model,
            "wake_word": self.model.wake_word_phrase,
            "cloud_enabled": self.cloud_enabled(),
            "api_host": self.api.host,
            "api_port": self.api.port,
        }


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_config: Optional[PenguConfig] = None


def load_config(config_path: Optional[str | Path] = None) -> PenguConfig:
    """
    Load configuration with priority:
      1. Environment variables (PENGU_*)
      2. YAML file
      3. Defaults
    """
    global _config

    yaml_data: dict[str, Any] = {}

    if config_path is None:
        # Check default locations
        candidates = [
            Path("pengu.yaml"),
            Path("pengu.yml"),
            Path("config/pengu.yaml"),
            Path("config/pengu.yml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

    # Build config, allowing env vars to override YAML
    _config = PenguConfig(**yaml_data)
    _config.ensure_dirs()

    return _config


def get_config() -> PenguConfig:
    """Get the global configuration (loads if needed)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
