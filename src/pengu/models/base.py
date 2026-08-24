"""
Base model provider abstraction.

Every provider (local Ollama, Gemini, Groq, etc.) implements this interface.
Providers are NEVER required — Pengu works without any of them via deterministic tools.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional


class ProviderType(str, Enum):
    LOCAL_OLLAMA = "local_ollama"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    DETERMINISTIC = "deterministic"  # not a real provider, just tool dispatch


class ProviderHealth:
    """Health status of a provider."""

    def __init__(self) -> None:
        self.available: bool = False
        self.latency_ms: float = 0
        self.error: str = ""
        self.last_checked: float = 0
        self.consecutive_failures: int = 0

    def mark_healthy(self, latency_ms: float = 0) -> None:
        self.available = True
        self.latency_ms = latency_ms
        self.error = ""
        self.last_checked = time.time()
        self.consecutive_failures = 0

    def mark_unhealthy(self, error: str) -> None:
        self.available = False
        self.error = error
        self.last_checked = time.time()
        self.consecutive_failures += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "last_checked": self.last_checked,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0
    error: str = ""
    is_fallback: bool = False


class ModelProvider(ABC):
    """Base class for all model providers."""

    def __init__(self, name: str, provider_type: ProviderType) -> None:
        self.name = name
        self.provider_type = provider_type
        self.health = ProviderHealth()
        self._is_required: bool = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if this provider is available right now."""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat completion."""
        yield ""  # pragma: no cover

    async def structured_output(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        model: str = "",
    ) -> dict[str, Any]:
        """
        Request structured output matching a JSON schema.
        Default: wrap in a prompt instruction. Override for native support.
        """
        schema_instruction = f"Respond ONLY with valid JSON matching this schema: {schema}"
        enhanced = messages + [ChatMessage(role="system", content=schema_instruction)]
        resp = await self.chat(enhanced, model=model)
        # Attempt to parse as JSON
        import json
        try:
            return json.loads(resp.content)
        except (json.JSONDecodeError, ValueError):
            return {"error": "Failed to parse structured output", "raw": resp.content}

    def is_available(self) -> bool:
        """Quick check without network call."""
        return self.health.available

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.provider_type.value,
            "health": self.health.to_dict(),
            "is_required": self._is_required,
        }
