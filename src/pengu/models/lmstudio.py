"""
LM Studio provider — local model inference via OpenAI-compatible API.

LM Studio exposes an OpenAI-compatible REST API at http://localhost:1234/v1.
This provider talks to that endpoint using httpx.

Runtime: LM Studio (https://lmstudio.ai)
License: Free for personal and commercial use (https://lmstudio.ai/license)
Cost: £0 — runs entirely on local hardware.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from pengu.logging import AuditLogger, get_logger
from pengu.models.base import (
    ChatMessage,
    ChatResponse,
    ModelProvider,
    ProviderHealth,
    ProviderType,
)

logger = get_logger("pengu.models.lmstudio")
audit = AuditLogger()


class LMStudioProvider(ModelProvider):
    """
    Local model provider via LM Studio's OpenAI-compatible API.

    Connects to http://localhost:1234/v1 by default.
    No API key required — runs entirely locally.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(name="lmstudio", provider_type=ProviderType.LOCAL_OLLAMA)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def health_check(self) -> bool:
        """Check if LM Studio server is reachable and a model is loaded."""
        try:
            client = await self._get_client()
            resp = await client.get("/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    self.health.mark_healthy()
                    available = [m.get("id", "unknown") for m in models]
                    logger.info(
                        "lmstudio_healthy",
                        models=available,
                        base_url=self.base_url,
                    )
                    return True
                else:
                    self.health.mark_unhealthy("No models loaded in LM Studio")
                    return False
            else:
                self.health.mark_unhealthy(f"HTTP {resp.status_code}")
                return False
        except httpx.ConnectError:
            self.health.mark_unhealthy(
                "LM Studio server not running. Start LM Studio and load a model."
            )
            return False
        except Exception as e:
            self.health.mark_unhealthy(str(e))
            return False

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to LM Studio."""
        target_model = model or self.model
        if not target_model:
            # Auto-detect from available models
            try:
                client = await self._get_client()
                resp = await client.get("/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        target_model = models[0].get("id", "")
            except Exception:
                pass

        if not target_model:
            return ChatResponse(
                content="",
                provider="lmstudio",
                model="",
                error="No model specified and no model loaded in LM Studio",
            )

        # Build request
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        start = time.perf_counter()

        try:
            client = await self._get_client()
            resp = await client.post("/chat/completions", json=payload)
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                error_text = resp.text[:500]
                self.health.mark_unhealthy(f"HTTP {resp.status_code}: {error_text}")
                audit.log_provider_call(
                    provider="lmstudio",
                    model=target_model,
                    success=False,
                    duration_ms=latency_ms,
                    error=error_text[:200],
                )
                return ChatResponse(
                    content="",
                    provider="lmstudio",
                    model=target_model,
                    latency_ms=latency_ms,
                    error=f"LM Studio returned {resp.status_code}: {error_text[:200]}",
                )

            data = resp.json()
            self.health.mark_healthy(latency_ms)

            # Parse response
            choices = data.get("choices", [])
            if not choices:
                return ChatResponse(
                    content="",
                    provider="lmstudio",
                    model=target_model,
                    latency_ms=latency_ms,
                    error="Empty response from LM Studio",
                )

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls_raw = message.get("tool_calls", [])

            # Parse tool calls
            tool_calls = []
            for tc in tool_calls_raw:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )

            usage = data.get("usage", {})

            audit.log_provider_call(
                provider="lmstudio",
                model=target_model,
                success=True,
                duration_ms=latency_ms,
            )

            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                provider="lmstudio",
                model=target_model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                latency_ms=latency_ms,
            )

        except httpx.ConnectError:
            latency_ms = (time.perf_counter() - start) * 1000
            self.health.mark_unhealthy("LM Studio server not reachable")
            audit.log_provider_call(
                provider="lmstudio",
                model=target_model,
                success=False,
                duration_ms=latency_ms,
                error="Connection refused",
            )
            return ChatResponse(
                content="",
                provider="lmstudio",
                model=target_model,
                latency_ms=latency_ms,
                error="LM Studio server not running. Start LM Studio and load a model.",
            )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start) * 1000
            self.health.mark_unhealthy("Request timed out")
            audit.log_provider_call(
                provider="lmstudio",
                model=target_model,
                success=False,
                duration_ms=latency_ms,
                error="Timeout",
            )
            return ChatResponse(
                content="",
                provider="lmstudio",
                model=target_model,
                latency_ms=latency_ms,
                error="LM Studio request timed out. Model may be overloaded.",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self.health.mark_unhealthy(str(e))
            audit.log_provider_call(
                provider="lmstudio",
                model=target_model,
                success=False,
                duration_ms=latency_ms,
                error=str(e)[:200],
            )
            return ChatResponse(
                content="",
                provider="lmstudio",
                model=target_model,
                latency_ms=latency_ms,
                error=f"LM Studio error: {e}",
            )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream a chat completion from LM Studio."""
        target_model = model or self.model

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            client = await self._get_client()
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    yield f"[Error: LM Studio returned {resp.status_code}]"
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"[Error: {e}]"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
