"""
Pengu FastAPI Backend.

Endpoints:
  GET  /              — Pengu status page
  GET  /health        — health check
  GET  /config        — current configuration summary
  GET  /hardware      — hardware detection report
  GET  /state         — current assistant state
  GET  /tools         — registered tools
  POST /command       — send a text command
  WS   /ws            — WebSocket for live state updates

Default: 127.0.0.1:8420
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pengu.config import PenguConfig, AssistantState, get_config
from pengu.hardware.detect import detect_hardware, HardwareInfo
from pengu.logging import get_logger, setup_logging, new_task_id
from pengu.state import AssistantStateMachine
from pengu.tools.registry import ToolRegistry

logger = get_logger("pengu.api")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_config: PenguConfig | None = None
_state_machine = AssistantStateMachine()
_tool_registry = ToolRegistry()
_hardware_info: HardwareInfo | None = None
_connected_websockets: list[WebSocket] = []


def get_config_cached() -> PenguConfig:
    global _config
    if _config is None:
        _config = get_config()
    return _config


def get_hardware_cached() -> HardwareInfo:
    global _hardware_info
    if _hardware_info is None:
        _hardware_info = detect_hardware()
    return _hardware_info


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pengu",
    description="₹0-cost local-first autonomous desktop assistant",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    config = get_config_cached()
    setup_logging(
        level="DEBUG" if config.debug else "INFO",
        json_output=False,
    )
    logger.info(
        "pengu_starting",
        version=config.version,
        cost_mode=config.cost_mode.value,
    )

    # Run hardware detection
    hw = get_hardware_cached()
    logger.info(
        "hardware_detected",
        tier=hw.tier.value,
        ram_gb=round(hw.ram_total_gb, 1),
        gpu=hw.gpu.name,
    )

    logger.info("pengu_ready", port=config.api.port)


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("pengu_stopping")


# ---------------------------------------------------------------------------
# Broadcast to WebSockets
# ---------------------------------------------------------------------------

async def broadcast_state() -> None:
    """Send current state to all connected WebSockets."""
    message = json.dumps({
        "type": "state_update",
        "state": _state_machine.state.value,
        "task_id": _state_machine.task_id,
        "context": _state_machine.context,
    })
    dead: list[WebSocket] = []
    for ws in _connected_websockets:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connected_websockets.remove(ws)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict[str, Any]:
    config = get_config_cached()
    hw = get_hardware_cached()
    return {
        "name": "Pengu",
        "version": config.version,
        "status": "running",
        "state": _state_machine.state.value,
        "cost_mode": config.cost_mode.value,
        "hardware_tier": hw.tier.value,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "config": "/config",
            "hardware": "/hardware",
            "state": "/state",
            "tools": "/tools",
            "command": "/command",
            "websocket": "/ws",
        },
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    config = get_config_cached()
    return {
        "status": "healthy",
        "version": config.version,
        "cost_mode": config.cost_mode.value,
        "cloud_enabled": config.cloud_enabled(),
    }


@app.get("/config")
async def config() -> dict[str, Any]:
    config = get_config_cached()
    return config.summary()


@app.get("/hardware")
async def hardware() -> dict[str, Any]:
    hw = get_hardware_cached()
    return hw.to_dict()


@app.get("/state")
async def state() -> dict[str, Any]:
    return {
        "state": _state_machine.state.value,
        "task_id": _state_machine.task_id,
        "is_active": _state_machine.is_active,
        "transitions": _state_machine.get_transition_log()[-10:],
    }


@app.get("/tools")
async def tools() -> dict[str, Any]:
    return _tool_registry.to_dict()


@app.post("/command")
async def command(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Send a text command to Pengu.

    Body: {"text": "open VS Code", "user_id": "optional"}
    """
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "No command text provided"}

    task_id = new_task_id()
    logger.info("command_received", text=text, task_id=task_id)

    # For Day 1: echo back with state info
    # Day 2+ will route through the actual intent pipeline
    try:
        await _state_machine.activate()
        await _state_machine.start_listening()
        await _state_machine.think()

        # Day 1: deterministic response
        response_text = f"Received: '{text}'. Intent routing not yet implemented (Day 2)."

        await _state_machine.execute()
        await _state_machine.speak()
        await _state_machine.complete()

        return {
            "task_id": task_id,
            "input": text,
            "response": response_text,
            "provider": "deterministic",
            "model": "none",
            "state": _state_machine.state.value,
        }
    except Exception as e:
        await _state_machine.error(str(e))
        return {
            "task_id": task_id,
            "input": text,
            "error": str(e),
            "state": _state_machine.state.value,
        }


@app.post("/activate")
async def activate() -> dict[str, Any]:
    """Manually activate Pengu (as if wake word was detected)."""
    try:
        await _state_machine.activate()
        return {
            "state": _state_machine.state.value,
            "task_id": _state_machine.task_id,
            "message": "Pengu activated. Listening...",
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/cancel")
async def cancel() -> dict[str, Any]:
    """Emergency cancel (Ctrl+Shift+P equivalent)."""
    await _state_machine.cancel()
    return {
        "state": _state_machine.state.value,
        "message": "Emergency cancel issued.",
    }


# ---------------------------------------------------------------------------
# WebSocket — live state stream
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _connected_websockets.append(ws)
    logger.info("ws_connected", total=len(_connected_websockets))

    try:
        # Send current state immediately
        await ws.send_text(json.dumps({
            "type": "state_update",
            "state": _state_machine.state.value,
            "task_id": _state_machine.task_id,
        }))

        while True:
            # Keep connection alive, handle client messages
            data = await asyncio.wait_for(ws.receive_text(), timeout=30)
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            elif msg.get("type") == "activate":
                await _state_machine.activate()
                await ws.send_text(json.dumps({
                    "type": "state_update",
                    "state": _state_machine.state.value,
                    "task_id": _state_machine.task_id,
                }))

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        pass
    finally:
        if ws in _connected_websockets:
            _connected_websockets.remove(ws)
        logger.info("ws_disconnected", total=len(_connected_websockets))
