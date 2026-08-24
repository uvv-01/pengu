# Pengu Architecture

## Overview

Pengu is a local-first autonomous desktop assistant built on these pillars:

1. **Deterministic First** — Don't use LLMs for things that can be hard-coded
2. **Local-First** — All core functionality works without internet or API keys
3. **₹0 Cost** — No paid services required; cloud is optional
4. **Real Functionality** — Every feature must actually work on the user's machine

## System Architecture

```
                    ┌──────────────────────────┐
                    │     DESKTOP UI (Electron) │  Day 7
                    │     React + TypeScript    │
                    └─────────┬────────────────┘
                              │ WebSocket
                    ┌─────────▼────────────────┐
                    │     FASTAPI BACKEND       │  Day 1
                    │  127.0.0.1:8420           │
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼─────┐  ┌─────▼──────┐  ┌─────▼──────┐
    │ STATE MACHINE  │  │   ROUTER   │  │  TOOL      │
    │ STANDBY→...   │  │ Intent→    │  │  REGISTRY  │
    │               │  │ Category→  │  │            │
    └───────────────┘  │ Provider   │  └────────────┘
                       └─────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │ DETERMINISTIC  │ │ LOCAL LLM  │ │  CLOUD     │
    │ TOOLS          │ │ (Ollama)   │ │ (Optional) │
    │ filesystem     │ │ Qwen       │ │ Gemini     │
    │ terminal       │ │            │ │ Groq       │
    │ applications   │ │            │ │            │
    │ git            │ │            │ │            │
    │ browser        │ │            │ │            │
    └───────────────┘ └────────────┘ └────────────┘
```

## State Machine

```
STANDBY → WAKE_DETECTED → ACTIVE → LISTENING → THINKING →
PLANNING → EXECUTING → SPEAKING → COMPLETE → STANDBY
```

On error at any stage → ERROR → STANDBY.
Emergency cancel (Ctrl+Shift+P) → STANDBY.

## Provider Fallback

```
USER REQUEST
     │
     ▼
DETERMINISTIC TOOL? ──YES──→ TOOL
     │
     NO
     │
     ▼
LOCAL MODEL ──SUFFICIENT──→ LOCAL
     │
     TOO WEAK
     │
     ▼
OPTIONAL CLOUD ──AVAILABLE──→ CLOUD
     │
     UNAVAILABLE
     │
     ▼
LOCAL FALLBACK
```

## Permission System

| Level | Risk | Examples | Confirmation |
|-------|------|----------|-------------|
| 0 - SAFE | Read-only | read files, list dirs, git status | No |
| 1 - LOW | Write/launch | create files, open apps, type text | No (configurable) |
| 2 - HIGH | Destructive | delete files, shell commands, git push | Yes |
| 3 - CRITICAL | System | disk ops, credentials, admin | Always |

## Hardware Classification

| Tier | RAM | GPU | Example |
|------|-----|-----|---------|
| LOW | < 16 GB | none | Budget laptop |
| MEDIUM | 16+ GB | none/2GB | Standard laptop |
| HIGH | 32+ GB | 4+ GB VRAM | Dev desktop |

Models are automatically selected based on tier.

## Directory Structure

```
pengu/src/pengu/
├── __init__.py          # Package root
├── __main__.py          # python -m pengu
├── api.py               # FastAPI backend
├── cli.py               # CLI entry point
├── config.py            # Configuration system
├── logging.py           # Structured logging
├── state.py             # State machine
├── core/                # Intent routing, agent loop (Day 2)
├── models/              # Model provider abstraction
│   └── base.py          # ModelProvider ABC
├── tools/               # Tool registry
│   └── registry.py      # Tool, ToolRegistry
├── hardware/            # Hardware detection
│   ├── detect.py        # Full hardware profiling
│   └── cli.py           # hw CLI
├── memory/              # SQLite memory (Day 9)
├── voice/               # Wake word, STT, TTS (Day 5-6)
└── utils/               # Shared utilities
```

## Day 1 Deliverable

Pengu backend starts successfully:
- Configuration loads from env vars / YAML
- Hardware is detected and classified
- FastAPI server responds on :8420
- State machine handles commands
- Tools are registered and callable
- Structured logs are produced
- All endpoints respond correctly
