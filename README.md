# Pengu

> A real ₹0-cost, local-first autonomous desktop assistant for Windows.

## What Is This?

Pengu is a voice-controlled computer-use assistant. It runs entirely on your machine
using local AI models, free/open-source software, and deterministic tooling.

**"Hello Pengu"** → wake up → voice command → real action → spoken response.

Pengu is NOT a chatbot demo. It is a real assistant that:

- Listens for a wake word ("Hello Pengu")
- Transcribes your voice locally (no paid STT)
- Understands intent using local models (no paid LLM required)
- Executes real actions: opens apps, edits files, runs terminals, uses Git, browses the web
- Speaks responses using local TTS (no paid TTS)

## ₹0 Cost Guarantee

Pengu is designed to operate without paid services. Local functionality requires no API
payments. Optional cloud providers may have changing free-tier limits and are not required.

The system works when ALL API keys are removed and ALL internet is disconnected.

## Current Status

| Day | Feature | Status |
|-----|---------|--------|
| 1 | Backend foundation, config, hardware detection, FastAPI | IMPLEMENTED |
| 2 | Local model provider (LM Studio), intent router, command pipeline | IMPLEMENTED |
| 3 | Real Windows OS control tools (apps, processes, filesystem, terminal, system info) | IMPLEMENTED |
| 4 | VS Code integration | IMPLEMENTED |
| 5 | Voice pipeline (wake word, STT, TTS) | NOT IMPLEMENTED |
| 6 | Electron desktop UI | NOT IMPLEMENTED |
| 7 | Browser automation | NOT IMPLEMENTED |
| 8 | Vision/screen capture | NOT IMPLEMENTED |
| 9 | Memory system | NOT IMPLEMENTED |
| 10 | Full autonomous agent loop | NOT IMPLEMENTED |

### Day 3 — What Works Now

Pengu can execute real Windows commands:

```
"what CPU do I have"      → system.info       → Real hardware details
"is VS Code running"      → application.is_running → "Yes, 23 processes"
"open VS Code"            → application.open   → Launches VS Code
"list files"              → filesystem.list    → Real directory listing
"read README.md"          → filesystem.read    → File contents
"git status"              → git.status         → Real git output
"hello"                   → chat               → Conversational response
```

**Deterministic-first**: Most commands execute without calling any LLM. Only ambiguous
or conversational requests use the local model.

## Quick Start

```bash
# Clone
git clone https://github.com/uvv-01/pengu.git
cd pengu

# Install
pip install -e ".[dev]"

# Check hardware
python -m pengu.hardware.detect

# Start backend
python -m pengu
# or: uvicorn pengu.api:app --host 127.0.0.1 --port 8420
```

## Architecture

```
Hello Pengu
     ↓
Wake Word Detection (openWakeWord)          [NOT IMPLEMENTED]
     ↓
Voice Activity Detection                    [NOT IMPLEMENTED]
     ↓
Speech-to-Text (faster-whisper)             [NOT IMPLEMENTED]
     ↓
Intent Router (deterministic rules first)
     ↓
┌─────────────────────────────────────────┐
│ SYSTEM_CONTROL  → application tools     │ ← IMPLEMENTED
│ FILE_OPERATION  → filesystem tools      │ ← IMPLEMENTED
│ GIT             → git tools             │ ← IMPLEMENTED
│ TERMINAL        → safe terminal         │ ← IMPLEMENTED
│ NETWORK         → network tools         │ ← IMPLEMENTED
│ CODING          → local LLM             │ ← IMPLEMENTED (needs model)
│ CHAT            → local LLM             │ ← IMPLEMENTED (needs model)
│ VISION          → screen capture        │ [NOT IMPLEMENTED]
│ BROWSER         → playwright            │ [NOT IMPLEMENTED]
└─────────────────────────────────────────┘
     ↓
Verification
     ↓
Text-to-Speech (Kokoro TTS)                [NOT IMPLEMENTED]
     ↓
Response
```

## OS Control Tools (Day 3)

### Application Management
- `application.open` — Launch an application by name
- `application.close` — Gracefully close an application
- `application.is_running` — Check if an app is running
- `application.list_installed` — List discovered installed apps
- `application.list_running` — List apps with visible windows

### Process Management
- `process.list` — List processes with optional filtering
- `process.info` — Get details for a specific PID
- `process.terminate` — Safely terminate a process

### Filesystem
- `filesystem.read_file` — Read file contents (blocks sensitive files)
- `filesystem.write_file` — Write to files (validates paths)
- `filesystem.list_directory` — List directory contents
- `filesystem.search_files` — Glob-based file search
- `filesystem.grep` — Search file contents

### Terminal (Safe)
- `terminal.execute` — Execute commands with allowlist validation
- Blocked: `format`, `del /s`, `shutdown`, encoded PowerShell, etc.

### System Information
- `system.info` — CPU, RAM, GPU, disk, uptime, installed tools

### VS Code
- `vscode.open_folder` — Open a project in VS Code
- `vscode.open_file` — Open a file (optionally at a line)
- `vscode.focus` — Bring VS Code to foreground

### Git
- `git.status`, `git.log`, `git.diff`, `git.execute`

## Security Model

- **Typed tools only** — LLM never executes arbitrary shell commands
- **Command allowlist** — Only safe commands execute without confirmation
- **Sensitive file blocking** — `.env`, SSH keys, credentials are blocked
- **Process protection** — System-critical processes cannot be terminated
- **Path validation** — File operations validate paths before execution

## Hardware Requirements

| Tier | RAM | GPU VRAM | Example |
|------|-----|----------|---------|
| LOW | 8 GB | none | Budget laptop |
| MEDIUM | 16 GB | none/2GB | Standard laptop |
| HIGH | 32+ GB | 4+ GB | Development desktop |

Pengu detects your hardware and recommends appropriate models.

## Project Structure

```
pengu/
├── src/pengu/
│   ├── __init__.py
│   ├── api.py              # FastAPI backend
│   ├── config.py           # Configuration system
│   ├── state.py            # Core state machine
│   ├── router.py           # Intent router (deterministic rules)
│   ├── pipeline.py         # Command pipeline
│   ├── models/             # Model abstraction
│   │   ├── base.py         # Provider ABC
│   │   └── lmstudio.py     # LM Studio provider
│   ├── os/                 # OS control tools (Day 3)
│   │   ├── app_manager.py  # Application discovery/management
│   │   ├── process_manager.py # Process inspection
│   │   ├── filesystem.py   # Secure filesystem ops
│   │   ├── terminal.py     # Safe terminal execution
│   │   ├── system_info.py  # System information
│   │   └── vscode.py       # VS Code integration
│   ├── tools/              # Tool registry
│   │   ├── registry.py     # Typed tool system
│   │   └── deterministic.py # All deterministic tool handlers
│   ├── hardware/           # Hardware detection
│   ├── memory/             # Memory system (NOT IMPLEMENTED)
│   └── voice/              # Voice pipeline (NOT IMPLEMENTED)
├── tests/
├── docs/
├── pyproject.toml
└── LICENSE
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with auto-reload
uvicorn pengu.api:app --reload --host 127.0.0.1 --port 8420

# Check hardware detection
python -m pengu.hardware.detect
```

## License

Apache-2.0

## ⚠️ Cost Disclaimer

Pengu is designed to operate without paid services. Local functionality requires no
API payments. Optional cloud providers may have changing free-tier limits and are not
required. Never write "Pengu is 100% free forever" — write only what is verified.
