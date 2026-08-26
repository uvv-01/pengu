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

| Day | Feature | Status | Tested |
|-----|---------|--------|--------|
| 1 | Backend foundation, config, hardware detection, FastAPI | IMPLEMENTED | YES |
| 2 | Local model provider (LM Studio), intent router, command pipeline | IMPLEMENTED | YES |
| 3 | Real Windows OS control tools (apps, processes, filesystem, terminal) | IMPLEMENTED | YES |
| 4 | VS Code integration | IMPLEMENTED | YES |
| 5 | Memory system (SQLite), web search, browser interface | IMPLEMENTED | YES |
| 6 | Voice + vision provider foundations | PARTIAL | INTERFACES TESTED |
| 7 | Desktop UI (HTML/CSS/JS) | IMPLEMENTED | YES |
| 8 | Security audit, documentation, cleanup | IMPLEMENTED | YES |

### Capability Matrix

| Feature | Status | Local | Tested | Notes |
|---------|--------|-------|--------|-------|
| Text commands | IMPLEMENTED | YES | YES | Via API or UI |
| Intent router | IMPLEMENTED | YES | YES | 13 categories, deterministic first |
| Local model | IMPLEMENTED | YES | YES | LM Studio (gemma-4-e4b) |
| Deterministic tools | IMPLEMENTED | YES | YES | 25+ tools |
| OS control | IMPLEMENTED | YES | YES | Apps, processes, system info |
| Application control | IMPLEMENTED | YES | YES | Open, close, is_running |
| Process management | IMPLEMENTED | YES | YES | List, info, terminate |
| Filesystem | IMPLEMENTED | YES | YES | Read, write, list, search, grep |
| Terminal safety | IMPLEMENTED | YES | YES | Command allowlist, blocked patterns |
| VS Code integration | IMPLEMENTED | YES | YES | Open folder, file at line |
| Git tools | IMPLEMENTED | YES | YES | Status, log, diff, branch, commit |
| Memory system | IMPLEMENTED | YES | YES | SQLite, session + persistent |
| Web search | IMPLEMENTED | YES* | YES | DuckDuckGo (free, no API key) |
| Browser interface | PARTIAL | YES* | YES | Playwright abstraction |
| STT | PARTIAL | YES | INTERFACES | faster-whisper (needs install) |
| TTS | PARTIAL | YES* | INTERFACES | edge-tts (needs internet) |
| Wake word | PARTIAL | YES | INTERFACES | openWakeWord (needs install) |
| Vision | PARTIAL | YES | INTERFACES | LM Studio multimodal model |
| Desktop UI | IMPLEMENTED | YES | YES | HTML/CSS/JS, JARVIS-style |
| API | IMPLEMENTED | YES | YES | FastAPI + WebSocket |
| Security | IMPLEMENTED | YES | YES | Path validation, sensitive file blocking |
| Offline mode | IMPLEMENTED | YES | YES | Deterministic tools work offline |

*Requires internet connection but no API key.

## Quick Start

```bash
# Install
cd pengu
pip install -e .

# Run the server (opens UI at http://localhost:8420)
python -m pengu

# Or run with specific options
python -m pengu --host 127.0.0.1 --port 8420
```

## What Works Now

### Text Commands (via UI or API)

```
"what CPU do I have"         → system.info        → Real hardware details
"is VS Code running"         → application.is_running → "Yes, 23 processes"
"open VS Code"               → application.open    → Launches VS Code
"list files"                 → filesystem.list     → Real directory listing
"read README.md"             → filesystem.read     → File contents
"git status"                 → git.status          → Real git output
"remember I prefer dark mode" → memory.save        → Stored in SQLite
"search for dark mode"       → memory.search       → Found in memory
"search for Python docs"     → web_search          → DuckDuckGo results
"hello"                      → chat (LLM)          → Conversational response
"what is 2 + 2?"             → chat (LLM)          → "4"
```

### Deterministic-First Architecture

Pengu does NOT call an LLM when a deterministic tool can handle the request:

| Command | LLM Called? | Method |
|---------|-------------|--------|
| "open VS Code" | NO | application.open |
| "git status" | NO | git.status |
| "list files" | NO | filesystem.list |
| "what CPU do I have" | NO | system.info |
| "is VS Code running" | NO | application.is_running |
| "hello" | YES (if model loaded) | LLM chat |

## Architecture

```
User Input (text/voice)
    ↓
Intent Router (deterministic rules first)
    ↓
Tool Selection / Model Routing
    ↓
Tool Execution / LLM Inference
    ↓
Response Generation
    ↓
Output (text/speech/UI)
```

## Project Structure

```
pengu/
├── src/pengu/
│   ├── __init__.py
│   ├── __main__.py           # Entry point
│   ├── api.py                # FastAPI backend + UI serving
│   ├── config.py             # Configuration system
│   ├── state.py              # Core state machine
│   ├── router.py             # Intent router (deterministic rules)
│   ├── pipeline.py           # Command pipeline
│   ├── pipeline_handlers.py  # Memory, web, browser handlers
│   ├── models/               # Model abstraction
│   │   ├── base.py           # Provider ABC
│   │   └── lmstudio.py       # LM Studio provider
│   ├── os/                   # OS control tools
│   │   ├── app_manager.py    # Application discovery/management
│   │   ├── process_manager.py # Process inspection
│   │   ├── filesystem.py     # Secure filesystem ops
│   │   ├── terminal.py       # Safe terminal execution
│   │   ├── system_info.py    # System information
│   │   └── vscode.py         # VS Code integration
│   ├── tools/                # Tool registry
│   │   ├── registry.py       # Typed tool system
│   │   └── deterministic.py  # All deterministic tool handlers
│   ├── hardware/             # Hardware detection
│   ├── memory/               # SQLite memory system
│   ├── web/                  # Web search + browser
│   │   ├── search.py         # DuckDuckGo search
│   │   └── browser.py        # Playwright browser
│   ├── voice/                # Voice pipeline
│   │   ├── wake_word.py      # Wake word detection
│   │   ├── stt.py            # Speech-to-text
│   │   └── tts.py            # Text-to-speech
│   ├── vision/               # Vision system
│   │   ├── provider.py       # Vision analysis
│   │   └── screen.py         # Screen capture
│   └── ui/                   # Desktop interface
│       └── static/           # HTML/CSS/JS
├── tests/
├── docs/
├── pyproject.toml
└── LICENSE
```

## Configuration

```bash
# Environment variables (all optional)
PENGU_COST_MODE=FREE_ONLY          # Default: no paid services
PENGU_DEBUG=false
PENGU_API_PORT=8420

# Cloud providers (all optional, blocked in FREE_ONLY mode)
GEMINI_API_KEY=...                 # Optional
GROQ_API_KEY=...                   # Optional
OPENROUTER_API_KEY=...             # Optional
```

## Hardware Requirements

**Minimum:**
- Windows 10/11
- 4 GB RAM
- CPU with 2+ cores
- 1 GB disk space

**Recommended:**
- 8+ GB RAM
- CPU with 4+ cores
- GPU with 4+ GB VRAM (for faster LLM inference)

**Tested on:**
- Intel 8 cores / 12 threads
- ~23.7 GB RAM
- No GPU (CPU-only inference)

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with auto-reload
uvicorn pengu.api:app --reload --host 127.0.0.1 --port 8420

# Check hardware detection
python -m pengu.hardware.detect
```

## Security

- No hardcoded secrets or API keys
- Sensitive files blocked from memory storage
- Path traversal protection in filesystem tools
- Terminal command allowlist with blocked dangerous patterns
- Process termination requires explicit typed tools
- No unrestricted shell execution
- Cloud providers optional and blocked in FREE_ONLY mode

## License

Apache-2.0

## ⚠️ Cost Disclaimer

Pengu is designed to operate without paid services. Local functionality requires no
API payments. Optional cloud providers may have changing free-tier limits and are not
required. Never write "Pengu is 100% free forever" — write only what is verified.
