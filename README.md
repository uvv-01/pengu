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

## Quick Start

```bash
# Clone
git clone <repo-url>
cd pengu

# Install
pip install -e ".[dev]"

# Check hardware
python -m pengu.hardware.detect

# Start backend
uvicorn pengu.api:app --host 127.0.0.1 --port 8420
```

## Architecture

```
Hello Pengu
     ↓
Wake Word Detection (openWakeWord)
     ↓
Voice Activity Detection
     ↓
Speech-to-Text (faster-whisper)
     ↓
Intent Router (deterministic + local LLM)
     ↓
Tool Execution (filesystem, terminal, apps, Git, browser, ...)
     ↓
Verification
     ↓
Text-to-Speech (Kokoro TTS)
     ↓
Response
```

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
│   ├── core/               # Core engine (Day 2)
│   ├── models/             # Model abstraction (Day 2)
│   ├── tools/              # Tool system (Day 3)
│   ├── memory/             # Memory system (Day 9)
│   ├── voice/              # Voice pipeline (Day 5-6)
│   ├── hardware/           # Hardware detection
│   └── utils/              # Utilities
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
