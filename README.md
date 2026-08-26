# Pengu

**₹0-cost local-first voice-first Windows desktop assistant.**

Pengu runs entirely on your local hardware. No paid APIs, no cloud dependency, no subscription.

Say **"Hello Pengu"** → Pengu listens → you speak a command → Pengu acts → Pengu speaks the result.

## Architecture

```
User says "Hello Pengu"
        ↓
Streaming Microphone (sounddevice)
        ↓
Energy-based VAD (voice activity detection)
        ↓
Speech segment detected
        ↓
faster-whisper STT (local, CPU, int8)
        ↓
Transcription checked for "hello pengu"
        ↓
ACKNOWLEDGE ("Yes?")
        ↓
Listen for command (VAD-based recording)
        ↓
STT transcribes command
        ↓
Deterministic Command Parser (regex, zero-cost)
        ↓
Application launcher / Filesystem / Git / Browser / System
        ↓
edge-tts speaks result (with echo protection)
        ↓
Return to STANDBY
```

## Features

### IMPLEMENTED

| Feature | Status | Local | Tested |
|---------|--------|-------|--------|
| Voice wake word ("Hello Pengu") | IMPLEMENTED | YES | YES |
| Speech-to-text (faster-whisper tiny) | IMPLEMENTED | YES | YES |
| Text-to-speech (edge-tts) | IMPLEMENTED | YES | YES |
| TTS barge-in / cancellation | IMPLEMENTED | YES | YES |
| Echo protection (mic muted during TTS) | IMPLEMENTED | YES | YES |
| Deterministic command routing | IMPLEMENTED | YES | YES |
| Application launcher (VS Code, Chrome, Edge, Explorer, etc.) | IMPLEMENTED | YES | YES |
| VS Code integration (open folder/file) | IMPLEMENTED | YES | YES |
| Google search via browser | IMPLEMENTED | YES | YES |
| ChatGPT URL opening | IMPLEMENTED | YES | YES |
| File operations (create file/folder) | IMPLEMENTED | YES | YES |
| Git commands (status, log, diff) | IMPLEMENTED | YES | YES |
| System information | IMPLEMENTED | YES | YES |
| Local LLM (LM Studio, Ollama) | IMPLEMENTED | YES | YES |
| Model auto-discovery | IMPLEMENTED | YES | YES |
| Desktop overlay (tkinter, always-on-top) | IMPLEMENTED | YES | YES |
| System tray (pystray) | IMPLEMENTED | YES | YES |
| SQLite memory with privacy filtering | IMPLEMENTED | YES | YES |
| Diagnostics command | IMPLEMENTED | YES | YES |
| Deterministic-first (no LLM for simple commands) | IMPLEMENTED | YES | YES |
| 335 automated tests | IMPLEMENTED | YES | YES |

### PARTIAL

| Feature | Status | Notes |
|---------|--------|-------|
| Local LLM inference speed | PARTIAL | ~2-30s on CPU-only hardware |
| Custom "Hello Pengu" wake model | PARTIAL | Uses STT transcription; not a trained wake-word model |
| Voice input | PARTIAL | Requires Windows microphone unmuted and with adequate gain |

### NOT IMPLEMENTED

| Feature | Status |
|---------|--------|
| True custom wake-word model (openWakeWord training) | NOT IMPLEMENTED |
| Vision / screen analysis | NOT IMPLEMENTED |
| Browser page interaction (Playwright) | NOT IMPLEMENTED |
| Windows startup integration | NOT IMPLEMENTED |
| Global hotkey (Ctrl+Alt+P) | NOT IMPLEMENTED |
| Electron desktop UI | NOT IMPLEMENTED |

## Quick Start

### Prerequisites

- Python 3.10+
- Windows 10/11
- Microphone (unmuted in Windows Sound Settings)
- Speaker or headphones

### Install

```bash
git clone https://github.com/uvv-01/pengu.git
cd pengu
pip install -e ".[voice]"
```

### Run Diagnostics

```bash
python -m pengu diagnostics
```

### Start Pengu

```bash
python -m pengu
```

### Voice Commands

1. Say **"Hello Pengu"** (wait for "Yes?")
2. Say your command, for example:
   - "Open VS Code"
   - "Open Chrome"
   - "Open File Explorer"
   - "Open my Pengu project in VS Code"
   - "Search Google for Python decorators"
   - "Open ChatGPT"
   - "What CPU do I have"
   - "Git status"
   - "Create a file called hello.py"
   - "What is the meaning of life?" (uses local LLM)

## Local Model Setup

Pengu supports local LLMs via LM Studio or Ollama.

### LM Studio (Recommended)

1. Download LM Studio: https://lmstudio.ai
2. Load a model (e.g., Qwen3, Gemma, Phi)
3. Start the local server (default: http://localhost:1234)
4. Pengu auto-detects the running model

### Ollama

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull qwen3:1.5b`
3. Pengu auto-detects available models

## Project Structure

```
pengu/
├── src/pengu/
│   ├── app.py              # Main application + command parser
│   ├── voice/
│   │   ├── engine.py       # Voice engine (VAD, wake word, STT, TTS)
│   │   ├── stt.py          # Speech-to-text provider
│   │   └── tts.py          # Text-to-speech provider
│   ├── os/
│   │   ├── app_launcher.py # Windows application launcher
│   │   ├── filesystem.py   # Filesystem operations
│   │   ├── process_manager.py
│   │   ├── system_info.py
│   │   ├── terminal.py
│   │   └── vscode.py
│   ├── models/
│   │   ├── base.py         # Model provider abstraction
│   │   └── lmstudio.py     # LM Studio provider
│   ├── memory/             # SQLite persistent memory
│   ├── web/                # Web search (DuckDuckGo)
│   ├── ui/
│   │   ├── overlay.py      # Desktop overlay (tkinter)
│   │   └── tray.py         # System tray (pystray)
│   ├── tools/              # Tool registry
│   ├── router.py           # Intent router
│   ├── config.py           # Configuration
│   └── api.py              # FastAPI server
├── tests/                  # 335 tests
├── docs/
└── pyproject.toml
```

## Security

- Deterministic-first routing (no LLM for simple commands)
- Terminal command allowlist
- Filesystem path validation
- Sensitive file protection (.env, SSH keys, credentials)
- No arbitrary shell execution by LLM
- Echo protection (mic muted during TTS)

## Hardware

- **CPU**: Intel 8 cores / 12 threads
- **RAM**: ~23.7 GB
- **GPU**: None (CPU-only)
- **OS**: Windows 10

Pengu runs comfortably on this hardware. STT (tiny model) loads in ~2s. TTS via edge-tts is fast.

## Testing

```bash
pytest                    # Run all 335 tests
pytest tests/test_voice_engine.py  # Voice engine tests (63 tests)
pytest tests/test_os_tools.py      # OS tools tests
```

## License

Apache-2.0
