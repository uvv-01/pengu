# Pengu — Local-First Windows Voice AI Assistant

A local-first, voice-first desktop AI assistant for Windows that can understand natural language, control applications, interact with browsers, manage files, and autonomously execute multi-step tasks — all running entirely on the user's machine.

## Features

### Voice Interaction
- **Wake-word detection** via STT-based phrase matching ("Hello Pengu")
- **Speech-to-Text** using Faster Whisper (tiny model, CPU-optimized)
- **Text-to-Speech** for voice responses
- **Barge-in support** — interrupt TTS by speaking
- **TTS event-loop fix** — proper async handling for voice callbacks

### Application Control
- Open/close/focus any installed application
- Open folders (Downloads, Documents, Desktop, etc.) and files
- VS Code integration (open projects in VS Code)
- Task Manager, Settings, Terminal access

### Browser Automation (Playwright)
- Navigate to URLs
- Search Google / ChatGPT via browser
- Click elements by visible text, ARIA role, label, or placeholder
- Type into input fields
- Submit forms
- Read page content
- Verify page state after actions
- Structured browser state observation (URL, title, interactive elements)

### Desktop Interaction
- Mouse: click, double-click, right-click, scroll
- Keyboard: type text, press keys, keyboard shortcuts (Ctrl+C, Alt+Tab)
- Window: find, focus, restore windows
- Screen: active window detection, UI element enumeration via Windows UI Automation

### System Information
- Battery status and charging info
- CPU, RAM, storage, OS details
- Wi-Fi network information
- Volume control (get/set/mute/unmute)
- Wallpaper management
- Process listing and inspection

### File System
- List directory contents
- Read files
- Create files and folders
- Navigate to folders

### Git Integration
- git status, log, diff, branch, add, commit, push, pull, merge

### Web Search
- DuckDuckGo search (free, no API key)
- URL content fetching

## Architecture

```
Voice Input
    |
    v
STT (Faster Whisper)
    |
    v
Wake Word Detection ("Hello Pengu")
    |
    v
Command Parser (deterministic-first)
    |
    +---> Safety Policy Check
    |         |
    |         +---> BLOCKED -> refuse
    |         +---> HIGH/MEDIUM_RISK -> confirm via voice
    |         +---> SAFE/LOW_RISK -> proceed
    |
    +---> Deterministic Router (Intent Router)
    |         |
    |         +---> SYSTEM_CONTROL -> application tools
    |         +---> FILE_OPERATION -> filesystem tools
    |         +---> GIT -> git tools
    |         +---> BROWSER -> browser interaction
    |         +---> WEB_SEARCH -> DuckDuckGo
    |         +---> VISION -> screen capture + analysis
    |         +---> NETWORK -> Wi-Fi/network tools
    |         +---> MEMORY -> memory save/search/forget
    |         +---> MISSIONS -> scheduler/missions
    |         +---> CHAT -> local LLM (LM Studio)
    |
    +---> Agent Brain (for complex/multi-step tasks)
              |
              +---> Understand goal
              +---> Observe world state
              +---> Plan steps
              +---> Execute actions through tools
              +---> Observe results
              +---> Verify success
              +---> Recover on failure
              +---> Replan if needed
              +---> Respond to user
    |
    v
TTS Response
    |
    v
Return to STANDBY
```

### Key Design Principles

- **Deterministic First**: Common commands are handled by regex/keyword rules with zero latency and zero cost. LLM is only used when rules cannot handle the request.
- **CPU-Only**: All models run locally. No cloud dependency for core features.
- **Safety-First**: Every action passes through a central safety policy before execution.
- **Modular**: Voice, routing, tools, browser, desktop, and safety are separate modules.

## Implementation Status

### Implemented and Verified

| Feature | Status | Notes |
|---------|--------|-------|
| Voice wake detection | Implemented | STT-based "Hello Pengu" phrase matching |
| STT (Speech-to-Text) | Implemented | Faster Whisper tiny model, CPU |
| TTS (Text-to-Speech) | Implemented | Kokoro TTS with barge-in |
| Application opening | Implemented | Chrome, VS Code, Explorer, Notepad, etc. |
| Folder navigation | Implemented | Downloads, Documents, Desktop, etc. |
| URL opening | Implemented | Chrome, Edge, Firefox |
| System information | Implemented | Battery, CPU, RAM, network |
| Volume control | Implemented | Get, set, mute, unmute |
| Wallpaper management | Implemented | Set wallpaper from file |
| Git commands | Implemented | Status, log, diff, branch, etc. |
| Web search | Implemented | DuckDuckGo (free) |
| Browser automation | Implemented | Playwright-based (click, type, navigate, verify) |
| Desktop interaction | Implemented | Mouse, keyboard, window management |
| Intent routing | Implemented | Deterministic rules + LLM fallback |
| LLM chat | Implemented | LM Studio / Ollama local models |
| Safety policy | Implemented | 5-level risk classification + confirmation |
| Scheduler | Implemented | One-time, delayed, recurring tasks with SQLite persistence |
| Memory | Implemented | Persistent SQLite memory with privacy filtering |
| Context resolution | Implemented | Pronouns, follow-ups, preference resolution |
| Agent brain | Implemented | Observe -> Plan -> Act -> Verify -> Recover loop |
| Multi-app execution | Implemented | Agent can plan across multiple applications |
| Loop detection | Implemented | Prevents infinite repeated actions |
| Global hotkey | Implemented | Ctrl+Alt+P toggle |
| System tray | Implemented | Start, pause, resume, exit |
| Desktop overlay | Implemented | Visual state indicator |

### Partially Implemented / Experimental

| Feature | Status | Notes |
|---------|--------|-------|
| Real Playwright browser interaction | Partial | Requires `playwright install chromium` manual setup |
| Vision/screen analysis | Partial | Screenshot capture works; vision model integration needs API |
| UI Automation tree | Partial | PowerShell-based; depth-limited |
| Voice confirmation flow | Partial | Safety policy classifies actions; voice confirmation needs UI wiring |
| Background scheduler loop | Partial | Scheduler module works; background loop starts in app.py but needs runtime testing |
| Memory preference resolution | Partial | "My browser" resolves from persistent memory if available |
| OpenWakeWord custom model | Not implemented | Currently uses STT-based wake phrase detection |

### Not Implemented

| Feature | Status |
|---------|--------|
| Custom wake-word model training | Not implemented |
| Full Electron desktop UI | Not implemented |
| Playwright browser page interaction (real DOM) | Requires manual Playwright install |
| Full Windows UI Automation (deep tree) | Limited to top-level elements |
| Multi-turn vision analysis | Screenshot + vision model needs API |

## Configuration

### Requirements

- Python 3.11+
- Windows 10/11
- LM Studio (for local LLM) or Ollama
- Microphone for voice input
- Speakers for TTS output

### Environment Variables

```bash
# Optional: Cloud providers (PENGU_ prefix)
PENGU_GEMINI_API_KEY=...
PENGU_GROQ_API_KEY=...
```

### Setup

```bash
pip install -e .
python -m pengu
```

## Testing

```bash
# Run all tests (excluding voice engine tests that need hardware)
python -m pytest tests/ --ignore=tests/test_voice_engine.py --ignore=tests/test_voice_vision.py

# Run specific test files
python -m pytest tests/test_phases_6_7_8_9_integration.py -v
python -m pytest tests/test_safety_scheduler_memory.py -v
python -m pytest tests/test_browser_interaction.py -v

# Startup verification
python -m pengu
# Verify: STARTING -> STANDBY -> pengu_ready
```

### Test Results

```
557 passed, 4 warnings
```

- Unit tests for all modules
- Integration tests for safety, scheduler, memory
- Browser interaction tests (mocked Playwright)
- Agent brain planning tests
- Pipeline routing tests
- Context resolution tests

## Safety

### Risk Classification

| Level | Examples | Behavior |
|-------|----------|----------|
| SAFE | Read system info, check battery, search web | Execute immediately |
| LOW_RISK | Open app, navigate page, type text | Execute immediately |
| MEDIUM_RISK | Install software, create files, git push | Confirm before execution |
| HIGH_RISK | Delete files, force push, uninstall | Confirm with explanation |
| BLOCKED | Format disk, destructive commands | Refuse execution |

### Confirmation Flow

1. Action classified by RiskClassifier
2. HIGH/MEDIUM_RISK actions generate confirmation message
3. User confirms via voice ("yes"/"no") or text
4. Session permissions granted for confirmed actions
5. BLOCKED actions always refuse

## Memory

- **Session memory**: Current conversation context (clears on restart)
- **Persistent memory**: SQLite-backed (survives restarts)
- **Privacy filtering**: Rejects passwords, API keys, tokens, secrets
- **Categories**: preference, project, task, reminder, summary, general
- **Commands**: "Remember...", "Forget...", "What do you know about..."

## Scheduler

- **One-time**: Execute once at specified time
- **Delayed**: Execute after N minutes/hours
- **Recurring**: Execute every day/week/hour
- **Persistence**: SQLite-backed (survives restarts)
- **Retry**: Bounded retry with failure tracking
- **Cancellation**: Cancel by ID or description

## License

Private project by Yuvraj Singh (uvv-01).
