# Pengu Setup Guide

## Prerequisites

- Python 3.10+ (3.11+ recommended)
- Git
- 8+ GB RAM (16+ recommended)

## Quick Start (Backend Only)

```bash
# Clone the repository
git clone <repo-url>
cd pengu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Run hardware detection
python -m pengu.hardware.detect

# Start the backend server
python -m pengu
```

The server starts at http://127.0.0.1:8420

## Verify Installation

```bash
# Check health
curl http://127.0.0.1:8420/health

# Check hardware
curl http://127.0.0.1:8420/hardware

# Open API docs
# Navigate to http://127.0.0.1:8420/docs
```

## Optional: Install Voice Stack

```bash
# Install voice dependencies
pip install -e ".[voice]"

# For STT: download the model (happens on first use)
# For TTS: Kokoro model downloads on first use
# For wake word: openWakeWord model downloads on first use
```

## Optional: Install Ollama (Local LLM)

```bash
# Install Ollama from https://ollama.com
# Then pull recommended models:
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b

# Verify
ollama list
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key settings:

```bash
PENGU_COST_MODE=FREE_ONLY          # or FREE_PLUS_CLOUD
PENGU_DEBUG=false
PENGU_API__PORT=8420
```

### YAML Configuration

Create `pengu.yaml` in the project root:

```yaml
debug: false
cost_mode: FREE_ONLY

api:
  host: 127.0.0.1
  port: 8420

model:
  local_llm_model: qwen2.5:7b
  stt_model: distil-small.en
  tts_model: kokoro
```

Priority: environment variables > YAML file > defaults.

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=pengu

# Specific test file
pytest tests/test_config.py -v
```

## Cost Modes

### FREE_ONLY (Default)

No paid services. All functionality is local.

```bash
PENGU_COST_MODE=FREE_ONLY
```

### FREE_PLUS_CLOUD

Local-first, with optional cloud acceleration.

```bash
PENGU_COST_MODE=FREE_PLUS_CLOUD
GEMINI_API_KEY=your-key-here
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| CPU | 4 cores | 8+ cores |
| GPU | None | 4+ GB VRAM |
| Storage | 5 GB | 20+ GB |
| OS | Windows 10 | Windows 11 |

## Troubleshooting

### "Module not found" errors

Make sure you activated the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Port already in use

Change the port:
```bash
PENGU_API__PORT=8421 python -m pengu
```

### Hardware detection fails

Run directly:
```bash
python -m pengu.hardware.detect
```

### Ollama not found

Install from https://ollama.com and ensure it's on PATH.
