# Pengu Model Report

> Documented: August 24, 2026
> Rule: Never claim a model is free unless verified. Verify current official sources.

## Summary Table

| Role | Model | Provider/Source | Runtime | Local? | Cost | License | HW Tier | Selected? |
|------|-------|----------------|---------|--------|------|---------|---------|-----------|
| General LLM | Qwen 2.5 7B | Ollama | Ollama | Yes | Free | Apache 2.0 | MEDIUM | Yes (auto) |
| Coding LLM | Qwen 2.5 Coder 7B | Ollama | Ollama | Yes | Free | Apache 2.0 | MEDIUM | Yes (auto) |
| Vision LLM | Qwen 2-VL 2B | Ollama | Ollama | Yes | Free | Apache 2.0 | MEDIUM+ | Yes (auto) |
| STT | faster-whisper distil-small.en | Hugging Face | faster-whisper | Yes | Free | MIT | LOW+ | Yes |
| TTS | Kokoro 82M | Kokoro | kokoro | Yes | Free | Apache 2.0 | LOW+ | Yes |
| Wake Word | openWakeWord | openWakeWord | openWakeWord | Yes | Free | Apache 2.0 | LOW+ | Yes |
| Cloud (optional) | Gemini 2.0 Flash | Google AI | API | No | Free tier* | Google TOS | ANY | Optional |

\* Free tier has rate limits that change. Not required.

---

## Selected Model Details

### General LLM: Qwen 2.5

**MODEL:** Qwen 2.5 (various sizes)
**SOURCE:** Alibaba Cloud / Qwen Team
**OFFICIAL URL:** https://ollama.com/library/qwen2.5
**LICENSE:** Apache 2.0
**LOCAL/CLOUD:** Local (via Ollama)
**COST:** Free
**MINIMUM PRACTICAL HARDWARE:**
  - 1.5B variant: 4 GB RAM
  - 7B variant: 16 GB RAM
**RECOMMENDED HARDWARE:**
  - 7B: 16 GB RAM, optional GPU
  - 14B: 32 GB RAM, GPU helpful
**RUNTIME:** Ollama
**WHY SELECTED:**
  - Strong instruction following
  - Tool/function calling support
  - Structured output support
  - Active development
  - Apache 2.0 license
  - Available in many quantized sizes via Ollama
**ALTERNATIVES:**
  - Phi-3/4 (Microsoft) — strong but smaller context
  - Gemma 2 (Google) — good but restrictive license
  - Mistral — good but newer to local ecosystem

### Coding LLM: Qwen 2.5 Coder

**MODEL:** Qwen 2.5 Coder
**SOURCE:** Alibaba Cloud / Qwen Team
**OFFICIAL URL:** https://ollama.com/library/qwen2.5-coder
**LICENSE:** Apache 2.0
**LOCAL/CLOUD:** Local (via Ollama)
**COST:** Free
**MINIMUM PRACTICAL HARDWARE:**
  - 1.5B: 4 GB RAM
  - 7B: 16 GB RAM
**RECOMMENDED HARDWARE:**
  - 14B: 32 GB RAM
**RUNTIME:** Ollama
**WHY SELECTED:**
  - Best open-source coding model for its size class
  - Strong code generation and understanding
  - Apache 2.0 license
  - Ollama ecosystem support
**ALTERNATIVES:**
  - DeepSeek Coder — strong but license has restrictions
  - CodeGemma — limited sizes

### STT: faster-whisper (distil-small.en)

**MODEL:** Distil-Whisper (distil-small.en)
**SOURCE:** Hugging Face (Systran)
**OFFICIAL URL:** https://huggingface.co/Systran/faster-whisper-small-en
**LICENSE:** MIT
**LOCAL/CLOUD:** Local
**COST:** Free
**MINIMUM PRACTICAL HARDWARE:** 2 GB RAM
**RECOMMENDED HARDWARE:** 4 GB RAM
**RUNTIME:** faster-whisper (CTranslate2)
**WHY SELECTED:**
  - Fast CPU inference
  - Low memory footprint
  - MIT license
  - Good accuracy for short commands
  - Streaming capable
**ALTERNATIVES:**
  - Moonshine — even lighter but less accurate
  - Parakeet — NVIDIA-optimized, heavier
  - Whisper.cpp — C++ port, good alternative

### TTS: Kokoro

**MODEL:** Kokoro 82M
**SOURCE:** Kokoro TTS
**OFFICIAL URL:** https://github.com/hexgrad/kokoro
**LICENSE:** Apache 2.0
**LOCAL/CLOUD:** Local
**COST:** Free
**MINIMUM PRACTICAL HARDWARE:** 2 GB RAM, runs on CPU
**RECOMMENDED HARDWARE:** 4 GB RAM
**RUNTIME:** kokoro (Python)
**WHY SELECTED:**
  - 82M parameters — very lightweight
  - Natural sounding voice
  - 54 voices, 8 languages
  - Apache 2.0 license
  - Fast CPU inference
  - Active community
**ALTERNATIVES:**
  - Piper — lighter but less natural
  - Coqui TTS — heavier, more complex setup

### Wake Word: openWakeWord

**MODEL:** openWakeWord
**SOURCE:** David Scripka
**OFFICIAL URL:** https://github.com/dscripka/openWakeWord
**LICENSE:** Apache 2.0
**LOCAL/CLOUD:** Local
**COST:** Free
**MINIMUM PRACTICAL HARDWARE:** 1 GB RAM
**RECOMMENDED HARDWARE:** 2 GB RAM
**RUNTIME:** openWakeWord (ONNX)
**WHY SELECTED:**
  - Supports custom wake words ("Hello Pengu")
  - Apache 2.0 license
  - Very low resource usage
  - Active development
  - Home Assistant ecosystem support
**ALTERNATIVES:**
  - Porcupine — has commercial restrictions on custom words
  - Snowboy — discontinued

### Cloud (Optional): Google Gemini

**MODEL:** Gemini 2.0 Flash
**SOURCE:** Google AI
**OFFICIAL URL:** https://ai.google.dev/
**LICENSE:** Google TOS (free tier)
**LOCAL/CLOUD:** Cloud
**COST:** Free tier exists (rate limits vary, reduced in Dec 2025)
**MINIMUM PRACTICAL HARDWARE:** N/A
**RECOMMENDED HARDWARE:** N/A
**RUNTIME:** Google AI API
**WHY SELECTED:**
  - Free tier available
  - Vision capable
  - Tool calling support
  - Large context window
**ALTERNATIVES:**
  - Groq — free tier for some models
  - OpenRouter — aggregated free models

---

## Hardware Tier → Model Mapping

### LOW (8-15 GB RAM, no GPU)

| Role | Model |
|------|-------|
| LLM | qwen2.5:1.5b |
| Coding | qwen2.5-coder:1.5b |
| Vision | (not available) |
| STT | distil-small.en |
| TTS | kokoro |

### MEDIUM (16-31 GB RAM, no/small GPU)

| Role | Model |
|------|-------|
| LLM | qwen2.5:7b |
| Coding | qwen2.5-coder:7b |
| Vision | qwen2-vl:2b |
| STT | distil-small.en |
| TTS | kokoro |

### HIGH (32+ GB RAM, 4+ GB VRAM)

| Role | Model |
|------|-------|
| LLM | qwen2.5:14b |
| Coding | qwen2.5-coder:14b |
| Vision | qwen2-vl:7b |
| STT | distil-small.en |
| TTS | kokoro |

---

## Notes

- All local models require Ollama to be installed separately
- Model downloads happen on first use, not during installation
- Quantization: Ollama uses Q4_K_M by default (good balance of quality/speed)
- Vision models require GPU for practical performance
- STT and TTS run well on CPU
- Cloud models are optional accelerators — never required
