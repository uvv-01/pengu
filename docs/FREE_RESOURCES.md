# Pengu Free Resource Inventory

> Every resource used by Pengu, verified for cost and licensing.

## Resource Table

| Resource | Purpose | Source | License | Local? | Free? | Required? | Cost Risk |
|----------|---------|--------|---------|--------|-------|-----------|-----------|
| Python 3.10+ | Runtime | python.org | PSF | Yes | Yes | Yes | None |
| FastAPI | Web framework | tiangolo/fastapi | MIT | Yes | Yes | Yes | None |
| Uvicorn | ASGI server | encode/uvicorn | BSD | Yes | Yes | Yes | None |
| Pydantic | Validation | samuelcolvin/pydantic | MIT | Yes | Yes | Yes | None |
| psutil | System info | giampaolo/psutil | BSD | Yes | Yes | Yes | None |
| structlog | Logging | hynek/structlog | Apache 2.0 | Yes | Yes | Yes | None |
| PyYAML | Config | yaml/pyyaml | MIT | Yes | Yes | Yes | None |
| httpx | HTTP client | encode/httpx | BSD | Yes | Yes | Yes | None |
| aiosqlite | SQLite async | omnilib/aiosqlite | MIT | Yes | Yes | Yes | None |
| numpy | Math | numpy/numpy | BSD | Yes | Yes | Yes | None |
| sounddevice | Audio | spatialaudio/python-sounddevice | MIT | Yes | Yes | Yes | None |
| Qwen 2.5 | LLM | Alibaba/Qwen | Apache 2.0 | Yes | Yes | No* | None |
| faster-whisper | STT | SYSTRAN/faster-whisper | MIT | Yes | Yes | No* | None |
| Kokoro | TTS | hexgrad/kokoro | Apache 2.0 | Yes | Yes | No* | None |
| openWakeWord | Wake word | dscripka/openWakeWord | Apache 2.0 | Yes | Yes | No* | None |
| Ollama | Model runtime | ollama/ollama | MIT | Yes | Yes | No* | None |
| SQLite | Memory DB | sqlite.org | Public Domain | Yes | Yes | Yes | None |
| Electron | Desktop UI | electron/electron | MIT | Yes | Yes | No** | None |
| React | UI framework | facebook/react | MIT | Yes | Yes | No** | None |
| TypeScript | UI language | microsoft/typescript | Apache 2.0 | Yes | Yes | No** | None |
| Playwright | Browser automation | microsoft/playwright | Apache 2.0 | Yes | Yes | No** | None |
| Git | Version control | git-scm.com | GPL-2.0 | Yes | Yes | No** | None |

\* Required for voice/LLM features, but not for basic startup
\** Required for desktop UI features, not for backend

## Cloud Resources (Optional)

| Resource | Purpose | Free Tier Status | Rate Limits | Required? | Fallback |
|----------|---------|-----------------|-------------|-----------|----------|
| Gemini API | Cloud LLM+Vision | Available (reduced Dec 2025) | Varies by model | No | Local Qwen |
| Groq API | Fast inference | Available | Varies | No | Local Qwen |
| OpenRouter | Model access | Limited | Varies | No | Local Qwen |

**Never required.** All cloud providers have local fallbacks.

## Free Resource Verification Checklist

Before adding any new dependency:

1. [ ] Is it actually free? (check license, not just "free download")
2. [ ] Is it open source? (check repository)
3. [ ] Does the license allow commercial use?
4. [ ] Is the license compatible with Apache 2.0?
5. [ ] Is it actively maintained?
6. [ ] Does it send user data externally?
7. [ ] Is there a local fallback if it breaks?
8. [ ] Does it require payment information?
9. [ ] Can its free tier change/disappear?
10. [ ] Is it actually needed? (don't add just because it exists)
