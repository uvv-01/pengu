# Pengu Development Guide

## Getting Started

```bash
git clone <repo-url>
cd pengu
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Project Conventions

- **Type hints** on all function signatures
- **Docstrings** on all public functions and classes
- **Async/await** for I/O-bound operations
- **Pydantic** for all configuration and data validation
- **structlog** for all logging (not print statements)
- **pytest** + **pytest-asyncio** for tests
- **ruff** for linting
- **mypy** for type checking

## Code Style

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/pengu/
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=pengu --cov-report=html

# Specific module
pytest tests/test_config.py -v

# Async tests
pytest tests/test_state.py -v -k async
```

## Architecture Principles

1. **Deterministic First** — Don't waste LLM calls on hard-coded operations
2. **Tool Registry** — Every OS capability is a typed, permissioned tool
3. **Provider Abstraction** — Local-first, cloud-optional
4. **Hardware Awareness** — Detect hardware, select appropriate models
5. **Fail Silently** — Cloud failures never crash the core system
6. **Real Functionality** — No mocks, no fakes, no demos

## Adding a New Tool

```python
from pengu.config import PermissionLevel
from pengu.tools.registry import Tool, ToolRegistry

async def my_handler(filename: str = "") -> str:
    # Your implementation here
    return f"Processed {filename}"

tool = Tool(
    name="filesystem.my_operation",
    description="Does something with a file",
    category="filesystem",
    permission_level=PermissionLevel.LOW_RISK,
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "File path"}
        },
        "required": ["filename"],
    },
    handler=my_handler,
)

registry.register(tool)
```

## Adding a Provider

```python
from pengu.models.base import ModelProvider, ProviderType, ChatMessage, ChatResponse

class MyProvider(ModelProvider):
    def __init__(self):
        super().__init__("my_provider", ProviderType.LOCAL_OLLAMA)

    async def health_check(self) -> bool:
        # Check if the provider is available
        return True

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        # Implement the chat interface
        return ChatResponse(content="response", provider=self.name)
```

## Day-by-Day Plan

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Foundation | Backend starts successfully |
| 2 | Models | Text → router → response |
| 3 | Tools | Basic Windows control |
| 4 | Git/VS Code | Real dev tasks |
| 5 | Wake Word | "Hello Pengu" works |
| 6 | Voice | Full voice conversation |
| 7 | Desktop UI | Electron overlay |
| 8 | Vision | Screen inspection |
| 9 | Memory | Persistent assistant |
| 10 | Integration | End-to-end demo |
