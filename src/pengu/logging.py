"""
Structured logging for Pengu.

Uses structlog for structured, contextual logs.
Every log entry includes:
  - timestamp
  - level
  - module
  - event
  - task_id (when available)
  - duration_ms (when available)
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog

# ---------------------------------------------------------------------------
# Context vars for request-scoped data
# ---------------------------------------------------------------------------

_current_task_id: ContextVar[str] = ContextVar("task_id", default="")
_current_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_task_id() -> str:
    return _current_task_id.get()


def set_task_id(task_id: str) -> None:
    _current_task_id.set(task_id)


def new_task_id() -> str:
    task_id = f"task_{uuid.uuid4().hex[:12]}"
    _current_task_id.set(task_id)
    return task_id


def get_request_id() -> str:
    return _current_request_id.get()


def set_request_id(request_id: str) -> None:
    _current_request_id.set(request_id)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class StepTimer:
    """Timer for measuring step durations."""

    def __init__(self, name: str, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.name = name
        self.logger = logger
        self.start_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "StepTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000
        if self.logger:
            self.logger.info(
                "step_complete",
                step=self.name,
                duration_ms=round(self.duration_ms, 2),
            )


# ---------------------------------------------------------------------------
# Audit logger — for security-sensitive operations
# ---------------------------------------------------------------------------

class AuditLogger:
    """Logs security-relevant actions for audit trail."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("pengu.audit")

    def log_tool_execution(
        self,
        tool_name: str,
        params: dict[str, Any],
        permission_level: int,
        granted: bool,
        result: str = "",
    ) -> None:
        self._logger.info(
            "tool_execution",
            tool=tool_name,
            permission_level=permission_level,
            granted=granted,
            params_keys=list(params.keys()),
            result=result[:200] if result else "",
            task_id=get_task_id(),
        )

    def log_state_transition(self, from_state: str, to_state: str) -> None:
        self._logger.info(
            "state_transition",
            from_state=from_state,
            to_state=to_state,
            task_id=get_task_id(),
        )

    def log_provider_call(
        self,
        provider: str,
        model: str,
        success: bool,
        duration_ms: float,
        error: str = "",
    ) -> None:
        self._logger.info(
            "provider_call",
            provider=provider,
            model=model,
            success=success,
            duration_ms=round(duration_ms, 2),
            error=error[:200] if error else "",
            task_id=get_task_id(),
        )


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structlog + stdlib logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: True for JSON logs (production), False for colored (dev)
        log_file: Optional file path for log output
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Stdlib handler
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quiet noisy libraries
    for name in ("httpx", "httpcore", "uvicorn.access", "websockets"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_logger(name: str = "pengu") -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger."""
    return structlog.stdlib.get_logger(name)
