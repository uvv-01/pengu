"""
Command Pipeline — the central processing loop for Pengu.

Flow:
  USER TEXT
    → INTENT ROUTER (classify)
    → DETERMINISTIC TOOL (if applicable)
    → LOCAL MODEL (if reasoning needed)
    → RESPONSE

Design principle: DETERMINISTIC FIRST.
Only call the LLM when rules/tools cannot handle the request.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pengu.config import TaskCategory, get_config
from pengu.logging import AuditLogger, get_logger, new_task_id
from pengu.models.base import ChatMessage, ChatResponse, ModelProvider
from pengu.router import Intent, IntentRouter, get_router
from pengu.tools.registry import ToolRegistry, ToolResult

logger = get_logger("pengu.pipeline")
audit = AuditLogger()


@dataclass
class PipelineResult:
    """Result of processing a command through the pipeline."""

    text: str
    intent: Intent
    response: str
    provider: str = "deterministic"
    model: str = ""
    tool_used: str = ""
    tool_result: Optional[ToolResult] = None
    latency_ms: float = 0
    error: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "category": self.intent.category.value,
            "confidence": self.intent.confidence,
            "method": self.intent.method,
            "response": self.response,
            "provider": self.provider,
            "model": self.model,
            "tool_used": self.tool_used,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "steps": self.steps,
        }


class CommandPipeline:
    """
    Processes user text through intent classification → tool execution → response.

    Usage:
        pipeline = CommandPipeline(tool_registry, provider)
        result = await pipeline.process("open VS Code")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        provider: Optional[ModelProvider] = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.provider = provider
        self.router = get_router()
        if provider:
            self.router.set_provider(provider)

    def set_provider(self, provider: ModelProvider) -> None:
        """Update the LLM provider."""
        self.provider = provider
        self.router.set_provider(provider)

    async def process(self, text: str, task_id: str = "") -> PipelineResult:
        """
        Process a user command through the full pipeline.

        Returns a PipelineResult with the response and metadata.
        """
        if not task_id:
            task_id = new_task_id()

        start = time.perf_counter()
        steps: list[dict[str, Any]] = []

        # Step 1: Classify intent
        steps.append({"step": "classify", "status": "running"})
        intent = self.router.classify(text)
        steps[-1] = {
            "step": "classify",
            "status": "complete",
            "category": intent.category.value,
            "confidence": intent.confidence,
            "method": intent.method,
        }

        logger.info(
            "pipeline_classified",
            task_id=task_id,
            category=intent.category.value,
            confidence=intent.confidence,
            method=intent.method,
        )

        # Step 2: Route to appropriate handler
        result = await self._route_intent(text, intent, task_id, steps)

        result.latency_ms = (time.perf_counter() - start) * 1000
        result.steps = steps

        audit.log_provider_call(
            provider=result.provider,
            model=result.model,
            success=not bool(result.error),
            duration_ms=result.latency_ms,
            error=result.error,
        )

        return result

    async def _route_intent(
        self,
        text: str,
        intent: Intent,
        task_id: str,
        steps: list[dict[str, Any]],
    ) -> PipelineResult:
        """Route to the appropriate handler based on intent category."""

        category = intent.category

        # SYSTEM_CONTROL → application tools
        if category == TaskCategory.SYSTEM_CONTROL:
            return await self._handle_system_control(text, intent, task_id, steps)

        # FILE_OPERATION → filesystem tools
        elif category == TaskCategory.FILE_OPERATION:
            return await self._handle_file_operation(text, intent, task_id, steps)

        # GIT → git tools
        elif category == TaskCategory.GIT:
            return await self._handle_git(text, intent, task_id, steps)

        # TERMINAL → terminal tools
        elif category == TaskCategory.TERMINAL:
            return await self._handle_terminal(text, intent, task_id, steps)

        # NETWORK → network tools
        elif category == TaskCategory.NETWORK:
            return await self._handle_network(text, intent, task_id, steps)

        # CODING → LLM (needs reasoning)
        elif category == TaskCategory.CODING:
            return await self._handle_with_model(text, intent, task_id, steps, coding=True)

        # VISION → placeholder for Day 3+
        elif category == TaskCategory.VISION:
            return PipelineResult(
                text=text,
                intent=intent,
                response="Vision/screen capture is not yet implemented.",
                provider="deterministic",
                error="NOT IMPLEMENTED",
            )

        # BROWSER → placeholder for Day 3+
        elif category == TaskCategory.BROWSER:
            return PipelineResult(
                text=text,
                intent=intent,
                response="Browser automation is not yet implemented.",
                provider="deterministic",
                error="NOT IMPLEMENTED",
            )

        # WEB_SEARCH → placeholder for Day 3+
        elif category == TaskCategory.WEB_SEARCH:
            return PipelineResult(
                text=text,
                intent=intent,
                response="Web search is not yet implemented.",
                provider="deterministic",
                error="NOT IMPLEMENTED",
            )

        # MEDIA → placeholder for future
        elif category == TaskCategory.MEDIA:
            return PipelineResult(
                text=text,
                intent=intent,
                response="Media control is not yet implemented.",
                provider="deterministic",
                error="NOT IMPLEMENTED",
            )

        # MEMORY → placeholder for future
        elif category == TaskCategory.MEMORY:
            return PipelineResult(
                text=text,
                intent=intent,
                response="Memory system is not yet implemented.",
                provider="deterministic",
                error="NOT IMPLEMENTED",
            )

        # MULTI_STEP_AGENT → LLM for planning
        elif category == TaskCategory.MULTI_STEP_AGENT:
            return await self._handle_with_model(text, intent, task_id, steps, coding=False)

        # CHAT → LLM conversation
        elif category == TaskCategory.CHAT:
            return await self._handle_chat(text, intent, task_id, steps)

        # Fallback
        return PipelineResult(
            text=text,
            intent=intent,
            response=f"I don't know how to handle '{text}' yet.",
            provider="deterministic",
        )

    # -----------------------------------------------------------------------
    # Handler implementations
    # -----------------------------------------------------------------------

    async def _handle_system_control(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle system control (open/close/focus apps)."""
        action = intent.extracted_action
        target = intent.extracted_target

        steps.append({"step": "system_control", "action": action, "target": target, "status": "running"})

        # Special case: VS Code with project/folder
        if intent.extracted_action == "vscode" or "vs code" in text.lower() or "vscode" in text.lower():
            # Check if there's a folder/project to open
            folder_match = None
            import re
            for pattern in [
                r"open\s+(?:my\s+)?(.+?)(?:\s+in\s+(?:vs\s*code|code))",
                r"(?:vs\s*code|code)\s+(?:open|show)\s+(?:my\s+)?(.+)",
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    folder_match = m.group(1).strip().rstrip(".")
                    break

            if folder_match:
                # Try to find the folder
                from pathlib import Path
                candidates = [
                    Path(folder_match),
                    Path.home() / folder_match,
                    Path.cwd() / folder_match,
                ]
                # Also check common project locations
                projects_dir = Path.home() / "projects"
                if projects_dir.exists():
                    for d in projects_dir.iterdir():
                        if d.is_dir() and folder_match.lower() in d.name.lower():
                            candidates.append(d)

                found_path = None
                for c in candidates:
                    if c.exists() and c.is_dir():
                        found_path = c
                        break

                if found_path:
                    tool_result = await self.tool_registry.execute(
                        "application.open",
                        application="code",
                        arguments=str(found_path),
                    )
                    if tool_result.success:
                        steps[-1]["status"] = "complete"
                        return PipelineResult(
                            text=text,
                            intent=intent,
                            response=f"Opened {found_path.name} in VS Code.",
                            provider="deterministic",
                            tool_used="application.open",
                            tool_result=tool_result,
                        )
                    else:
                        steps[-1]["status"] = "error"
                        return PipelineResult(
                            text=text,
                            intent=intent,
                            response=f"Found {found_path.name} but could not open VS Code: {tool_result.error}",
                            provider="deterministic",
                            tool_used="application.open",
                            tool_result=tool_result,
                            error=tool_result.error,
                        )
                else:
                    # Folder not found — open VS Code anyway
                    tool_result = await self.tool_registry.execute(
                        "application.open",
                        application="code",
                    )
                    steps[-1]["status"] = "complete"
                    return PipelineResult(
                        text=text,
                        intent=intent,
                        response=f"Could not find '{folder_match}' folder. Opened VS Code.",
                        provider="deterministic",
                        tool_used="application.open",
                        tool_result=tool_result,
                    )
            else:
                # Just open VS Code
                tool_result = await self.tool_registry.execute(
                    "application.open",
                    application="code",
                )
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response="VS Code is open.",
                    provider="deterministic",
                    tool_used="application.open",
                    tool_result=tool_result,
                )

        # Generic application open/close/focus
        if target:
            tool_result = await self.tool_registry.execute(
                "application.open",
                application=target,
            )
            if tool_result.success:
                app_name = tool_result.output.get("application", target)
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"{app_name} is open.",
                    provider="deterministic",
                    tool_used="application.open",
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Could not open {target}: {tool_result.error}",
                    provider="deterministic",
                    tool_used="application.open",
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text,
            intent=intent,
            response="What application would you like me to open?",
            provider="deterministic",
        )

    async def _handle_file_operation(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle file operations."""
        action = intent.extracted_action
        target = intent.extracted_target

        steps.append({"step": "file_operation", "action": action, "target": target, "status": "running"})

        if action == "read" and target:
            tool_result = await self.tool_registry.execute(
                "filesystem.read_file", path=target
            )
            if tool_result.success:
                content = tool_result.output.get("content", "")
                # Truncate for display
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Contents of {target}:\n\n{content}",
                    provider="deterministic",
                    tool_used="filesystem.read_file",
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Could not read {target}: {tool_result.error}",
                    provider="deterministic",
                    tool_used="filesystem.read_file",
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        elif action == "list":
            path = target or "."
            tool_result = await self.tool_registry.execute(
                "filesystem.list_directory", path=path
            )
            if tool_result.success:
                entries = tool_result.output.get("entries", [])
                lines = []
                for e in entries[:30]:
                    prefix = "[DIR] " if e["type"] == "dir" else "      "
                    size = f"  ({e['size']} bytes)" if e["type"] == "file" and e["size"] else ""
                    lines.append(f"{prefix}{e['name']}{size}")
                listing = "\n".join(lines)
                count = tool_result.output.get("count", 0)
                if count > 30:
                    listing += f"\n... and {count - 30} more entries"
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Directory {path}:\n{listing}",
                    provider="deterministic",
                    tool_used="filesystem.list_directory",
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Could not list directory: {tool_result.error}",
                    provider="deterministic",
                    tool_used="filesystem.list_directory",
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        elif action == "search" and target:
            tool_result = await self.tool_registry.execute(
                "filesystem.grep", query=target
            )
            if tool_result.success:
                results = tool_result.output.get("results", [])
                if results:
                    lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results[:20]]
                    steps[-1]["status"] = "complete"
                    return PipelineResult(
                        text=text,
                        intent=intent,
                        response=f"Found {len(results)} matches:\n" + "\n".join(lines),
                        provider="deterministic",
                        tool_used="filesystem.grep",
                        tool_result=tool_result,
                    )
                else:
                    steps[-1]["status"] = "complete"
                    return PipelineResult(
                        text=text,
                        intent=intent,
                        response=f"No matches found for '{target}'.",
                        provider="deterministic",
                        tool_used="filesystem.grep",
                        tool_result=tool_result,
                    )

        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text,
            intent=intent,
            response="I understand you want to work with files, but I need more details. What file operation?",
            provider="deterministic",
        )

    async def _handle_git(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle git operations."""
        action = intent.extracted_action

        steps.append({"step": "git", "action": action, "status": "running"})

        # Map git actions to tool calls
        git_handler_map = {
            "git.status": ("git.status", {}),
            "git.log": ("git.log", {}),
            "git.diff": ("git.diff", {}),
        }

        if action in git_handler_map:
            tool_name, kwargs = git_handler_map[action]
            tool_result = await self.tool_registry.execute(tool_name, **kwargs)
            if tool_result.success:
                output = tool_result.output
                stdout = output.get("stdout", "").strip()
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=stdout or f"Git {action.split('.')[-1]} completed.",
                    provider="deterministic",
                    tool_used=tool_name,
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Git error: {tool_result.error}",
                    provider="deterministic",
                    tool_used=tool_name,
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        # For other git commands, execute via git tool
        import re
        git_cmd_match = re.search(r"\bgit\s+(\S+)(?:\s+(.+))?", text, re.IGNORECASE)
        if git_cmd_match:
            subcmd = git_cmd_match.group(1)
            rest = git_cmd_match.group(2) or ""
            args = [subcmd] + (rest.split() if rest else [])
            tool_result = await self.tool_registry.execute("git.execute", args=args)
            if tool_result.success:
                stdout = tool_result.output.get("stdout", "").strip()
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=stdout or f"git {subcmd} completed.",
                    provider="deterministic",
                    tool_used="git.execute",
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Git error: {tool_result.error}",
                    provider="deterministic",
                    tool_used="git.execute",
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text,
            intent=intent,
            response="What git operation would you like? (status, diff, log, branch, etc.)",
            provider="deterministic",
        )

    async def _handle_terminal(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle terminal commands."""
        steps.append({"step": "terminal", "status": "running"})

        # Extract the command from text
        import re
        command_match = re.search(
            r"(?:run|execute|type)\s+(?:the\s+)?(?:command\s+)?[\"']?(.+?)[\"']?\s*$",
            text,
            re.IGNORECASE,
        )
        if command_match:
            command = command_match.group(1).strip()
            tool_result = await self.tool_registry.execute(
                "terminal.execute", command=command
            )
            if tool_result.success:
                stdout = tool_result.output.get("stdout", "").strip()
                stderr = tool_result.output.get("stderr", "").strip()
                output = stdout
                if stderr:
                    output += f"\n[stderr]: {stderr}"
                steps[-1]["status"] = "complete"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=output or "Command executed successfully.",
                    provider="deterministic",
                    tool_used="terminal.execute",
                    tool_result=tool_result,
                )
            else:
                steps[-1]["status"] = "error"
                return PipelineResult(
                    text=text,
                    intent=intent,
                    response=f"Command failed: {tool_result.error}",
                    provider="deterministic",
                    tool_used="terminal.execute",
                    tool_result=tool_result,
                    error=tool_result.error,
                )

        steps[-1]["status"] = "error"
        return PipelineResult(
            text=text,
            intent=intent,
            response="What command would you like me to run?",
            provider="deterministic",
        )

    async def _handle_network(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle network operations."""
        steps.append({"step": "network", "status": "running"})

        text_lower = text.lower()
        if "list" in text_lower or "available" in text_lower or "show" in text_lower:
            tool_result = await self.tool_registry.execute("network.list_wifi")
        elif "status" in text_lower or "current" in text_lower or "connected" in text_lower:
            tool_result = await self.tool_registry.execute("network.wifi_status")
        else:
            tool_result = await self.tool_registry.execute("network.wifi_status")

        if tool_result.success:
            raw = tool_result.output.get("raw", "")
            steps[-1]["status"] = "complete"
            return PipelineResult(
                text=text,
                intent=intent,
                response=raw[:2000] if raw else "Network info retrieved.",
                provider="deterministic",
                tool_used="network.wifi_status",
                tool_result=tool_result,
            )
        else:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text,
                intent=intent,
                response=f"Network error: {tool_result.error}",
                provider="deterministic",
                tool_used="network.wifi_status",
                tool_result=tool_result,
                error=tool_result.error,
            )

    async def _handle_chat(
        self, text: str, intent: Intent, task_id: str, steps: list[dict[str, Any]]
    ) -> PipelineResult:
        """Handle conversational chat — uses the local LLM."""
        steps.append({"step": "chat", "status": "running"})

        if not self.provider or not self.provider.is_available():
            steps[-1]["status"] = "no_model"
            return PipelineResult(
                text=text,
                intent=intent,
                response=self._get_offline_chat_response(text),
                provider="deterministic",
                model="",
                error="No local model available",
            )

        # Build messages
        system_prompt = (
            "You are Pengu, a local-first desktop assistant. "
            "Be concise and helpful. Respond in 1-2 sentences unless asked for detail. "
            "You run entirely on the user's local machine."
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=text),
        ]

        steps[-1]["status"] = "calling_model"
        response = await self.provider.chat(messages, temperature=0.7, max_tokens=512)

        if response.error:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text,
                intent=intent,
                response=self._get_offline_chat_response(text),
                provider="lmstudio",
                model=response.model,
                error=response.error,
            )

        steps[-1]["status"] = "complete"
        return PipelineResult(
            text=text,
            intent=intent,
            response=response.content,
            provider=response.provider,
            model=response.model,
        )

    async def _handle_with_model(
        self,
        text: str,
        intent: Intent,
        task_id: str,
        steps: list[dict[str, Any]],
        coding: bool = False,
    ) -> PipelineResult:
        """Handle requests that need LLM reasoning."""
        steps.append({"step": "model_reasoning", "status": "running"})

        if not self.provider or not self.provider.is_available():
            steps[-1]["status"] = "no_model"
            context = "coding" if coding else "general"
            return PipelineResult(
                text=text,
                intent=intent,
                response=f"No local model available for {context} tasks. Please load a model in LM Studio.",
                provider="deterministic",
                error="No local model available",
            )

        system_prompt = (
            "You are Pengu, a local-first desktop assistant running on the user's machine. "
            "Be concise and practical. "
        )
        if coding:
            system_prompt += (
                "You help with coding tasks. Provide clear, working code. "
                "Explain briefly what the code does."
            )
        else:
            system_prompt += (
                "Help the user accomplish their task step by step. "
                "If the task requires multiple steps, outline the plan first."
            )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=text),
        ]

        steps[-1]["status"] = "calling_model"
        response = await self.provider.chat(messages, temperature=0.7, max_tokens=1024)

        if response.error:
            steps[-1]["status"] = "error"
            return PipelineResult(
                text=text,
                intent=intent,
                response=f"Model error: {response.error}",
                provider="lmstudio",
                model=response.model,
                error=response.error,
            )

        steps[-1]["status"] = "complete"
        return PipelineResult(
            text=text,
            intent=intent,
            response=response.content,
            provider=response.provider,
            model=response.model,
        )

    def _get_offline_chat_response(self, text: str) -> str:
        """Generate a response when no model is available."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["hello", "hi", "hey"]):
            return "Hello! I'm Pengu. No local model is loaded right now, but I can still help with file operations, git, and system commands."
        elif "what can you do" in text_lower or "help" in text_lower:
            return (
                "I can help with: opening applications, file operations, git commands, "
                "terminal commands, and network info. For AI-powered chat, "
                "load a model in LM Studio."
            )
        elif "who are you" in text_lower:
            return "I'm Pengu, a local-first desktop assistant. I'm running on your machine with no cloud dependency."
        elif "2 + 2" in text_lower or "what is" in text_lower:
            return "I can compute that: 2 + 2 = 4. For more complex reasoning, load a model in LM Studio."
        else:
            return (
                f"I received your message: '{text}'. "
                "No local model is currently loaded for AI responses. "
                "I can still help with file operations, git, and system commands. "
                "Load a model in LM Studio for full AI capabilities."
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pipeline: Optional[CommandPipeline] = None


def get_pipeline(
    tool_registry: ToolRegistry, provider: Optional[ModelProvider] = None
) -> CommandPipeline:
    """Get or create the global command pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = CommandPipeline(tool_registry, provider)
    return _pipeline
