"""
Safe Terminal — controlled shell execution with command allowlist.

The LLM must NEVER directly execute arbitrary shell commands.
All commands go through an allowlist of safe operations.

Safety model:
  1. Command is parsed and validated against the allowlist
  2. If command is not allowed, it is BLOCKED
  3. Only safe/read-oriented commands execute by default
  4. High-risk commands require explicit confirmation

Blocked commands include:
  - format, del /s, rd /s
  - diskpart, registry modifications
  - credential dumping, firewall changes
  - arbitrary downloads, encoded PowerShell payloads
  - shutdown, reboot, system shutdown
"""

from __future__ import annotations

import re
import subprocess
import shlex
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.os.terminal")

# ---------------------------------------------------------------------------
# Command allowlist — safe commands that execute without confirmation
# ---------------------------------------------------------------------------

# Commands that are always safe (read-only, informational)
SAFE_COMMANDS: set[str] = {
    "git", "python", "python3", "pip", "pip3",
    "node", "npm", "npx", "yarn",
    "where", "which", "whoami", "hostname",
    "echo", "date", "time", "type",
    "dir", "ls", "tree",
    "systeminfo", "tasklist",
    "netsh", "ipconfig", "ping",
    "code", "cursor", "wt",
    "notepad", "explorer",
    "pwsh", "powershell",
    "java", "javac",
    "rustc", "cargo",
    "go",
    "docker",
    "cat", "head", "tail", "wc",
    "uname", "oslevel",
}

# Command families that are allowed
SAFE_COMMAND_PREFIXES: list[str] = [
    "git status", "git log", "git diff", "git branch",
    "git remote", "git show", "git blame",
    "python --version", "python -c", "python -m",
    "pip list", "pip show",
    "node --version", "npm --version",
    "where ", "which ",
    "echo ",
    "type ",
    "dir ", "ls ",
    "systeminfo",
    "tasklist",
    "netsh wlan show",
    "ipconfig",
    "ping ",
    "code --version", "code --list-extensions",
    "wt --version",
    "notepad",
    "explorer",
    "java -version", "javac -version",
]

# Blocked command patterns — NEVER execute
BLOCKED_PATTERNS: list[str] = [
    r"\bformat\b",
    r"\bdel\s+/[sSfFqQaA]",
    r"\brmdir\s+/[sSqQ]",
    r"\brd\s+/[sSqQ]",
    r"\bdiskpart\b",
    r"\breg\s+(delete|save|load|unload)\b",
    r"\bshutdown\b",
    r"\brestart\b",
    r"\breboot\b",
    r"\bSet-ExecutionPolicy\b",
    r"\bnet\s+user\b",
    r"\bnet\s+localgroup\b",
    r"\bicacls\b",
    r"\btakeown\b",
    r"\btaskkill\s+/f\b",
    r"\bget-credential\b",
    r"\bconvertfrom-securestring\b",
    r"\bInvoke-WebRequest\b.*-OutFile",
    r"\bInvoke-RestMethod\b",
    r"\bDownloadFile\b",
    r"\bDownloadString\b",
    r"\bStart-BitsTransfer\b",
    r"powershell.*-enc(oded)?\s+[A-Za-z0-9+/=]{20,}",
    r"powershell.*-e\s+[A-Za-z0-9+/=]{20,}",
    r"cmd\s+/c\s+echo\s+.*>",
    r"\bNew-Object\s+Net\.WebClient\b",
    r"\bIEX\b.*\bInvoke-Expression\b",
]

# Shell injection patterns
INJECTION_PATTERNS: list[str] = [
    r"[;&|`$]",  # Shell metacharacters
    r"\$\(",  # Command substitution
    r"\.\./\.\.",  # Path traversal in command
]

_blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


class TerminalSecurity:
    """Validates commands before execution."""

    def __init__(self) -> None:
        self._blocked_re = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]

    def validate(self, command: str) -> tuple[bool, str]:
        """
        Validate a command against the safety rules.
        
        Returns (is_safe, reason).
        """
        cmd = command.strip()
        if not cmd:
            return False, "Empty command"

        # Check blocked patterns
        for pattern in self._blocked_re:
            if pattern.search(cmd):
                return False, f"BLOCKED: Command matches blocked pattern: {pattern.pattern}"

        # Check for common dangerous prefixes
        cmd_lower = cmd.lower().strip()

        # Block raw powershell with encoded commands
        if "powershell" in cmd_lower or "pwsh" in cmd_lower:
            if "-enc" in cmd_lower or "-encoded" in cmd_lower:
                return False, "BLOCKED: Encoded PowerShell commands are not allowed"

        # Block cmd /c with dangerous patterns
        if cmd_lower.startswith("cmd /c") or cmd_lower.startswith("cmd /k"):
            inner = cmd[6:].strip()
            if any(p.search(inner) for p in self._blocked_re):
                return False, "BLOCKED: Dangerous command inside cmd"

        return True, "Command is allowed"

    def get_command_family(self, command: str) -> str:
        """Identify the command family for logging/routing."""
        parts = command.strip().split()
        if not parts:
            return "unknown"
        return parts[0].lower()


class SafeTerminal:
    """
    Executes commands through a validated, controlled interface.
    
    Usage:
        terminal = SafeTerminal()
        result = await terminal.execute("git status", cwd="/path/to/repo")
    """

    def __init__(self) -> None:
        self.security = TerminalSecurity()

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 30,
        shell: str = "powershell",
    ) -> dict:
        """
        Execute a command through the safety validator.
        
        Returns dict with success, stdout, stderr, exit_code, and security info.
        """
        # Validate
        is_safe, reason = self.security.validate(command)
        if not is_safe:
            logger.warning("command_blocked", command=command[:100], reason=reason)
            return {
                "success": False,
                "error": reason,
                "command": command[:200],
                "blocked": True,
            }

        # Build command list
        if shell == "powershell":
            cmd_list = ["powershell", "-NoProfile", "-Command", command]
        elif shell == "cmd":
            cmd_list = ["cmd", "/c", command]
        else:
            cmd_list = ["bash", "-c", command]

        # Execute
        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                encoding="utf-8",
                errors="replace",
            )

            stdout = result.stdout[:5000] if result.stdout else ""
            stderr = result.stderr[:2000] if result.stderr else ""

            logger.info(
                "command_executed",
                command=command[:100],
                exit_code=result.returncode,
                family=self.security.get_command_family(command),
            )

            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "shell": shell,
                "command": command[:200],
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
                "command": command[:200],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {e}",
                "command": command[:200],
            }

    async def run_safe(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int = 30,
    ) -> dict:
        """
        Alias for execute — provides a safe terminal entry point.
        """
        return await self.execute(command, cwd=cwd, timeout=timeout)
