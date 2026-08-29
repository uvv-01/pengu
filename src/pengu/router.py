"""
Intent Router — classifies user text into task categories.

Uses a deterministic rule-based classifier first (zero latency, zero cost).
Falls back to the local LLM only when rules are insufficient.

Categories:
  CHAT, SYSTEM_CONTROL, FILE_OPERATION, CODING, TERMINAL, BROWSER,
  WEB_SEARCH, VISION, GIT, NETWORK, MEDIA, MEMORY, MULTI_STEP_AGENT

Design rule: DETERMINISTIC FIRST.
Do NOT call an LLM when a regex/keyword can classify the intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pengu.config import TaskCategory
from pengu.logging import get_logger

logger = get_logger("pengu.router")


@dataclass
class Intent:
    """Classified intent from user text."""

    category: TaskCategory
    confidence: float  # 0.0 - 1.0
    method: str  # "rule", "model", "fallback"
    extracted_action: str = ""  # e.g. "open", "read", "write"
    extracted_target: str = ""  # e.g. "VS Code", "README.md"
    raw_text: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Rule definitions — ordered by priority (first match wins)
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A single classification rule."""

    category: TaskCategory
    patterns: list[re.Pattern]
    action: str = ""
    confidence: float = 0.95
    description: str = ""


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# VS Code specific (highest priority — very specific)
VSCODE_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.SYSTEM_CONTROL,
        patterns=_compile([
            r"\bvs\s*code\b",
            r"\bvisual\s+studio\s+code\b",
            r"\bcode\s+\.?\s*open\b",
            r"\bopen\s+.*\bin\s+vs\s*code\b",
            r"\bopen\s+.*\bin\s+code\b",
            r"\bopen\s+(my\s+)?(.+?)\s+in\s+(vs\s*code|code)\b",
        ]),
        action="vscode",
        confidence=0.95,
        description="VS Code operations",
    ),
]

# System info rules
SYSTEM_INFO_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.SYSTEM_CONTROL,
        patterns=_compile([
            r"\bwhat('s|\s+is)\s+(my\s+)?(cpu|processor|ram|memory|gpu|disk|storage|system)\b",
            r"\bwhat\s+\w*\s+(do|does)\s+(i|this\s+computer)\s+have\b",
            r"\bhow\s+(much|many)\s+(ram|memory|storage|disk|cpu|cores)\b",
            r"\bsystem\s+(info|information|details|specs)\b",
            r"\bwhat\s+(tier|class)\s+(is\s+)?(this|my)\b",
            r"\bshow\s+(me\s+)?(system|hardware|specs)\b",
            r"\bcheck\s+(my\s+)?(system|hardware|specs)\b",
            r"\bwhat\s+computer\s+is\s+this\b",
            r"\brun\s+system\s+info\b",
        ]),
        action="system_info",
        confidence=0.95,
        description="System information queries",
    ),
]

# Process management rules
PROCESS_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.SYSTEM_CONTROL,
        patterns=_compile([
            r"\b(list|show|what)\s+(is\s+)?(running|processes?|apps?)\b",
            r"\bwhat('s|\s+is)\s+(using|eating|consuming)\s+(cpu|memory|ram|disk)\b",
            r"\btop\s+(cpu|memory|ram)\b",
            r"\bprocess\s+(list|info|details)\b",
            r"\b(is|are)\s+.+?\s+(running|open|alive)\b",
        ]),
        action="process",
        confidence=0.9,
        description="Process inspection and management",
    ),
]

# Git patterns (high priority — very specific)
GIT_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.GIT,
        patterns=_compile([
            r"\bgit\s+status\b",
            r"\bgit\s+diff\b",
            r"\bgit\s+log\b",
            r"\bgit\s+branch\b",
            r"\bgit\s+checkout\b",
            r"\bgit\s+add\b",
            r"\bgit\s+commit\b",
            r"\bgit\s+push\b",
            r"\bgit\s+pull\b",
            r"\bgit\s+merge\b",
            r"\bgit\s+stash\b",
            r"\bgit\s+clone\b",
            r"\bcheck\s+(git|repo|repository)\b",
        ]),
        action="git",
        confidence=0.95,
        description="Git operations",
    ),
]

# Terminal patterns (before SYSTEM_CONTROL so "run the command" matches here first)
TERMINAL_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.TERMINAL,
        patterns=_compile([
            r"\b(?:run|execute|type)\b.*?\b(command|script)\b",
            r"\bterminal\b",
            r"\bpowershell\b",
            r"\bcmd\b",
            r"\bcommand\s+prompt\b",
            r"\bshell\b",
        ]),
        action="terminal",
        confidence=0.9,
        description="Terminal/shell operations",
    ),
]

# Application control patterns
# Desktop interaction rules (click, type, scroll, go-to)
DESKTOP_INTERACTION_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.SYSTEM_CONTROL,
        patterns=_compile([
            r"\bclick\s+(.+)",
            r"\bpress\s+(.+)",
            r"\bscroll\s+(down|up|left|right)",
            r"\btype\s+(.+)",
            r"\benter\s+(.+)",
            r"\bgo\s+to\s+(.+)",
            r"\bnavigate\s+to\s+(.+)",
            r"\bclick\s+the\s+(.+)",
            r"\bclick\s+first\s+result",
            r"\bclick\s+first\s+(link|button|result)",
        ]),
        action="desktop_interaction",
        confidence=0.85,
        description="Desktop UI interaction: click, type, scroll",
    ),
]

APPLICATION_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.SYSTEM_CONTROL,
        patterns=_compile([
            r"\bopen\s+(.+)",
            r"\blaunch\s+(.+)",
            r"\bstart\s+(.+)",
            r"\bclose\s+(.+)",
            r"\bquit\s+(.+)",
            r"\bkill\s+(.+)",
            r"\bfocus\s+(.+)",
            r"\bswitch\s+to\s+(.+)",
        ]),
        action="application",
        confidence=0.9,
        description="Open/close/focus applications",
    ),
]

# Filesystem patterns
FILE_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.FILE_OPERATION,
        patterns=_compile([
            r"\bread\s+(this|the|that|my)?\s*(file|readme|doc|config)",
            r"\bshow\s+(this|the|that|my)?\s*(file|readme|doc|config)",
            r"\blist\s+(files?|directories?|folders?|contents?)",
            r"\bfind\b.*?\b(files?|folders?|directories?)\b",
            r"\bsearch\s+(for\s+)?(.+)\s+(in|inside|through)",
            r"\bcreate\b.*?\b(file|directory|folder)\b",
            r"\bdelete\s+(the\s+)?(file|directory|folder)",
            r"\bwhat('s|\s+is)\s+in\s+(this|the|that)\s+directory",
            r"\bwhere\s+is\s+(.+)",
            r"\bopen\s+(the\s+)?(file|folder|directory)",
        ]),
        action="filesystem",
        confidence=0.9,
        description="File read/write/list/search",
    ),
]

# Coding patterns (before FILE to catch "write a program" before "write a file")
CODING_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.CODING,
        patterns=_compile([
            r"\bwrite\b.*?\b(program|function|class|script|code|solution)\b",
            r"\bcreate\b.*?\b(program|function|class|script|code|solution)\b",
            r"\bimplement\s+(a\s+)?",
            r"\bfix\s+(the\s+)?(bug|error|test|code|issue)",
            r"\bdebug\s+(the\s+)?",
            r"\brefactor\s+(the\s+)?",
            r"\bcode\s+(a\s+|the\s+|this\s+)",
            r"\bprogram\s+(a\s+|the\s+|this\s+)",
            r"\bsolve\s+(this|the|that)",
            r"\bwrite\s+.*\b(in|using)\s+(python|java|javascript|typescript|rust|go|c\+\+|html|css)",
            r"\bexplain\s+(this|the|that|this\s+code)",
            r"\breview\s+(this|the|that)\s+(code|function|file)",
        ]),
        action="coding",
        confidence=0.9,
        description="Code writing, fixing, reviewing",
    ),
]

# Web search patterns
WEB_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.WEB_SEARCH,
        patterns=_compile([
            r"\bsearch\s+(for\s+)?(.+)",
            r"\bgoogle\s+(for\s+)?(.+)",
            r"\blook\s+up\s+(.+)",
            r"\bwhat('s|\s+is)\s+the\s+(latest|current|newest)",
            r"\bfind\s+(me\s+)?(information|info|news|docs|documentation)",
            r"\bbrowse\s+(to|for|the)",
            r"\bcheck\s+(the\s+)?(website|web|online|internet)",
        ]),
        action="web_search",
        confidence=0.85,
        description="Web search and browsing",
    ),
]

# Vision patterns
VISION_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.VISION,
        patterns=_compile([
            r"\blook\s+at\s+(my\s+)?(screen|display|monitor)",
            r"\bwhat('s|\s+is)\s+(on\s+)?(my\s+)?(screen|display|monitor)",
            r"\bscreenshot\b",
            r"\bcapture\s+(my\s+)?(screen|display)",
            r"\bwhat\s+do\s+you\s+see",
            r"\binspect\s+(my\s+)?(screen|display)",
        ]),
        action="vision",
        confidence=0.9,
        description="Screen capture and vision analysis",
    ),
]

# Network patterns
NETWORK_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.NETWORK,
        patterns=_compile([
            r"\bwi-?fi\b",
            r"\bwifi\b",
            r"\bwi\s+fi\b",
            r"\bconnect\s+to\s+(.+)\s+(wifi|network|wi-?fi)",
            r"\bdisconnect\s+(from\s+)?(wifi|network)",
            r"\blist\s+(available\s+)?(wifi|networks?|wi-?fi)",
            r"\bwhat('s|\s+is)\s+(my\s+)?(wifi|network|ip)\s*(address)?",
            r"\binternet\b",
            r"\bnetwork\s+(status|connection|info)",
        ]),
        action="network",
        confidence=0.9,
        description="Wi-Fi and network operations",
    ),
]

# Chat patterns (lowest priority — catch-all for conversation)
CHAT_RULES: list[Rule] = [
    Rule(
        category=TaskCategory.CHAT,
        patterns=_compile([
            r"^(hi|hello|hey|howdy|sup|yo|greetings)\b",
            r"\bhow\s+(are\s+you|do\s+you\s+do|is\s+it\s+going)",
            r"\bthank(s|\s+you)\b",
            r"\bwhat\s+(is|are)\s+you\b",
            r"\btell\s+me\s+(about|something)",
            r"\bwho\s+(are\s+you|made\s+you|created\s+you)",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bhelp\s+me\b",
            r"\bwhat\s+time\b",
            r"\bwhat('s|\s+is)\s+the\s+date\b",
        ]),
        action="chat",
        confidence=0.8,
        description="General conversation",
    ),
]

# All rules in priority order
ALL_RULES: list[Rule] = (
    SYSTEM_INFO_RULES
    + PROCESS_RULES
    + VSCODE_RULES
    + GIT_RULES
    + TERMINAL_RULES
    + CODING_RULES
    + DESKTOP_INTERACTION_RULES
    + APPLICATION_RULES
    + FILE_RULES
    + WEB_RULES
    + VISION_RULES
    + NETWORK_RULES
    + CHAT_RULES
)


# ---------------------------------------------------------------------------
# Action/target extraction
# ---------------------------------------------------------------------------

# Common application names for extraction
KNOWN_APPLICATIONS: dict[str, str] = {
    "vs code": "code",
    "visual studio code": "code",
    "code": "code",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "terminal": "wt",
    "windows terminal": "wt",
    "powershell": "pwsh",
    "cmd": "cmd",
    "command prompt": "cmd",
    "file explorer": "explorer",
    "explorer": "explorer",
    "notepad": "notepad",
    "notepad++": "notepad++",
    "git bash": "git-bash",
    "intellij": "idea64",
    "intellij idea": "idea64",
    "pycharm": "pycharm64",
    "webstorm": "webstorm64",
    "cursor": "cursor",
}


def _extract_application(text: str) -> tuple[str, str]:
    """Extract application action and target from text."""
    text_lower = text.lower().strip()

    action = "open"
    for verb in ["close", "quit", "kill", "stop"]:
        if verb in text_lower.split():
            action = verb
            break
    for verb in ["launch", "start", "run"]:
        if verb in text_lower.split():
            action = "open"
            break
    for verb in ["focus", "switch"]:
        if verb in text_lower.split():
            action = "focus"
            break

    target = ""
    for verb_pattern in [
        r"open\s+(.+)",
        r"launch\s+(.+)",
        r"start\s+(.+)",
        r"run\s+(.+)",
        r"close\s+(.+)",
        r"quit\s+(.+)",
        r"kill\s+(.+)",
        r"focus\s+(.+)",
        r"switch\s+to\s+(.+)",
    ]:
        m = re.search(verb_pattern, text, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
            break

    target = re.sub(r"\s+in\s+(vs\s*code|visual\s+studio\s+code|code)\s*$", "", target, flags=re.IGNORECASE)
    target = target.strip(" .,!?")

    return action, target


def _extract_file_target(text: str) -> tuple[str, str]:
    """Extract file operation action and target from text."""
    text_lower = text.lower().strip()

    action = "read"
    if any(w in text_lower for w in ["create", "make", "new", "write"]):
        action = "create"
    elif any(w in text_lower for w in ["delete", "remove", "trash"]):
        action = "delete"
    elif any(w in text_lower for w in ["list", "show", "what"]):
        action = "list"
    elif any(w in text_lower for w in ["find", "search", "look"]):
        action = "search"
    elif any(w in text_lower for w in ["read", "open", "show", "cat"]):
        action = "read"

    target = ""
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    if quoted:
        target = quoted[0]
    else:
        file_match = re.search(
            r"(?:file|folder|directory|readme|config|doc)\s+(?:named?|called?|titled?)\s+(.+?)(?:\s+(?:in|inside|through|from|to)\s+|$)",
            text,
            re.IGNORECASE,
        )
        if file_match:
            target = file_match.group(1).strip()
        else:
            path_match = re.search(r"([\w./\\-]+\.\w+)", text)
            if path_match:
                target = path_match.group(1)

    return action, target


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Classifies user text into task categories.

    Uses deterministic rules first (zero cost, zero latency).
    Falls back to local LLM for ambiguous cases.
    """

    def __init__(self) -> None:
        self._rules = ALL_RULES
        self._provider = None  # Set later if LLM fallback is available

    def set_provider(self, provider) -> None:
        """Set the LLM provider for fallback classification."""
        self._provider = provider

    def classify(self, text: str) -> Intent:
        """
        Classify user text into an intent.

        Priority:
          1. Deterministic rules (fast, free)
          2. LLM fallback (if rules fail or confidence is low)
        """
        if not text or not text.strip():
            return Intent(
                category=TaskCategory.CHAT,
                confidence=1.0,
                method="fallback",
                raw_text=text,
                reasoning="Empty input classified as chat",
            )

        text_clean = text.strip()

        # Phase 1: Rule-based classification
        intent = self._classify_by_rules(text_clean)
        if intent and intent.confidence >= 0.8:
            logger.info(
                "intent_classified",
                category=intent.category.value,
                confidence=intent.confidence,
                method=intent.method,
                action=intent.extracted_action,
                target=intent.extracted_target,
            )
            return intent

        # Phase 2: LLM fallback (if available and rules were low confidence)
        if self._provider and self._provider.is_available():
            llm_intent = self._classify_by_model(text_clean)
            if llm_intent:
                logger.info(
                    "intent_classified_by_model",
                    category=llm_intent.category.value,
                    confidence=llm_intent.confidence,
                )
                return llm_intent

        # Phase 3: Return best rule match or default to CHAT
        if intent:
            logger.info(
                "intent_classified_low_confidence",
                category=intent.category.value,
                confidence=intent.confidence,
            )
            return intent

        return Intent(
            category=TaskCategory.CHAT,
            confidence=0.5,
            method="fallback",
            raw_text=text_clean,
            reasoning="No rule matched, defaulting to chat",
        )

    def _classify_by_rules(self, text: str) -> Optional[Intent]:
        """Classify using deterministic rules."""
        text_lower = text.lower().strip()

        for rule in self._rules:
            for pattern in rule.patterns:
                if pattern.search(text):
                    action = rule.action
                    target = ""

                    if rule.category == TaskCategory.SYSTEM_CONTROL:
                        if rule.action and rule.action not in ('', 'application'):
                            action = rule.action
                        else:
                            action, target = _extract_application(text)
                    elif rule.category == TaskCategory.FILE_OPERATION:
                        action, target = _extract_file_target(text)
                    elif rule.category == TaskCategory.GIT:
                        git_match = re.search(r"\bgit\s+(\w+)", text, re.IGNORECASE)
                        if git_match:
                            action = f"git.{git_match.group(1)}"
                    elif rule.category == TaskCategory.CODING:
                        action = rule.action
                    elif rule.category == TaskCategory.WEB_SEARCH:
                        query_match = re.search(
                            r"(?:search|google|look\s+up|find)\s+(?:for\s+)?(.+)",
                            text,
                            re.IGNORECASE,
                        )
                        if query_match:
                            target = query_match.group(1).strip()
                    elif rule.category == TaskCategory.NETWORK:
                        action = rule.action
                    elif rule.category == TaskCategory.VISION:
                        action = rule.action
                    elif rule.category == TaskCategory.TERMINAL:
                        action = rule.action
                    elif rule.category == TaskCategory.CHAT:
                        action = rule.action

                    return Intent(
                        category=rule.category,
                        confidence=rule.confidence,
                        method="rule",
                        extracted_action=action,
                        extracted_target=target,
                        raw_text=text,
                        reasoning=f"Matched rule: {rule.description}",
                    )

        return None

    def _classify_by_model(self, text: str) -> Optional[Intent]:
        """Classify using the local LLM for ambiguous cases."""
        import asyncio

        from pengu.models.base import ChatMessage

        categories = ", ".join([c.value for c in TaskCategory])
        prompt = f"""Classify this user command into exactly ONE category.

Categories: {categories}

User command: "{text}"

Respond with ONLY a JSON object:
{{"category": "CATEGORY_NAME", "confidence": 0.0-1.0, "action": "brief action", "target": "target if any"}}

No other text. Just the JSON."""

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._provider.chat(
                            [ChatMessage(role="user", content=prompt)],
                            temperature=0.1,
                            max_tokens=100,
                        ),
                    )
                    response = future.result(timeout=30)
            else:
                response = loop.run_until_complete(
                    self._provider.chat(
                        [ChatMessage(role="user", content=prompt)],
                        temperature=0.1,
                        max_tokens=100,
                    )
                )

            if response.error or not response.content:
                return None

            import json
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]).strip()

            data = json.loads(content)
            category_str = data.get("category", "CHAT").upper()

            try:
                category = TaskCategory(category_str)
            except ValueError:
                category = TaskCategory.CHAT

            return Intent(
                category=category,
                confidence=float(data.get("confidence", 0.7)),
                method="model",
                extracted_action=data.get("action", ""),
                extracted_target=data.get("target", ""),
                raw_text=text,
                reasoning=f"LLM classification: {category_str}",
            )

        except Exception as e:
            logger.warning("model_classification_failed", error=str(e))
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router: Optional[IntentRouter] = None


def get_router() -> IntentRouter:
    """Get or create the global intent router."""
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def reset_router() -> IntentRouter:
    """Reset the global router (for testing)."""
    global _router
    _router = IntentRouter()
    return _router
