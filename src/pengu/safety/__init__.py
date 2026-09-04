"""
Safety, Permissions and Risk Classification -- central policy layer for Pengu.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.safety")


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    BLOCKED = "blocked"


@dataclass
class ActionClassification:
    action: str
    target: str
    risk_level: RiskLevel
    reason: str
    reversible: bool = True
    needs_confirmation: bool = False
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "reversible": self.reversible,
            "needs_confirmation": self.needs_confirmation,
            "explanation": self.explanation,
        }


class RiskClassifier:
    _BLOCKED_PATTERNS = [
        re.compile(r"\b(format|wipe)\s+(c:|disk|drive|volume)", re.I),
        re.compile(r"\b(rmdir|rd)\s+[/\\]", re.I),
        re.compile(r"\bdel\s+[/\\][sqr]", re.I),
        re.compile(r"\bshutdown\s+/[sf]", re.I),
        re.compile(r"\bregedit\b", re.I),
        re.compile(r"\bnet\s+user\b", re.I),
        re.compile(r"\bmkfs\b", re.I),
        re.compile(r"\bdd\s+if=", re.I),
    ]
    _HIGH_RISK_PATTERNS = [
        re.compile(r"\bdelete\b.*\b(folder|directory|files?|all|everything)\b", re.I),
        re.compile(r"\bremove\b.*\b(folder|directory|files?|all)\b", re.I),
        re.compile(r"\buninstall\b", re.I),
        re.compile(r"\b(rm|rmdir)\b", re.I),
        re.compile(r"\bmove\b.*\b(to\s+recycle|trash)\b", re.I),
        re.compile(r"\bempty\s+(recycle\s+bin|trash)\b", re.I),
        re.compile(r"\bgit\s+push\s+--force\b", re.I),
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
        re.compile(r"\bgit\s+clean\s+-fd\b", re.I),
    ]
    _MEDIUM_RISK_PATTERNS = [
        re.compile(r"\binstall\b", re.I),
        re.compile(r"\bmodify\b.*\b(settings?|config|configuration)\b", re.I),
        re.compile(r"\bchange\b.*\b(settings?|password|wallpaper|theme)\b", re.I),
        re.compile(r"\bcreate\b.*\b(file|folder|directory)\b", re.I),
        re.compile(r"\bsave\b.*\b(to|in)\b", re.I),
        re.compile(r"\bwrite\b.*\b(to|file)\b", re.I),
        re.compile(r"\bsend\b", re.I),
        re.compile(r"\bsubmit\b", re.I),
        re.compile(r"\bgit\s+push\b", re.I),
        re.compile(r"\bgit\s+commit\b", re.I),
        re.compile(r"\bmove\b.*\b(to|into)\b", re.I),
        re.compile(r"\brename\b", re.I),
        re.compile(r"\bcopy\b.*\b(to|into)\b", re.I),
    ]

    def classify(self, action: str, target: str = "", context: Optional[dict[str, Any]] = None) -> ActionClassification:
        combined = f"{action} {target}".lower().strip()
        for pattern in self._BLOCKED_PATTERNS:
            if pattern.search(combined):
                return ActionClassification(action=action, target=target, risk_level=RiskLevel.BLOCKED, reason=f"Blocked: {pattern.pattern}", reversible=False, explanation="This action is blocked because it could cause irreversible system damage.")
        for pattern in self._HIGH_RISK_PATTERNS:
            if pattern.search(combined):
                return ActionClassification(action=action, target=target, risk_level=RiskLevel.HIGH_RISK, reason=f"High-risk: {pattern.pattern}", reversible=False, needs_confirmation=True, explanation=self._explain_high_risk(action, target))
        for pattern in self._MEDIUM_RISK_PATTERNS:
            if pattern.search(combined):
                return ActionClassification(action=action, target=target, risk_level=RiskLevel.MEDIUM_RISK, reason=f"Medium-risk: {pattern.pattern}", reversible=True, needs_confirmation=True, explanation=self._explain_medium_risk(action, target))
        safe_actions = {"system.info", "system.battery", "system.volume", "system.wallpaper", "network.wifi_status", "network.list_wifi", "browser_get_state", "browser_read", "browser_find_elements", "browser_verify", "screen_inspect", "get_active_window", "process.list", "application.is_running", "list_files", "web_search", "memory.search", "memory.list", "chat", "git.status", "git.log", "git.diff", "git.branch"}
        if action in safe_actions:
            return ActionClassification(action=action, target=target, risk_level=RiskLevel.SAFE, reason="Read-only action")
        return ActionClassification(action=action, target=target, risk_level=RiskLevel.LOW_RISK, reason="Default classification")

    def _explain_high_risk(self, action: str, target: str) -> str:
        if "delete" in action.lower() or "remove" in action.lower():
            return f"This will permanently delete '{target}'. NOT easily reversible."
        if "uninstall" in action.lower():
            return f"This will uninstall '{target}'."
        if "git" in action.lower() and "force" in action.lower():
            return "This will force-push to the remote repository."
        return f"This is a high-risk action involving '{target}'."

    def _explain_medium_risk(self, action: str, target: str) -> str:
        if "install" in action.lower():
            return f"This will install '{target}'."
        if "create" in action.lower():
            return f"This will create '{target}'."
        if "send" in action.lower() or "submit" in action.lower():
            return f"This will send/submit '{target}'. Please verify."
        if "git" in action.lower():
            return f"This will execute a git operation: {target}."
        return f"This action may modify '{target}'."


class ConfirmationManager:
    def __init__(self) -> None:
        self._pending: dict[str, ActionClassification] = {}
        self._session_permissions: dict[str, float] = {}
        self._permission_expiry: float = 300.0

    def request_confirmation(self, classification: ActionClassification) -> str:
        conf_id = f"conf_{int(time.time() * 1000)}"
        self._pending[conf_id] = classification
        risk = classification.risk_level
        explanation = classification.explanation or f"This is a {risk.value} action."
        if risk == RiskLevel.HIGH_RISK:
            return f"HIGH RISK ACTION\n\n{explanation}\n\nDo you want me to continue? (yes/no)"
        elif risk == RiskLevel.MEDIUM_RISK:
            return f"This requires confirmation\n\n{explanation}\n\nShould I proceed? (yes/no)"
        return ""

    def check_session_permission(self, action: str, target: str = "") -> bool:
        key = f"{action}:{target}"
        now = time.time()
        self._session_permissions = {k: v for k, v in self._session_permissions.items() if now - v < self._permission_expiry}
        return key in self._session_permissions

    def grant_session_permission(self, action: str, target: str = "") -> None:
        self._session_permissions[f"{action}:{target}"] = time.time()

    def revoke_session_permission(self, action: str, target: str = "") -> None:
        self._session_permissions.pop(f"{action}:{target}", None)

    def resolve(self, conf_id: str, approved: bool) -> Optional[ActionClassification]:
        return self._pending.pop(conf_id, None)


class SafetyPolicy:
    def __init__(self) -> None:
        self._classifier = RiskClassifier()
        self._confirmer = ConfirmationManager()

    @property
    def classifier(self) -> RiskClassifier:
        return self._classifier

    @property
    def confirmation_manager(self) -> ConfirmationManager:
        return self._confirmer

    def check(self, action: str, target: str = "", context: Optional[dict[str, Any]] = None) -> ActionClassification:
        classification = self._classifier.classify(action, target, context)
        if classification.needs_confirmation and self._confirmer.check_session_permission(action, target):
            classification.needs_confirmation = False
        logger.info("safety_check", action=action, target=target[:50], risk=classification.risk_level.value, confirm=classification.needs_confirmation)
        return classification

    def confirm_action(self, classification: ActionClassification) -> str:
        return self._confirmer.request_confirmation(classification)

    def grant_permission(self, action: str, target: str = "") -> None:
        self._confirmer.grant_session_permission(action, target)


_policy = None


def get_safety_policy() -> SafetyPolicy:
    global _policy
    if _policy is None:
        _policy = SafetyPolicy()
    return _policy


def reset_safety_policy() -> SafetyPolicy:
    global _policy
    _policy = SafetyPolicy()
    return _policy
