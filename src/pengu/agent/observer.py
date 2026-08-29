"""
Screen Observer — observe the current desktop state.

Provides:
  - Active window detection (title, app, position)
  - UI element enumeration via Windows UI Automation (UIA)
  - Screen content inspection
  - Element location by name/text/type

Uses Windows UI Automation when available, falls back to window info.
Does NOT send screenshots to external models by default.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.agent.observer")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class UIElement:
    """A detected UI element on screen."""
    name: str = ""
    control_type: str = ""   # "Button", "Edit", "Text", "Hyperlink", etc.
    automation_id: str = ""
    bounding_rect: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    is_enabled: bool = True
    is_visible: bool = True
    children: list["UIElement"] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        """Get center coordinates of the element."""
        x, y, w, h = self.bounding_rect
        return (x + w // 2, y + h // 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.control_type,
            "rect": self.bounding_rect,
            "enabled": self.is_enabled,
        }


@dataclass
class ScreenState:
    """Complete description of the current screen state."""
    active_window_title: str = ""
    active_window_app: str = ""
    active_window_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    screen_width: int = 0
    screen_height: int = 0
    elements: list[UIElement] = field(default_factory=list)
    timestamp: float = 0.0

    def get_summary(self) -> str:
        """Human-readable summary of screen state."""
        parts = [f"Active: {self.active_window_title}"]
        if self.active_window_app:
            parts.append(f"App: {self.active_window_app}")
        parts.append(f"Screen: {self.screen_width}x{self.screen_height}")
        if self.elements:
            btn_names = [e.name for e in self.elements[:10] if e.name]
            if btn_names:
                parts.append(f"Elements: {', '.join(btn_names)}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Screen Observer
# ---------------------------------------------------------------------------

class ScreenObserver:
    """
    Observe the current desktop state.

    Uses Windows UI Automation (UIA) for element discovery when available.
    Falls back to basic window info when UIA is unavailable.
    """

    def __init__(self) -> None:
        self._uia_available: Optional[bool] = None
        self._uia = None  # UIAutomation core object

    def _ensure_uia(self) -> bool:
        """Try to initialize Windows UI Automation."""
        if self._uia_available is not None:
            return self._uia_available
        try:
            import comtypes.client
            from ctypes import windll
            # Try COM initialization for UIA
            try:
                comtypes.CoInitialize()
            except Exception:
                pass
            # Use the simpler approach: PowerShell-based UIA via ctypes
            self._uia_available = False  # Will use window-based detection
            logger.info("uia_fallback", reason="using_window_detection")
            return False
        except ImportError:
            self._uia_available = False
            return False

    def get_active_window(self) -> dict[str, Any]:
        """Get information about the currently active window."""
        from pengu.agent.desktop import get_desktop
        desktop = get_desktop()
        info = desktop.window.get_active_window_info()

        # Extract app name from title
        title = info.get("title", "")
        app = self._extract_app_name(title)
        info["app"] = app

        logger.info("active_window", title=title[:80], app=app)
        return info

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen dimensions."""
        from pengu.agent.desktop import get_desktop
        return get_desktop().window.get_screen_size()

    def get_elements(self) -> list[UIElement]:
        """
        Get UI elements from the active window.
        
        Uses PowerShell/UIA automation when available.
        Returns empty list if UIA is not available.
        """
        try:
            return self._get_elements_via_powershell()
        except Exception as e:
            logger.debug("element_detection_failed", error=str(e))
            return []

    def find_element(self, name: str = "", control_type: str = "") -> Optional[UIElement]:
        """Find a UI element by name or type."""
        elements = self.get_elements()
        for elem in elements:
            if name and name.lower() in elem.name.lower():
                return elem
            if control_type and control_type.lower() == elem.control_type.lower():
                return elem
        return None

    def find_all_elements(self, name: str = "", control_type: str = "") -> list[UIElement]:
        """Find all UI elements matching criteria."""
        elements = self.get_elements()
        results = []
        for elem in elements:
            if name and name.lower() in elem.name.lower():
                results.append(elem)
            elif control_type and control_type.lower() == elem.control_type.lower():
                results.append(elem)
        return results

    def get_state(self) -> ScreenState:
        """Get complete screen state."""
        active = self.get_active_window()
        screen_w, screen_h = self.get_screen_size()
        elements = self.get_elements()
        return ScreenState(
            active_window_title=active.get("title", ""),
            active_window_app=active.get("app", ""),
            active_window_rect=(
                int(active.get("x", 0)),
                int(active.get("y", 0)),
                int(active.get("width", 0)),
                int(active.get("height", 0)),
            ),
            screen_width=screen_w,
            screen_height=screen_h,
            elements=elements,
            timestamp=time.time(),
        )

    def _get_elements_via_powershell(self) -> list[UIElement]:
        """Get UI elements using PowerShell UIA automation."""
        import subprocess
        # PowerShell script to enumerate UI elements of the foreground window
        ps_script = """
Add-Type -AssemblyName UIAutomationClient
$uia = [System.Windows.Automation.AutomationElement]::RootElement
$focus = [System.Windows.Automation.AutomationElement]::FocusedElement
if ($focus -ne $null) {
    $treeWalker = [System.Windows.Automation.TreeWalker]::RawViewWalker
    $child = $treeWalker.GetFirstChild($focus)
    while ($child -ne $null) {
        try {
            $name = $child.Current.Name
            $type = $child.Current.ControlType.ProgrammaticName
            $rect = $child.Current.BoundingRectangle
            if ($name -ne "" -and $rect.Width -gt 0) {
                Write-Output "$name|$type|$($rect.X)|$($rect.Y)|$($rect.Width)|$($rect.Height)"
            }
        } catch {}
        $child = $treeWalker.GetNextSibling($child)
    }
}
"""
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            elements = []
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    parts = line.strip().split("|")
                    if len(parts) >= 6:
                        elements.append(UIElement(
                            name=parts[0],
                            control_type=parts[1].replace("ControlTypeId.", ""),
                            bounding_rect=(
                                int(float(parts[2])),
                                int(float(parts[3])),
                                int(float(parts[4])),
                                int(float(parts[5])),
                            ),
                        ))
            return elements
        except Exception as e:
            logger.debug("powershell_uia_failed", error=str(e))
            return []

    def _extract_app_name(self, title: str) -> str:
        """Extract the application name from a window title."""
        if not title:
            return ""
        # Common patterns: "Page Title - App Name" or "App Name - Page Title"
        separators = [" - ", " — ", " | ", " :: ", " · "]
        for sep in separators:
            if sep in title:
                parts = title.split(sep)
                # The app name is usually the last part
                return parts[-1].strip()
        return title


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_observer: Optional[ScreenObserver] = None


def get_observer() -> ScreenObserver:
    """Get the global screen observer."""
    global _observer
    if _observer is None:
        _observer = ScreenObserver()
    return _observer
