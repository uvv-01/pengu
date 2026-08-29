"""
Desktop Automation — Windows mouse, keyboard, and window management.

Uses ctypes (Win32 API) directly — no external dependencies.
Provides reliable, deterministic desktop interaction.

Architecture:
  - MouseController: move, click, double-click, right-click, drag
  - KeyboardController: type, press keys, hotkeys, shortcuts
  - WindowController: find, focus, resize, minimize, maximize, close
  - DesktopAutomation: facade combining all controllers
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from dataclasses import dataclass
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.agent.desktop")

# ---------------------------------------------------------------------------
# Win32 API declarations
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

WHEEL_DELTA = 120

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001

VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_TAB = 0x09
VK_BACK = 0x08
VK_DELETE = 0x2E
VK_SPACE = 0x20
VK_HOME = 0x24
VK_END = 0x23
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28

SW_RESTORE = 9
SW_SHOW = 5
SW_HIDE = 0
SW_MINIMIZE = 6
SW_MAXIMIZE = 3

HWND_TOP = 0
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040


@dataclass
class Point:
    x: int
    y: int


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]


# ---------------------------------------------------------------------------
# Mouse Controller
# ---------------------------------------------------------------------------

class MouseController:
    """Windows mouse control via SendInput API."""

    def move(self, x: int, y: int) -> None:
        """Move mouse to absolute screen coordinates."""
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        abs_x = int(x * 65535 / screen_w)
        abs_y = int(y * 65535 / screen_h)
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = abs_x
        inp.union.mi.dy = abs_y
        inp.union.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def click(self, x: int, y: int) -> None:
        """Left click at screen coordinates."""
        self.move(x, y)
        time.sleep(0.05)
        self._button_down(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.05)
        self._button_up(MOUSEEVENTF_LEFTUP)
        logger.info("mouse_click", x=x, y=y)

    def double_click(self, x: int, y: int) -> None:
        """Double click at screen coordinates."""
        self.click(x, y)
        time.sleep(0.05)
        self._button_down(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.05)
        self._button_up(MOUSEEVENTF_LEFTUP)
        logger.info("mouse_double_click", x=x, y=y)

    def right_click(self, x: int, y: int) -> None:
        """Right click at screen coordinates."""
        self.move(x, y)
        time.sleep(0.05)
        self._button_down(MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.05)
        self._button_up(MOUSEEVENTF_RIGHTUP)
        logger.info("mouse_right_click", x=x, y=y)

    def scroll(self, x: int, y: int, delta: int = -3) -> None:
        """Scroll at screen coordinates. Negative = down, positive = up."""
        self.move(x, y)
        time.sleep(0.05)
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.mouseData = delta * WHEEL_DELTA
        inp.union.mi.dwFlags = MOUSEEVENTF_WHEEL
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        logger.info("mouse_scroll", x=x, y=y, delta=delta)

    def get_position(self) -> Point:
        """Get current mouse position."""
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return Point(pt.x, pt.y)

    def _button_down(self, flag: int) -> None:
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dwFlags = flag
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _button_up(self, flag: int) -> None:
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dwFlags = flag
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


# ---------------------------------------------------------------------------
# Keyboard Controller
# ---------------------------------------------------------------------------

class KeyboardController:
    """Windows keyboard control via SendInput API."""

    # Common VK code mapping
    VK_CODES: dict[str, int] = {
        "enter": VK_RETURN, "return": VK_RETURN,
        "escape": VK_ESCAPE, "esc": VK_ESCAPE,
        "tab": VK_TAB,
        "backspace": VK_BACK, "back": VK_BACK,
        "delete": VK_DELETE, "del": VK_DELETE,
        "space": VK_SPACE,
        "home": VK_HOME,
        "end": VK_END,
        "left": VK_LEFT, "right": VK_RIGHT,
        "up": VK_UP, "down": VK_DOWN,
        "ctrl": 0x11, "control": 0x11,
        "alt": 0x12, "menu": 0x12,
        "shift": 0x10,
        "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
        "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
        "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
        "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    }

    def press_key(self, key: str) -> None:
        """Press and release a single key."""
        vk = self.VK_CODES.get(key.lower())
        if vk is None and len(key) == 1:
            vk = ord(key.upper())
        if vk is None:
            logger.warning("unknown_key", key=key)
            return
        self._key_event(vk)
        logger.info("key_press", key=key)

    def type_text(self, text: str, delay: float = 0.02) -> None:
        """Type text character by character using keybd_event."""
        for char in text:
            vk = ord(char.upper())
            shift = char.isupper() or char in '!@#$%^&*()_+{}|:"<>?'
            if shift:
                self._key_down(0x10)  # Shift
            self._key_event(vk)
            if shift:
                self._key_up(0x10)
            if delay > 0:
                time.sleep(delay)
        logger.info("type_text", length=len(text))

    def hotkey(self, *keys: str) -> None:
        """Press a key combination (e.g., hotkey('ctrl', 'c'))."""
        vks = []
        for key in keys:
            vk = self.VK_CODES.get(key.lower())
            if vk is None and len(key) == 1:
                vk = ord(key.upper())
            if vk is not None:
                vks.append(vk)
        # Press all down
        for vk in vks:
            self._key_down(vk)
        time.sleep(0.05)
        # Release in reverse order
        for vk in reversed(vks):
            self._key_up(vk)
        logger.info("hotkey", keys="+".join(keys))

    def _key_event(self, vk: int) -> None:
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.dwFlags = KEYEVENTF_KEYUP
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _key_down(self, vk: int) -> None:
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _key_up(self, vk: int) -> None:
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = vk
        inp.union.ki.dwFlags = KEYEVENTF_KEYUP
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


# ---------------------------------------------------------------------------
# Window Controller
# ---------------------------------------------------------------------------

class WindowController:
    """Windows window management via Win32 API."""

    def get_foreground_window(self) -> int:
        """Get handle of the currently focused window."""
        return user32.GetForegroundWindow()

    def get_window_title(self, hwnd: int) -> str:
        """Get the title of a window."""
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def get_window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        """Get window position and size: (x, y, width, height)."""
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)

    def find_window(self, title: str) -> Optional[int]:
        """Find a window by title substring (case-insensitive)."""
        result = []

        def enum_callback(hwnd: int, _: int) -> bool:
            if user32.IsWindowVisible(hwnd):
                wtitle = self.get_window_title(hwnd)
                if wtitle and title.lower() in wtitle.lower():
                    result.append(hwnd)
            return True

        ENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(ENUMPROC(enum_callback), 0)
        return result[0] if result else None

    def focus_window(self, hwnd: int) -> bool:
        """Bring a window to the foreground."""
        try:
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            return True
        except Exception:
            return False

    def minimize_window(self, hwnd: int) -> None:
        """Minimize a window."""
        user32.ShowWindow(hwnd, SW_MINIMIZE)

    def maximize_window(self, hwnd: int) -> None:
        """Maximize a window."""
        user32.ShowWindow(hwnd, SW_MAXIMIZE)

    def restore_window(self, hwnd: int) -> None:
        """Restore a minimized window."""
        user32.ShowWindow(hwnd, SW_RESTORE)

    def get_active_window_info(self) -> dict[str, str]:
        """Get info about the currently active window."""
        hwnd = self.get_foreground_window()
        title = self.get_window_title(hwnd)
        rect = self.get_window_rect(hwnd)
        return {
            "hwnd": str(hwnd),
            "title": title,
            "x": str(rect[0]),
            "y": str(rect[1]),
            "width": str(rect[2]),
            "height": str(rect[3]),
        }

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen dimensions."""
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))


# ---------------------------------------------------------------------------
# Desktop Automation Facade
# ---------------------------------------------------------------------------

class DesktopAutomation:
    """
    Combined desktop automation controller.

    Provides mouse, keyboard, and window management in one interface.
    """

    def __init__(self) -> None:
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.window = WindowController()

    def click_center_of_active_window(self) -> bool:
        """Click the center of the currently active window."""
        hwnd = self.window.get_foreground_window()
        rect = self.window.get_window_rect(hwnd)
        cx = rect[0] + rect[2] // 2
        cy = rect[1] + rect[3] // 2
        self.mouse.click(cx, cy)
        return True

    def focus_and_click(self, title_substring: str) -> bool:
        """Find a window by title, focus it, and click its center."""
        hwnd = self.window.find_window(title_substring)
        if hwnd is None:
            logger.warning("window_not_found", title=title_substring)
            return False
        self.window.restore_window(hwnd)
        time.sleep(0.3)
        self.window.focus_window(hwnd)
        time.sleep(0.3)
        self.click_center_of_active_window()
        return True

    def type_in_active_window(self, text: str, press_enter: bool = False) -> None:
        """Type text into the currently focused window."""
        # Click center first to ensure focus
        self.click_center_of_active_window()
        time.sleep(0.1)
        self.keyboard.type_text(text)
        if press_enter:
            time.sleep(0.1)
            self.keyboard.press_key("enter")

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen dimensions."""
        return self.window.get_screen_size()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_desktop: Optional[DesktopAutomation] = None


def get_desktop() -> DesktopAutomation:
    """Get the global desktop automation instance."""
    global _desktop
    if _desktop is None:
        _desktop = DesktopAutomation()
    return _desktop
