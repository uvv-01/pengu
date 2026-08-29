"""
Global hotkey support for Pengu.

Uses Windows API (RegisterHotKey/UnregisterHotKey) via ctypes.
No external dependencies required.

Default hotkey: Ctrl + Alt + P
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
from typing import Callable, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.hotkey")

# Windows API constants
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
HOTKEY_ID = 1

# Win32 API
user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


class GlobalHotkey:
    """
    Windows global hotkey using RegisterHotKey/UnregisterHotKey.

    Usage:
        hotkey = GlobalHotkey()
        hotkey.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)
        hotkey.start()
        ...
        hotkey.stop()
    """

    def __init__(self) -> None:
        self._registered = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[], None]] = None

    def register(
        self,
        modifiers: int,
        vk_code: int,
        callback: Callable[[], None],
        hotkey_id: int = HOTKEY_ID,
    ) -> bool:
        """Register a global hotkey."""
        if user32 is None:
            logger.warning("hotkey_not_available_not_windows")
            return False

        if self._registered:
            self.unregister()

        self._callback = callback

        result = user32.RegisterHotKey(None, hotkey_id, modifiers, vk_code)
        if result == 0:
            logger.warning("hotkey_register_failed", modifiers=modifiers, vk=vk_code)
            return False

        self._registered = True
        logger.info("hotkey_registered", modifiers=modifiers, vk=vk_code)
        return True

    def register_default(self, callback: Callable[[], None]) -> bool:
        """Register Ctrl+Alt+P as the default hotkey."""
        return self.register(MOD_CONTROL | MOD_ALT, ord("P"), callback)

    def unregister(self, hotkey_id: int = HOTKEY_ID) -> None:
        """Unregister the global hotkey."""
        if user32 is None:
            return
        if self._registered:
            user32.UnregisterHotKey(None, hotkey_id)
            self._registered = False
            logger.info("hotkey_unregistered")

    def start(self) -> None:
        """Start listening for hotkey messages in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._message_loop, daemon=True, name="hotkey-listener"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop listening and unregister."""
        self._running = False
        self.unregister()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _message_loop(self) -> None:
        """Windows message loop to receive hotkey events."""
        if user32 is None:
            return

        msg = ctypes.wintypes.MSG()
        while self._running:
            result = user32.PeekMessageW(
                ctypes.byref(msg), None, WM_HOTKEY, WM_HOTKEY, 1
            )
            if result != 0:
                if msg.wParam == HOTKEY_ID:
                    logger.info("hotkey_pressed")
                    if self._callback:
                        try:
                            self._callback()
                        except Exception as e:
                            logger.error("hotkey_callback_error", error=str(e))
            else:
                import time
                time.sleep(0.05)

    @property
    def is_registered(self) -> bool:
        return self._registered


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_hotkey: Optional[GlobalHotkey] = None


def get_hotkey() -> GlobalHotkey:
    """Get the global hotkey instance."""
    global _hotkey
    if _hotkey is None:
        _hotkey = GlobalHotkey()
    return _hotkey
