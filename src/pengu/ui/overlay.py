"""
Pengu Desktop Overlay — lightweight always-on-top UI panel.

Shows current state, status messages, and tool activity.
Uses tkinter for minimal dependencies.

States:
  STANDBY    → small dot indicator
  LISTENING  → blue pulse
  THINKING   → yellow pulse
  EXECUTING  → green
  SPEAKING   → cyan pulse
  ERROR      → red
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.ui.overlay")


class OverlayState:
    """Overlay visual states."""
    STANDBY = "STANDBY"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ERROR = "ERROR"


# Colors for each state
STATE_COLORS = {
    OverlayState.STANDBY: {"bg": "#0a0e17", "fg": "#64748b", "dot": "#334155", "text": "STANDBY"},
    OverlayState.LISTENING: {"bg": "#0a0e17", "fg": "#00d4ff", "dot": "#00d4ff", "text": "LISTENING..."},
    OverlayState.THINKING: {"bg": "#0a0e17", "fg": "#f59e0b", "dot": "#f59e0b", "text": "THINKING..."},
    OverlayState.EXECUTING: {"bg": "#0a0e17", "fg": "#22c55e", "dot": "#22c55e", "text": "EXECUTING..."},
    OverlayState.SPEAKING: {"bg": "#0a0e17", "fg": "#00d4ff", "dot": "#00d4ff", "text": "SPEAKING..."},
    OverlayState.ERROR: {"bg": "#0a0e17", "fg": "#ef4444", "dot": "#ef4444", "text": "ERROR"},
}


class PenguOverlay:
    """
    Desktop overlay for Pengu.

    Always-on-top panel showing assistant state.
    Positioned in the bottom-right corner.
    """

    def __init__(self) -> None:
        self._root: Optional[tk.Tk] = None
        self._state_label: Optional[tk.Label] = None
        self._message_label: Optional[tk.Label] = None
        self._dot_label: Optional[tk.Label] = None
        self._current_state = OverlayState.STANDBY
        self._visible = False
        self._thread: Optional[threading.Thread] = None
        self._pulse_id: Optional[str] = None
        self._pulse_on = False

    def start(self) -> None:
        """Start the overlay in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("overlay_started")

    def stop(self) -> None:
        """Stop the overlay."""
        if self._root:
            self._root.after(0, self._root.destroy)
            self._root = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("overlay_stopped")

    def _run(self) -> None:
        """Run the tkinter main loop."""
        try:
            self._root = tk.Tk()
            self._root.title("Pengu")
            self._root.overrideredirect(True)  # Remove title bar
            self._root.attributes("-topmost", True)  # Always on top
            self._root.attributes("-alpha", 0.92)  # Slight transparency
            self._root.configure(bg="#0a0e17")

            # Set window size and position
            width = 220
            height = 80
            screen_width = self._root.winfo_screenwidth()
            x = screen_width - width - 20
            y = self._root.winfo_screenheight() - height - 60
            self._root.geometry(f"{width}x{height}+{x}+{y}")

            # Main frame
            frame = tk.Frame(self._root, bg="#0a0e17", padx=12, pady=8)
            frame.pack(fill=tk.BOTH, expand=True)

            # Top row: dot + title
            top_row = tk.Frame(frame, bg="#0a0e17")
            top_row.pack(fill=tk.X, pady=(0, 4))

            self._dot_label = tk.Label(
                top_row,
                text="\u25cf",  # Filled circle
                font=("Segoe UI", 12),
                fg="#334155",
                bg="#0a0e17",
            )
            self._dot_label.pack(side=tk.LEFT, padx=(0, 6))

            title_label = tk.Label(
                top_row,
                text="PENGU",
                font=("Consolas", 11, "bold"),
                fg="#e2e8f0",
                bg="#0a0e17",
            )
            title_label.pack(side=tk.LEFT)

            # State label
            self._state_label = tk.Label(
                frame,
                text="STANDBY",
                font=("Consolas", 9),
                fg="#64748b",
                bg="#0a0e17",
                anchor="w",
            )
            self._state_label.pack(fill=tk.X)

            # Message label
            self._message_label = tk.Label(
                frame,
                text="",
                font=("Segoe UI", 9),
                fg="#94a3b8",
                bg="#0a0e17",
                anchor="w",
                wraplength=190,
            )
            self._message_label.pack(fill=tk.X)

            # Make window click-through for interaction
            self._root.bind("<Button-1>", self._on_click)

            self._visible = True
            self._root.mainloop()

        except Exception as e:
            logger.error("overlay_error", error=str(e))

    def _on_click(self, event: tk.Event) -> None:
        """Handle click on overlay."""
        # Could be used to toggle visibility or show menu
        pass

    def set_state(self, state: str, message: str = "") -> None:
        """Update the overlay state."""
        self._current_state = state
        if self._root:
            self._root.after(0, self._update_ui, state, message)

    def _update_ui(self, state: str, message: str) -> None:
        """Update UI elements (must be called from tkinter thread)."""
        colors = STATE_COLORS.get(state, STATE_COLORS[OverlayState.STANDBY])

        if self._dot_label:
            self._dot_label.configure(fg=colors["dot"])

        if self._state_label:
            self._state_label.configure(text=colors["text"], fg=colors["fg"])

        if self._message_label:
            self._message_label.configure(text=message)

        # Pulse effect for listening/speaking
        if state in (OverlayState.LISTENING, OverlayState.SPEAKING):
            self._start_pulse()
        else:
            self._stop_pulse()

    def _start_pulse(self) -> None:
        """Start pulsing the dot."""
        if self._root and self._dot_label:
            self._pulse_on = not self._pulse_on
            color = "#00d4ff" if self._pulse_on else "#334155"
            self._dot_label.configure(fg=color)
            self._pulse_id = self._root.after(500, self._start_pulse)

    def _stop_pulse(self) -> None:
        """Stop pulsing."""
        if self._pulse_id and self._root:
            self._root.after_cancel(self._pulse_id)
            self._pulse_id = None

    def show(self) -> None:
        """Show the overlay."""
        if self._root:
            self._root.after(0, lambda: self._root.deiconify() if self._root else None)
            self._visible = True

    def hide(self) -> None:
        """Hide the overlay."""
        if self._root:
            self._root.after(0, lambda: self._root.withdraw() if self._root else None)
            self._visible = False

    def toggle(self) -> None:
        """Toggle overlay visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()
