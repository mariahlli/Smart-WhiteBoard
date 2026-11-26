"""Presentation-remote keyboard listener, inspector, and event routing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Callable


@dataclass
class RemoteEvent:
    """One normalized OS keyboard press or release event."""

    name: str
    code: str
    pressed: bool


def normalize_key(key) -> tuple[str, str]:
    """Convert pynput Key/KeyCode values to stable, readable identifiers."""
    name_attr = getattr(key, "name", None)
    if name_attr:
        name = str(name_attr).lower()
    else:
        char = getattr(key, "char", None)
        name = str(char).lower() if char is not None else str(key).replace("Key.", "").lower()
    virtual = getattr(key, "vk", None)
    scan = getattr(key, "scan", None)
    code = f"vk={virtual if virtual is not None else '?'} scan={scan if scan is not None else '?'}"
    return name, code


class RemoteInput:
    """Background OS-level listener with a thread-safe application queue.

    The application—not the operating system—receives virtual clicks at the
    tracked pointer. This avoids moving the real desktop mouse unexpectedly.
    """

    def __init__(self, log_path: str | Path):
        # A bounded queue prevents the background listener from growing forever.
        self.log_path = Path(log_path)
        self.events: deque[RemoteEvent] = deque(maxlen=200)
        self.lock = threading.Lock()
        self.listener = None
        self.available = False
        self.status = "Starting keyboard listener…"
        self.last_event: RemoteEvent | None = None

    def start(self) -> None:
        """Start the global OS listener and degrade safely without permissions."""
        try:
            from pynput import keyboard
            self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self.listener.daemon = True
            self.listener.start()
            self.available = True
            self.status = "Listening globally"
        except Exception as exc:  # macOS accessibility permissions can deny access
            self.available = False
            self.status = f"Global listener unavailable: {exc}. Use focused-window fallback keys."

    def _record(self, key, pressed: bool) -> None:
        """Normalize, queue, and optionally append one key event to the log."""
        name, code = normalize_key(key)
        event = RemoteEvent(name, code, pressed)
        with self.lock:
            self.events.append(event)
            self.last_event = event
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now().isoformat(timespec='milliseconds')} "
                            f"{'PRESS' if pressed else 'RELEASE'} {name} {code}\n")
        except OSError:
            pass

    def _on_press(self, key) -> None:
        self._record(key, True)

    def _on_release(self, key) -> None:
        self._record(key, False)

    def poll(self) -> list[RemoteEvent]:
        """Atomically drain pending events for the main application loop."""
        with self.lock:
            result = list(self.events)
            self.events.clear()
        return result

    def stop(self) -> None:
        """Stop the OS listener when the application exits."""
        if self.listener is not None:
            self.listener.stop()


class RemoteRouter:
    """Bind observed key names to virtual left/right clicks and inspection."""

    def __init__(self, left_keys: list[str], right_keys: list[str]):
        self.left_keys = set(left_keys)
        self.right_keys = set(right_keys)
        self.capture_action: str | None = None

    def begin_capture(self, action: str) -> None:
        """Enter a short rebind state for the left or right action."""
        if action not in {"left", "right"}:
            raise ValueError("Action must be left or right")
        self.capture_action = action

    def route(self, event: RemoteEvent) -> str | None:
        """Route press/release events through capture or configured bindings."""
        if event.pressed and self.capture_action:
            action = self.capture_action
            if action == "left":
                self.left_keys = {event.name}
            else:
                self.right_keys = {event.name}
            self.capture_action = None
            return "captured"
        if not event.pressed:
            return "release_left" if event.name in self.left_keys else None
        if event.name in self.left_keys:
            return "left"
        if event.name in self.right_keys:
            return "right"
        return None
