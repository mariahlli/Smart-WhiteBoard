#!/usr/bin/env python3
"""Smart Interactive Whiteboard entry point.

Run ``python main.py``. The application opens directly in its GUI; use
Settings for camera, calibration, pointer color, and remote configuration.
``python main.py --smoke-test`` checks the complete pipeline without a window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.calibration import Calibration, compute_homography, map_points  # noqa: E402
from core.camera import CameraManager, SYNTHETIC_CORNERS  # noqa: E402
from core.remote_input import RemoteInput, RemoteRouter  # noqa: E402
from core.hand_tracking import HandTracker  # noqa: E402
from core.imports import ImportManager  # noqa: E402
from core.vision import PointerStabilizer, segment_pointer  # noqa: E402
from ui.interaction import BOARD_HEIGHT, BOARD_WIDTH, InteractionState  # noqa: E402
from ui.renderer import Renderer  # noqa: E402

WINDOW_NAME = "Smart Interactive Whiteboard"


def choose_import_file(request: str) -> str:
    """Open a native picker without mixing Tk and OpenCV event loops on macOS."""
    # macOS uses AppleScript here because its native chooser is reliable while
    # OpenCV owns the full-screen window.
    if sys.platform == "darwin":
        types = ('{"png", "jpg", "jpeg"}' if request == "import_image" else
                '{"com.microsoft.powerpoint.ppt", "org.openxmlformats.presentationml.presentation"}')
        script = (f'set selectedFile to choose file with prompt "Import into Study Board" of type {types}\n'
                'POSIX path of selectedFile')
        completed = subprocess.run(["osascript", "-e", script], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0:
            # AppleScript -128 is an ordinary user cancellation, not an app error.
            if "-128" in completed.stderr or "User canceled" in completed.stderr:
                return ""
            raise RuntimeError("The macOS file picker could not open.")
        return completed.stdout.strip()
    import tkinter as tk
    from tkinter import filedialog
    dialog = tk.Tk()
    try:
        dialog.withdraw()
        dialog.attributes("-topmost", True)
        filetypes = [("Images", "*.png *.jpg *.jpeg")] if request == "import_image" else [
            ("PowerPoint", "*.ppt *.pptx")]
        return filedialog.askopenfilename(parent=dialog, title="Import into Study Board", filetypes=filetypes)
    finally:
        dialog.destroy()


class WhiteboardApplication:
    """Coordinate camera, tracking, interaction state, rendering, and window events."""

    def __init__(self, synthetic: bool = False, enable_remote: bool = True):
        # Load persistent preferences first, then construct hardware/services.
        config_path = ROOT / "config.json"
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if synthetic:
            self.config["camera_index"] = -1
        self.camera = CameraManager(
            self.config.get("camera_index", -1),
            self.config.get("mirror", False),
            self.config.get("flip_vertical", False),
        )
        self.remote = RemoteInput(ROOT / "remote_keycodes.log")
        if enable_remote:
            self.remote.start()
        else:
            self.remote.status = "Disabled by command line"
        self.router = RemoteRouter(self.config["remote_left_keys"], self.config["remote_right_keys"])
        self.imports = ImportManager(ROOT)
        self.state = InteractionState(self.config, config_path, self.camera, self.router, self.imports)
        self.state.fullscreen_callback = self.set_fullscreen
        self.stabilizer = PointerStabilizer()
        self.hand_tracker = HandTracker(far_search=True)
        self.previous_pinch = False
        self.renderer = Renderer()
        self.running = True
        self.mouse_override_until = 0.0
        self.last_detection = None
        self.last_mask = None
        self.last_camera_point = None
        self.frame = None
        self.fps = 0.0
        self._fps_started = time.monotonic()
        self._fps_frames = 0

        if self.camera.index < 0:
            board = [[0, 0], [BOARD_WIDTH - 1, 0], [BOARD_WIDTH - 1, BOARD_HEIGHT - 1], [0, BOARD_HEIGHT - 1]]
            matrix = compute_homography(SYNTHETIC_CORNERS, board)
            # Keep this in memory only: an offline test must never overwrite a
            # physical camera's saved calibration in config.json.
            self.state.calibration = Calibration(matrix, SYNTHETIC_CORNERS.tolist(), board, 0.0, (640, 360))
            self.state.notify("Synthetic camera calibrated exactly. Move your mouse to take control temporarily.")

    def service_imports(self) -> None:
        """Poll workers and show a native picker only after a GUI request."""
        self.state.poll_imports()
        request = self.state.import_request
        if request is None:
            return
        self.state.import_request = None
        try:
            selected = choose_import_file(request)
        except Exception as exc:
            self.state.import_in_progress = False
            self.state.notify(str(exc) or "The file picker could not open; the whiteboard is still running.", 6)
            return
        if not selected:
            self.state.import_in_progress = False
            self.state.notify("Import cancelled; current content is unchanged.")
            return
        try:
            if request == "import_image":
                self.imports.submit_image(selected, self.state.import_generation)
                self.state.notify("Loading image in the background…")
            else:
                self.imports.submit_presentation(selected, self.state.import_generation)
                self.state.notify("Converting PowerPoint in the background. Camera tracking remains active.", 6)
        except Exception as exc:
            self.state.import_in_progress = False
            self.state.notify(str(exc), 6)

    def set_fullscreen(self, enabled: bool) -> None:
        """Ask OpenCV to switch the single application window between modes."""
        flag = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, flag)

    def mouse_callback(self, event, x, y, flags, _param=None) -> None:
        """Bridge OpenCV mouse/wheel events into the interaction state machine."""
        # In physical-camera mode ordinary mouse motion must not steal control
        # from the pointer. A deliberate mouse click still enables a 2 s fallback.
        if event == getattr(cv2, "EVENT_MOUSEWHEEL", -999) and self.state.mode == "images":
            self.state.activate("zoom_in" if flags > 0 else "zoom_out")
        elif event == cv2.EVENT_MOUSEMOVE and self.camera.index < 0:
            self.mouse_override_until = time.monotonic() + 2.0
            self.state.set_pointer((float(x), float(y)), "Mouse")
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_override_until = time.monotonic() + 2.0
            self.state.left_click((float(x), float(y)))
        elif event == cv2.EVENT_LBUTTONUP:
            self.state.left_release()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_override_until = time.monotonic() + 2.0
            self.state.right_click((float(x), float(y)))

    def process_remote(self) -> None:
        """Drain OS-level remote-key events and translate them into UI actions."""
        for event in self.remote.poll():
            self.state.last_remote_event = event
            action = self.router.route(event)
            if action == "captured":
                self.state.save_config()
                self.state.notify(f"Bound '{event.name}' successfully.", 5)
            elif action == "left":
                self.state.left_click()
            elif action == "right":
                self.state.right_click()
            elif action == "release_left":
                self.state.left_release()

    def _map_detection(self, detection) -> tuple[tuple[float, float] | None, str]:
        """Map through H, with a useful approximate fallback before calibration."""
        height, width = self.frame.shape[:2]
        calibration = self.state.calibration
        if calibration and tuple(calibration.camera_size) == (width, height):
            try:
                mapped = map_points(detection.centroid, calibration.matrix)
                if -35 <= mapped[0] < BOARD_WIDTH + 35 and -35 <= mapped[1] < BOARD_HEIGHT + 35:
                    return ((float(np.clip(mapped[0], 0, BOARD_WIDTH - 1)),
                             float(np.clip(mapped[1], 0, BOARD_HEIGHT - 1))), "Calibrated mapping")
            except ValueError:
                pass
        # Detection should still be demonstrably useful before/without valid
        # calibration. Normalized camera mapping is approximate, not a substitute
        # for the four-corner wizard, and is labeled clearly in the interface.
        x = detection.centroid[0] / max(1, width - 1) * (BOARD_WIDTH - 1)
        y = detection.centroid[1] / max(1, height - 1) * (BOARD_HEIGHT - 1)
        return (float(x), float(y)), "Approximate — run Calibration"

    def process_frame(self) -> None:
        """Run one complete camera/input/render pipeline iteration."""
        self.service_imports()
        ok, frame = self.camera.read()
        if not ok:
            self.state.notify("Camera frame unavailable. Check Settings -> Camera.", 1)
            return
        self.frame = frame
        self.state.current_frame = frame
        self.state.frame_shape = frame.shape
        mode = self.config.get("detection_mode", "hand")
        if mode == "hand":
            detection = self.hand_tracker.detect(frame)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        else:
            detection, mask, _ = segment_pointer(
                frame, np.asarray(self.config["hsv_lower"]), np.asarray(self.config["hsv_upper"]),
                "marker", self.config.get("min_area", 45), preferred_point=self.last_camera_point,
            )
        self.last_detection, self.last_mask = detection, mask
        self.state.current_detection = detection
        if detection:
            self.last_camera_point = detection.centroid
            self.state.tracking_status = "POINTER LOCKED"
            self.state.detection_area = detection.area
            self.state.hand_gesture = getattr(detection, "gesture", "MARKER")
            self.state.hand_fingers = getattr(detection, "fingers", [])
        else:
            self.state.tracking_status = "SEARCHING FOR HAND" if mode == "hand" else "SEARCHING FOR MARKER"
            self.state.detection_area = 0
            self.state.hand_gesture = self.hand_tracker.error if mode == "hand" else "NO MARKER"
            if self.stabilizer.missing > 10:
                self.last_camera_point = None
        camera_controls = time.monotonic() >= self.mouse_override_until
        if camera_controls:
            board_point = None
            mapping = "No pointer"
            if detection:
                board_point, mapping = self._map_detection(detection)
            stable = self.stabilizer.update(board_point)
            self.state.mapping_status = mapping
            self.state.set_pointer(stable, f"Camera {self.fps:.0f} FPS")
        pinch = bool(detection is not None and getattr(detection, "pinch", False))
        if camera_controls and self.state.pointer is not None:
            if pinch and not self.previous_pinch:
                self.state.left_click()
            elif self.previous_pinch and not pinch:
                self.state.left_release()
        self.previous_pinch = pinch
        if getattr(self.state, "reset_requested", False):
            self.stabilizer.reset()
            self.last_camera_point = None
            self.state.reset_requested = False
        self.process_remote()

    def handle_fallback_key(self, key: int) -> None:
        """Handle keyboard shortcuts used when the OS remote listener is unavailable."""
        if key < 0:
            return
        if key == 27 and self.state.presentation.active:
            self.state.activate("presentation_exit")
        elif key in (27, ord("q")):
            self.running = False
        elif key == ord("f"):
            self.state.activate("fullscreen")
        elif key == ord("h"):
            self.state.activate("home")
        elif key == ord("d"):
            self.state.show_debug = not self.state.show_debug
        elif key == ord("c"):
            self.state.activate("calibrate")
        elif key in (ord("+"), ord("=")) and self.state.mode == "images":
            self.state.activate("zoom_in")
        elif key == ord("-") and self.state.mode == "images":
            self.state.activate("zoom_out")
        elif self.state.mode == "presentation" and key in (ord("n"), 32, 13):
            self.state.activate("deck_next")
        elif self.state.mode == "presentation" and key in (ord("p"), 8):
            self.state.activate("deck_prev")
        elif key in (13, 32, ord("n")):
            self.state.left_click()
        elif key in (8, ord("p")):
            self.state.right_click()

    def run(self, smoke_test: bool = False) -> None:
        """Run the continuous OpenCV loop or execute exactly one smoke-test frame."""
        if not smoke_test:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        try:
            while self.running:
                self.process_frame()
                if self.frame is None:
                    if smoke_test:
                        raise RuntimeError("No camera frame available")
                    time.sleep(0.03)
                    continue
                canvas = self.renderer.render(self.state, self.frame, self.last_mask, self.last_detection, self.remote)
                self._fps_frames += 1
                elapsed = time.monotonic() - self._fps_started
                if elapsed >= 0.75:
                    self.fps = self._fps_frames / elapsed
                    self._fps_frames = 0
                    self._fps_started = time.monotonic()
                if smoke_test:
                    print(json.dumps({
                        "status": "ok", "frame": list(self.frame.shape), "canvas": list(canvas.shape),
                        "camera": "synthetic" if self.camera.index < 0 else self.camera.index,
                        "calibrated": self.state.calibration is not None,
                        "pointer_detected": self.last_detection is not None,
                        "mapping": getattr(self.state, "mapping_status", "Mouse override"),
                    }))
                    break
                cv2.imshow(WINDOW_NAME, canvas)
                self.handle_fallback_key(cv2.waitKey(1) & 0xFF)
        finally:
            self.remote.stop()
            self.imports.close()
            self.hand_tracker.close()
            self.camera.release()
            if not smoke_test:
                cv2.destroyAllWindows()


def main() -> int:
    """Parse command-line options and start the application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Force the offline synthetic camera")
    parser.add_argument("--no-remote", action="store_true", help="Disable the global keyboard listener")
    parser.add_argument("--smoke-test", action="store_true", help="Process and render one frame without opening a window")
    args = parser.parse_args()
    app = WhiteboardApplication(args.synthetic, not args.no_remote)
    app.run(args.smoke_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
