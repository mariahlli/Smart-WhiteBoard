"""Application state machine and GUI-only configuration workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable
import cv2
import numpy as np

from core.calibration import Calibration
from core.imports import ImportManager
from core.vision import auto_sample_marker, sample_hsv_bounds
from ui.primitives import contained_rect

# Shared screen geometry: these rectangles are in logical 1280x720 board pixels.
BOARD_WIDTH, BOARD_HEIGHT = 1280, 720
CONTENT = (250, 100, 985, 560)                 # Main drawing and interaction canvas.
CAMERA_RECT = (270, 120, 960, 540)             # Live camera preview used for calibration.
SETTINGS_PREVIEW = (490, 190, 640, 360)        # Camera preview used for pointer sampling.
PRESENTATION_RECT = (250, 100, 985, 452)       # Fitted presentation slide viewport.
PRESENTATION_THUMBNAILS = (250, 568, 985, 90)  # Reserved thumbnail-strip area.
PRESENTATION_THUMBNAIL_COUNT = 5               # Maximum visible presentation thumbnails.
PEN_BACKGROUND = (247, 248, 250)               # Eraser preview color.


@dataclass
class Button:
    """One rendered control and the action it activates."""

    action: str
    label: str
    rect: tuple[int, int, int, int]
    active: bool = False
    accent: str = "normal"


@dataclass
class Stroke:
    """A recorded ink stroke in screen or normalized presentation coordinates."""

    points: list[tuple[float, float]]
    color: tuple[int, int, int]
    thickness: int


@dataclass
class BoardObject:
    """A movable board shape with its current position and original reset position."""

    center: np.ndarray
    size: int
    color: tuple[int, int, int]
    kind: str
    home: tuple[float, float]


@dataclass
class PresentationState:
    """An imported deck and its independent, normalized per-slide ink."""
    path: Path | None = None
    name: str = ""
    slides: list[np.ndarray] = field(default_factory=list)
    thumbnails: list[np.ndarray] = field(default_factory=list)
    index: int = 0
    annotations: list[list[Stroke]] = field(default_factory=list)
    thumbnail_offset: int = 0
    active: bool = False
    return_fullscreen: bool = False


def inside(point, rect) -> bool:
    """Return whether a board-space point falls inside an (x, y, w, h) rectangle."""
    if point is None:
        return False
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


class InteractionState:
    """Owns every mutable interaction; rendering remains a separate concern."""

    def __init__(self, config: dict, config_path: Path, camera, router: RemoteRouter,
                importer: ImportManager | None = None):
        # External services and persistent settings are retained here for every mode.
        self.config = config
        self.config_path = Path(config_path)
        self.camera = camera
        self.router = router
        self.importer = importer
        self.import_request: str | None = None
        self.import_in_progress = False
        self.import_generation = 0

        # Navigation state determines which mode-specific controls and canvas are shown.
        self.mode = "home"
        self.settings_tab = "camera"

        # Pointer state is shared by mouse, camera, hand, remote, and calibration input.
        self.pointer: tuple[float, float] | None = (640, 360)
        self.previous_pointer = self.pointer
        self.pointer_source = "Mouse"
        self.frame_shape = (360, 640)
        self.calibration = Calibration.from_dict(config.get("calibration"))
        self.tracking_status = "SEARCHING FOR POINTER"
        self.detection_area = 0
        self.mapping_status = "Calibrated mapping" if self.calibration else "Approximate — run Calibration"
        self.calibration_points: list[tuple[float, float]] = []
        self.calibration_message = "Click the projected board corners in order."
        self.sample_mode = False
        self.show_debug = False

        # Temporary status and visual effects are rendered by the UI layer.
        self.status = "Ready — choose a mode or open Settings to calibrate."
        self.status_until = time.monotonic() + 5
        self.ripples: list[tuple[float, float, float, str]] = []
        self.fullscreen = False
        self.fullscreen_callback: Callable[[], None] | None = None

        # Freehand drawing state.
        self.pen_color = (55, 70, 230)
        self.pen_thickness = 5
        self.eraser = False
        self.draw_toggle_mode = True
        self.drawing = False
        self.strokes: list[Stroke] = []

        # Imported images are loaded lazily and start with no bundled demo images.
        self.image_files = []
        self.image_index = 0
        self.images = []
        self._original_image_files = self.image_files.copy()
        self._original_images = self.images.copy()
        self.image_zoom = 1.0
        self.image_pan = [0.0, 0.0]
        self.image_panning = False
        self.presentation = PresentationState()

        # Built-in lesson slides and per-slide annotations.
        lesson_path = self.config_path.parent / "assets" / "lessons" / "slides.json"
        self.slides = json.loads(lesson_path.read_text(encoding="utf-8"))
        self.slide_index = 0
        self.slide_annotations: list[list[Stroke]] = [[] for _ in self.slides]

        # Playground objects have mutable positions/colors and fixed reset positions.
        self.objects = [
            BoardObject(np.array([430.0, 260.0]), 54, (60, 105, 225), "circle", (430, 260)),
            BoardObject(np.array([690.0, 365.0]), 65, (65, 165, 100), "square", (690, 365)),
            BoardObject(np.array([960.0, 245.0]), 62, (190, 105, 55), "diamond", (960, 245)),
        ]
        self.grabbed_object: int | None = None
        self.selected_object: int | None = 0
        self.object_color = (60, 105, 225)
        self.reset_armed_until = 0.0
        self.reset_requested = False
        self.hand_gesture = "SHOW YOUR HAND"
        self.hand_fingers = []

    def notify(self, message: str, seconds: float = 3.0) -> None:
        """Set a temporary status message for the on-screen status bar."""
        self.status = message
        self.status_until = time.monotonic() + seconds

    def poll_imports(self) -> None:
        """Apply completed worker results without ever blocking the camera loop."""
        if self.importer is None:
            return
        for result in self.importer.poll():
            if result.get("generation") != self.import_generation:
                continue
            self.import_in_progress = False
            if result["kind"] == "error":
                self.notify(result["message"], 7)
                continue
            self._stop_all_motion()
            if result["kind"] == "image":
                self.image_files.append(Path(result["path"]))
                self.images.append(result["image"])
                self.image_index = len(self.images) - 1
                self.image_zoom, self.image_pan = 1.0, [0.0, 0.0]
                self.mode = "images"
                self.notify(f"Image ready: {result['name']}", 5)
            elif result["kind"] == "presentation":
                slides = result["slides"]
                self.presentation = PresentationState(
                    path=Path(result["path"]), name=result["name"], slides=slides,
                    thumbnails=result["thumbnails"], annotations=[[] for _ in slides],
                )
                self.mode = "presentation"
                self.notify(f"Presentation ready: {len(slides)} slides. Pinch on a slide to annotate.", 6)

    def presentation_viewport(self) -> tuple[int, int, int, int]:
        """Return a normal editor viewport or the full board for presentation mode."""
        return (0, 0, BOARD_WIDTH, BOARD_HEIGHT) if self.presentation.active else PRESENTATION_RECT

    def presentation_slide_rect(self) -> tuple[int, int, int, int]:
        """Return the slide rectangle with aspect-ratio-preserving fit inside the viewport."""
        deck = self.presentation
        if not deck.slides:
            return self.presentation_viewport()
        slide = deck.slides[deck.index]
        return contained_rect((slide.shape[1], slide.shape[0]), self.presentation_viewport())

    def save_config(self) -> None:
        """Persist camera, pointer, remote, and calibration settings to JSON."""
        self.config.update({
            "camera_index": self.camera.index,
            "mirror": self.camera.mirror,
            "flip_vertical": self.camera.flip_vertical,
            "detection_mode": self.config.get("detection_mode", "hand"),
            "hsv_lower": self.config.get("hsv_lower", [88, 81, 43]),
            "hsv_upper": self.config.get("hsv_upper", [159, 255, 255]),
            "min_area": self.config.get("min_area", 45),
            "remote_left_keys": sorted(self.router.left_keys),
            "remote_right_keys": sorted(self.router.right_keys),
            "calibration": self.calibration.to_dict() if self.calibration else None,
        })
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    def set_pointer(self, point, source: str = "Camera") -> None:
        """Update pointer input and apply drawing, pan, or object-drag side effects."""
        self.previous_pointer = self.pointer
        self.pointer = point
        self.pointer_source = source
        if point is None:
            return
        if self.drawing:
            if self.mode == "presentation" and self.presentation.slides:
                rect = self.presentation_slide_rect()
                if inside(point, rect):
                    x, y, w, h = rect
                    normalized = ((point[0] - x) / w, (point[1] - y) / h)
                    target = self.presentation.annotations[self.presentation.index]
                    if target and (not target[-1].points or
                                   np.linalg.norm(np.array(normalized) - target[-1].points[-1]) > 0.0015):
                        target[-1].points.append(normalized)
            elif inside(point, CONTENT):
                target = self.slide_annotations[self.slide_index] if self.mode == "lesson" else self.strokes
                if target and (not target[-1].points or np.linalg.norm(np.array(point) - target[-1].points[-1]) > 1.2):
                    target[-1].points.append(point)
        if self.image_panning and self.previous_pointer is not None and self.image_zoom > 1:
            self.image_pan[0] -= point[0] - self.previous_pointer[0]
            self.image_pan[1] -= point[1] - self.previous_pointer[1]
        if self.grabbed_object is not None and inside(point, CONTENT):
            obj = self.objects[self.grabbed_object]
            obj.center = np.array([
                np.clip(point[0], CONTENT[0] + obj.size, CONTENT[0] + CONTENT[2] - obj.size),
                np.clip(point[1], CONTENT[1] + obj.size, CONTENT[1] + CONTENT[3] - obj.size),
            ])

    def buttons(self) -> list[Button]:
        """Return all currently clickable regions; no GUI toolkit is required."""
        # Presentation fullscreen uses a separate compact control set.
        if self.mode == "presentation" and self.presentation.active:
            return [
                Button("pen_next", "INK", (20, 18, 110, 40)),
                Button("clear_deck_slide", "CLEAR INK", (145, 18, 145, 40)),
                Button("presentation_exit", "EXIT  ESC", (1100, 18, 155, 40)),
                Button("deck_prev", "< PREV", (20, 645, 140, 46)),
                Button("deck_next", "NEXT >", (1120, 645, 140, 46)),
            ]
        buttons = [
            # These top-bar controls are available in every non-fullscreen mode.
            Button("home", "Home", (820, 15, 115, 42), self.mode == "home"),
            Button("reset_app", "CONFIRM RESET" if time.monotonic() < self.reset_armed_until else "Reset App", (947, 15, 135, 42)),
            Button("fullscreen", "Window" if self.fullscreen else "Fullscreen", (1094, 15, 160, 42)),
        ]
        if self.mode == "home":
            cards = [
                ("draw", "Draw", "Free ink, colors and eraser"),
                ("images", "Images", "Import, zoom and pan"),
                ("presentation", "PowerPoint", "Slides + separate live ink"),
                ("objects", "Objects", "Grab, move and transform"),
                ("settings", "Settings", "Camera, pointer and remote"),
                ("calibrate", "Calibrate", "Four-corner board mapping"),
            ]
            for index, (action, label, _) in enumerate(cards):
                column, row = index % 2, index // 2
                buttons.append(Button(action, label, (280 + column * 470, 165 + row * 140, 430, 112), accent="card"))
            return buttons
        if self.mode == "calibrate":
            buttons.extend([
                Button("settings", "Back to Settings", (25, 120, 205, 48)),
                Button("calibrate", "Restart Calibration", (25, 180, 205, 48)),
            ])
            return buttons
        if self.mode == "draw":
            items = [
                ("pen_red", "Red pen"), ("pen_blue", "Blue pen"),
                ("pen_green", "Green pen"), ("pen_black", "Black pen"),
                ("eraser", "Eraser"), ("thickness", f"Thickness: {self.pen_thickness}"),
                ("undo", "Undo stroke"), ("clear", "Clear canvas"),
                ("draw_mode", "Mode: Toggle" if self.draw_toggle_mode else "Mode: Hold"),
                ("debug", "Camera debug: " + ("On" if self.show_debug else "Off")),
            ]
            buttons.extend(Button(action, label, (25, 110 + i * 50, 200, 42),
                                  (action == "eraser" and self.eraser) or
                                  (action.startswith("pen_") and not self.eraser and self._action_color(action) == self.pen_color))
                           for i, (action, label) in enumerate(items))
        elif self.mode == "images":
            items = [("import_image", "IMPORTING..." if self.import_in_progress else "+ Import Image")]
            start = max(0, min(self.image_index - 1, len(self.image_files) - 4))
            items += [(f"image_{i}", self.image_files[i].stem.replace("_", " ").title()[:23])
                      for i in range(start, min(len(self.image_files), start + 4))]
            items += [("zoom_in", "Zoom +  (wheel)"), ("zoom_out", "Zoom -  (wheel)"),
                      ("zoom_2x", "2x close-up"), ("reset_view", "Fit / Reset View"),
                      ("pan_mode", "Pan: " + ("On" if self.image_panning else "Off")),
                      ("debug", "Camera debug: " + ("On" if self.show_debug else "Off"))]
            buttons.extend(Button(action, label, (25, 105 + i * 48, 200, 40),
                                  action == f"image_{self.image_index}") for i, (action, label) in enumerate(items))
            buttons.extend([Button("zoom_out", "-", (690, 595, 58, 48)),
                            Button("zoom_in", "+", (930, 595, 58, 48)),
                            Button("reset_view", "FIT", (1005, 595, 80, 48)),
                            Button("pan_mode", "PAN", (1100, 595, 90, 48), self.image_panning)])
        elif self.mode == "presentation":
            deck = self.presentation
            items = [("import_pptx", "IMPORTING..." if self.import_in_progress else "+ Import PPTX"),
                     ("deck_prev", "< Previous slide"), ("deck_next", "Next slide >"),
                     ("present", "PRESENT FULLSCREEN"), ("clear_deck_slide", "Clear this slide ink"),
                     ("pen_next", "Cycle ink color"), ("thumb_prev", "Thumbnails up"),
                     ("thumb_next", "Thumbnails down"),
                     ("debug", "Camera debug: " + ("On" if self.show_debug else "Off"))]
            buttons.extend(Button(action, label, (25, 105 + i * 49, 200, 41),
                                  action == "present" and deck.active)
                           for i, (action, label) in enumerate(items))
            if deck.slides:
                end = min(len(deck.slides), deck.thumbnail_offset + PRESENTATION_THUMBNAIL_COUNT)
                for slot, index in enumerate(range(deck.thumbnail_offset, end)):
                    buttons.append(Button(f"deck_page_{index}", str(index + 1),
                                          (250 + slot * 197, 568, 190, 90),
                                          index == deck.index, accent="thumbnail"))
        elif self.mode == "lesson":
            items = [("slide_prev", "Previous slide"), ("slide_next", "Next slide"),
                     ("clear_slide", "Clear slide ink"), ("debug", "Camera debug: " + ("On" if self.show_debug else "Off"))]
            buttons.extend(Button(action, label, (25, 110 + i * 52, 200, 44)) for i, (action, label) in enumerate(items))
        elif self.mode == "objects":
            items = [("add_circle", "+ Circle"), ("add_square", "+ Square"), ("add_diamond", "+ Diamond"),
                     ("object_red", "Color: Red"), ("object_blue", "Color: Blue"),
                     ("object_green", "Color: Green"), ("object_yellow", "Color: Yellow"),
                     ("object_purple", "Color: Purple"), ("object_size", "Resize selected"),
                     ("delete_object", "Delete selected"), ("reset_objects", "Reset all objects")]
            buttons.extend(Button(action, label, (25, 105 + i * 49, 200, 41),
                                  action == "delete_object" and self.selected_object is not None)
                           for i, (action, label) in enumerate(items))
        elif self.mode == "settings":
            tabs = [("camera", "Camera"), ("pointer", "Pointer"), ("remote", "Remote"), ("calibrate", "Calibration")]
            buttons.extend(Button("tab_" + action, label, (25, 110 + i * 52, 200, 44), self.settings_tab == action)
                           for i, (action, label) in enumerate(tabs))
            if self.settings_tab == "camera":
                status = self.camera.status()
                source = "Synthetic" if status.synthetic else f"Device {status.index}"
                buttons.extend([
                    Button("camera_cycle", f"Source: {source}", (280, 190, 330, 52)),
                    Button("mirror", f"Horizontal flip: {'On' if self.camera.mirror else 'Off'}", (280, 260, 330, 52), self.camera.mirror),
                    Button("flip_vertical", f"Vertical flip: {'On' if self.camera.flip_vertical else 'Off'}", (280, 330, 330, 52), self.camera.flip_vertical),
                ])
            elif self.settings_tab == "pointer":
                buttons.extend([
                    Button("mode_hand", "Hand + Fingers", (280, 120, 200, 44), self.config["detection_mode"] == "hand"),
                    Button("mode_marker", "Marker color", (495, 120, 190, 44), self.config["detection_mode"] == "marker"),
                    Button("sample", "Sampling: click pointer" if self.sample_mode else "Sample Color", (700, 120, 240, 44), self.sample_mode),
                    Button("auto_tune", "Auto Tune Marker", (955, 120, 270, 44)),
                    Button("debug", "Camera debug: " + ("On" if self.show_debug else "Off"), (25, 330, 200, 44), self.show_debug),
                ])
            elif self.settings_tab == "remote":
                buttons.extend([
                    Button("bind_left", "Rebind Left Click", (290, 360, 280, 54), self.router.capture_action == "left"),
                    Button("bind_right", "Rebind Right Click", (600, 360, 280, 54), self.router.capture_action == "right"),
                    Button("restore_remote", "Restore common defaults", (910, 360, 280, 54)),
                ])
            elif self.settings_tab == "calibrate":
                buttons.append(Button("calibrate", "Start Calibration", (280, 250, 330, 54)))
        return buttons

    @staticmethod
    def _action_color(action: str):
        """Convert a named pen button into its RGB display color."""
        return {"pen_red": (55, 70, 230), "pen_blue": (220, 125, 50),
                "pen_green": (65, 165, 90), "pen_black": (35, 40, 50)}.get(action)

    def _start_stroke(self) -> None:
        """Begin a new screen-space or normalized presentation-space stroke."""
        color = PEN_BACKGROUND if self.eraser else self.pen_color
        thickness = self.pen_thickness * (3 if self.eraser else 1)
        if self.mode == "presentation" and self.presentation.slides:
            rect = self.presentation_slide_rect()
            x, y, w, h = rect
            point = ((self.pointer[0] - x) / w, (self.pointer[1] - y) / h)
            target = self.presentation.annotations[self.presentation.index]
        else:
            point = self.pointer
            target = self.slide_annotations[self.slide_index] if self.mode == "lesson" else self.strokes
        target.append(Stroke([point], color, thickness))
        self.drawing = True

    def _stop_stroke(self) -> None:
        """End the current stroke without removing its recorded points."""
        self.drawing = False

    def left_release(self) -> None:
        """Stop held drawing mode when a left/release event is received."""
        if self.drawing and not self.draw_toggle_mode:
            self._stop_stroke()

    def left_click(self, point=None) -> None:
        """Handle left mouse or remote clicks, prioritizing controls over canvas input."""
        if point is not None:
            self.set_pointer(point, "Mouse")
        if self.pointer is None:
            return
        self.ripples.append((time.monotonic(), self.pointer[0], self.pointer[1], "left"))
        if self.mode == "calibrate":
            for button in self.buttons():
                if inside(self.pointer, button.rect):
                    self.activate(button.action)
                    return
            self._calibration_click()
            return
        if self.mode == "settings" and self.settings_tab == "pointer":
            if self.sample_mode and inside(self.pointer, SETTINGS_PREVIEW):
                self._sample_color()
                return
            if self._slider_click(self.pointer):
                return
        for button in self.buttons():
            if inside(self.pointer, button.rect):
                self._stop_all_motion()
                self.activate(button.action)
                return
        draw_region = self.presentation_slide_rect() if self.mode == "presentation" and self.presentation.slides else CONTENT
        if not inside(self.pointer, draw_region):
            return
        if self.mode in {"draw", "lesson", "presentation"}:
            if self.mode == "presentation" and not self.presentation.slides:
                self.notify("Import a .pptx presentation first.")
            elif self.draw_toggle_mode:
                if self.drawing:
                    self._stop_stroke()
                else:
                    self._start_stroke()
            elif not self.drawing:
                self._start_stroke()
        elif self.mode == "images":
            self.image_panning = not self.image_panning
            self.notify("Pan active — move pointer; click again to release." if self.image_panning else "Pan released.")
        elif self.mode == "objects":
            if self.grabbed_object is not None:
                self.grabbed_object = None
                self.notify("Object released.")
            else:
                for index in reversed(range(len(self.objects))):
                    if np.linalg.norm(self.objects[index].center - self.pointer) <= self.objects[index].size:
                        self.grabbed_object = index
                        self.selected_object = index
                        self.notify("Object selected and grabbed — move pointer; click to release.")
                        break

    def right_click(self, point=None) -> None:
        """Apply mode-specific right-click shortcuts and object cycling."""
        if point is not None:
            self.set_pointer(point, "Mouse")
        if self.pointer is not None:
            self.ripples.append((time.monotonic(), self.pointer[0], self.pointer[1], "right"))
        if self.mode == "draw":
            self.activate("undo")
        elif self.mode == "lesson":
            self.activate("slide_prev")
        elif self.mode == "presentation":
            self.activate("deck_prev")
        elif self.mode == "images":
            self.activate("reset_view")
        elif self.mode == "objects":
            for obj in self.objects:
                if self.pointer is not None and np.linalg.norm(obj.center - self.pointer) <= obj.size:
                    self._cycle_object(obj)
                    return
            self.notify("Right-click an object to cycle its color and size.")
        elif self.mode == "settings":
            self.mode = "home"
        else:
            self.notify("Right click: Previous / undo / reset, depending on mode.")

    def _stop_all_motion(self) -> None:
        """Cancel transient drawing, panning, and object-grab interactions."""
        self.drawing = False
        self.image_panning = False
        self.grabbed_object = None

    def _set_fullscreen(self, enabled: bool) -> None:
        self.fullscreen = enabled
        if self.fullscreen_callback:
            self.fullscreen_callback(enabled)

    def _exit_presentation(self) -> None:
        deck = self.presentation
        if not deck.active:
            return
        deck.active = False
        self.drawing = False
        self._set_fullscreen(deck.return_fullscreen)
        self.notify("Presentation closed — deck and every slide annotation are preserved.", 4)

    def _select_deck_slide(self, index: int) -> None:
        deck = self.presentation
        if not deck.slides:
            return
        deck.index = int(np.clip(index, 0, len(deck.slides) - 1))
        self.drawing = False
        if deck.index < deck.thumbnail_offset:
            deck.thumbnail_offset = deck.index
        elif deck.index >= deck.thumbnail_offset + PRESENTATION_THUMBNAIL_COUNT:
            deck.thumbnail_offset = deck.index - PRESENTATION_THUMBNAIL_COUNT + 1
        deck.thumbnail_offset = int(np.clip(deck.thumbnail_offset, 0,
                                            max(0, len(deck.slides) - PRESENTATION_THUMBNAIL_COUNT)))

    def activate(self, action: str) -> None:
        """Dispatch a UI button action to navigation, settings, or workspace behavior."""
        # Primary workspace navigation.
        if action in {"draw", "images", "presentation", "lesson", "objects", "settings"}:
            if self.presentation.active and action != "presentation":
                self._exit_presentation()
            self.mode = action
            self._stop_all_motion()
            if action == "settings":
                self.sample_mode = False
            return
        if action == "home":
            self._exit_presentation()
            self.mode = "home"
            self._stop_all_motion()
            return
        if action == "reset_app":
            self.reset_app()
            return

        # Global controls and settings tabs.
        if action == "fullscreen":
            self._set_fullscreen(not self.fullscreen)
            return
        if action == "calibrate":
            self._exit_presentation()
            self.start_calibration()
            return
        if action.startswith("tab_"):
            self.settings_tab = action[4:]
            self.sample_mode = False
            return
        if action == "camera_cycle":
            self.camera.cycle()
            self.calibration = None
            self.save_config()
            self.notify(self.camera.last_error or f"Camera switched to {self.camera.index}; please recalibrate.", 5)
        elif action == "mirror":
            self.camera.mirror = not self.camera.mirror
            self.calibration = None
            self.save_config()
            self.notify("Horizontal flip changed; please recalibrate.")
        elif action == "flip_vertical":
            self.camera.flip_vertical = not self.camera.flip_vertical
            self.calibration = None
            self.save_config()
            self.notify("Vertical flip changed; please recalibrate.")

        # Pointer detection and marker settings.
        elif action in {"mode_hand", "mode_marker"}:
            self.config["detection_mode"] = "hand" if action == "mode_hand" else "marker"
            self.save_config()
            self.notify("Detection mode: " + self.config["detection_mode"])
        elif action == "sample":
            self.sample_mode = not self.sample_mode
            self.notify("Click the colored pointer in the camera preview." if self.sample_mode else "Color sampling cancelled.")
        elif action == "auto_tune":
            frame = getattr(self, "current_frame", None)
            result = auto_sample_marker(frame) if frame is not None else None
            if result is None:
                self.notify("Auto Tune could not find a compact colored marker. Try Sample Color instead.", 5)
            else:
                lower, upper, detection = result
                self.config["hsv_lower"], self.config["hsv_upper"] = lower, upper
                self.config["detection_mode"] = "marker"
                self.sample_mode = False
                self.save_config()
                self.notify(f"Marker auto-tuned at {tuple(round(v) for v in detection.centroid)}: HSV {lower} to {upper}.", 6)
        elif action == "debug":
            self.show_debug = not self.show_debug

        # Presentation remote binding settings.
        elif action in {"bind_left", "bind_right"}:
            self.router.begin_capture("left" if action == "bind_left" else "right")
            self.notify("Press the desired remote button now…", 10)
        elif action == "restore_remote":
            self.router.left_keys = {"page_down", "right", "down"}
            self.router.right_keys = {"page_up", "left", "up"}
            self.router.capture_action = None
            self.save_config()
            self.notify("Common presentation-remote bindings restored.")

        # Native-file imports are queued and completed by the background worker.
        elif action in {"import_image", "import_pptx"}:
            if self.importer is None:
                self.notify("Import service is unavailable in this test context.")
            elif self.import_in_progress or self.importer.busy:
                self.notify("An import is already running. The camera and pointer remain active.")
            else:
                self.import_request = action
                self.import_in_progress = True
                self.notify("Choose a file. Conversion will run in the background.", 5)
        elif action == "present":
            if not self.presentation.slides:
                self.notify("Import a .pptx presentation first.")
            else:
                deck = self.presentation
                deck.return_fullscreen = self.fullscreen
                deck.active = True
                self.drawing = False
                self._set_fullscreen(True)
                self.notify("Presenting — pinch to draw, remote Previous/Next to navigate, Escape to exit.", 5)

        # Presentation navigation and slide-ink controls.
        elif action == "presentation_exit":
            self._exit_presentation()
        elif action in {"deck_prev", "deck_next"}:
            if self.presentation.slides:
                step = -1 if action == "deck_prev" else 1
                self._select_deck_slide(self.presentation.index + step)
        elif action.startswith("deck_page_"):
            self._select_deck_slide(int(action.rsplit("_", 1)[1]))
        elif action in {"thumb_prev", "thumb_next"}:
            deck = self.presentation
            maximum = max(0, len(deck.slides) - PRESENTATION_THUMBNAIL_COUNT)
            step = -1 if action == "thumb_prev" else 1
            deck.thumbnail_offset = int(np.clip(deck.thumbnail_offset + step * PRESENTATION_THUMBNAIL_COUNT, 0, maximum))
        elif action == "clear_deck_slide":
            if self.presentation.slides:
                self.presentation.annotations[self.presentation.index].clear()
                self.drawing = False
                self.notify("Ink cleared from this slide only.")

        # Freehand pen controls used across draw and presentation modes.
        elif action == "pen_next":
            colors = [(55, 70, 230), (220, 125, 50), (65, 165, 90), (35, 40, 50), (190, 70, 170)]
            index = colors.index(self.pen_color) if self.pen_color in colors else -1
            self.pen_color = colors[(index + 1) % len(colors)]
            self.eraser = False
            self.notify("Ink color changed — see the current-ink preview.")
        elif action.startswith("pen_"):
            self.pen_color = self._action_color(action)
            self.eraser = False
        elif action == "eraser":
            self.eraser = not self.eraser
        elif action == "thickness":
            self.pen_thickness = {3: 5, 5: 9, 9: 3}.get(self.pen_thickness, 3)
        elif action == "draw_mode":
            self.draw_toggle_mode = not self.draw_toggle_mode
            self.drawing = False
        elif action == "undo":
            if self.mode == "presentation" and self.presentation.slides:
                target = self.presentation.annotations[self.presentation.index]
            else:
                target = self.slide_annotations[self.slide_index] if self.mode == "lesson" else self.strokes
            if target:
                target.pop()
            self.drawing = False
            self.notify("Last stroke removed.")
        elif action == "clear":
            self.strokes.clear()
            self.drawing = False

        # Image selection, zoom, and pan controls.
        elif action.startswith("image_"):
            index = int(action.split("_")[1])
            if 0 <= index < len(self.images):
                self.image_index = index
                self.image_zoom, self.image_pan = 1.0, [0.0, 0.0]
        elif action == "zoom_in":
            self.image_zoom = min(6.0, self.image_zoom + 0.25)
        elif action == "zoom_out":
            self.image_zoom = max(1.0, self.image_zoom - 0.25)
            if self.image_zoom == 1.0:
                self.image_pan = [0.0, 0.0]
        elif action == "zoom_2x":
            self.image_zoom = 2.0
        elif action == "pan_mode":
            self.image_panning = not self.image_panning
        elif action == "reset_view":
            self.image_zoom, self.image_pan, self.image_panning = 1.0, [0.0, 0.0], False

        # Built-in lesson navigation and per-slide ink.
        elif action == "slide_next":
            self.slide_index = (self.slide_index + 1) % len(self.slides)
            self.drawing = False
        elif action == "slide_prev":
            self.slide_index = (self.slide_index - 1) % len(self.slides)
            self.drawing = False
        elif action == "clear_slide":
            self.slide_annotations[self.slide_index].clear()
            self.drawing = False

        # Object playground creation, editing, and reset controls.
        elif action.startswith("add_") and len(self.objects) < 24:
            kind = action.split("_")[1]
            index = len(self.objects)
            center = np.array([520.0 + (index % 4) * 130, 250.0 + (index % 3) * 100])
            self.objects.append(BoardObject(center, 52, self.object_color, kind, tuple(center)))
            self.selected_object = len(self.objects) - 1
            self.notify(f"Added {kind}. Drag it anywhere; choose a color or resize it.")
        elif action.startswith("object_"):
            if action == "object_size" and self.selected_object is not None:
                obj = self.objects[self.selected_object]
                obj.size = 36 if obj.size >= 84 else obj.size + 12
            elif action == "object_red" or action == "object_blue" or action == "object_green" or action == "object_yellow" or action == "object_purple":
                colors = {"object_red": (70, 70, 230), "object_blue": (220, 130, 60),
                        "object_green": (80, 175, 95), "object_yellow": (55, 195, 240), "object_purple": (180, 90, 170)}
                self.object_color = colors[action]
                if self.selected_object is not None:
                    self.objects[self.selected_object].color = self.object_color
        elif action == "delete_object" and self.selected_object is not None:
            del self.objects[self.selected_object]
            self.selected_object = None
            self.grabbed_object = None
        elif action == "reset_objects":
            self.objects = [
                BoardObject(np.array([430.0, 260.0]), 54, (60, 105, 225), "circle", (430, 260)),
                BoardObject(np.array([690.0, 365.0]), 65, (65, 165, 100), "square", (690, 365)),
                BoardObject(np.array([960.0, 245.0]), 62, (190, 105, 55), "diamond", (960, 245)),
            ]
            self.grabbed_object = None
            self.selected_object = 0

    @staticmethod
    def _cycle_object(obj: BoardObject) -> None:
        colors = [(60, 105, 225), (65, 165, 100), (190, 105, 55), (150, 70, 180)]
        obj.color = colors[(colors.index(obj.color) + 1) % len(colors)] if obj.color in colors else colors[0]
        obj.size = 45 if obj.size >= 70 else obj.size + 12

    def reset_app(self) -> None:
        """Two-click reset: first arm the operation, then restore the complete factory state."""
        if time.monotonic() >= self.reset_armed_until:
            self.reset_armed_until = time.monotonic() + 4
            self.notify("Click CONFIRM RESET within 4 seconds to clear every workspace and restore preferences.", 4)
            return
        self._exit_presentation()
        self._stop_all_motion()
        self.reset_armed_until = 0.0

        # Cancel pending imports and discard imported presentations/images.
        self.import_generation += 1
        self.import_in_progress = False
        self.import_request = None
        self.presentation = PresentationState()
        self.image_files = self._original_image_files.copy()
        self.images = self._original_images.copy()
        self.image_index = 0

        # Restore all workspace annotations and drawing preferences.
        self.strokes.clear()
        self.slide_annotations = [[] for _ in self.slides]
        self.slide_index = 0
        self.image_zoom, self.image_pan, self.image_panning = 1.0, [0.0, 0.0], False
        self.pen_color, self.pen_thickness, self.eraser = (55, 70, 230), 5, False
        self.draw_toggle_mode = True
        self.show_debug = False
        self.sample_mode = False
        self.settings_tab = "camera"
        self.calibration_points = []
        self.calibration = None
        self.calibration_message = "Click the projected board corners in order."
        self.mapping_status = "Approximate — run Calibration"
        self.detection_area = 0
        self.tracking_status = "SEARCHING FOR POINTER"
        self.object_color = (60, 105, 225)

        # Reset pointer and camera/view settings.
        self.pointer = (640, 360)
        self.previous_pointer = self.pointer
        self.pointer_source = "Mouse"
        self.camera.open(-1)
        self.camera.mirror = False
        self.camera.flip_vertical = False

        # Restore default pointer detection and presentation-remote bindings.
        self.config["detection_mode"] = "hand"
        self.config["hsv_lower"] = [88, 81, 43]
        self.config["hsv_upper"] = [159, 255, 255]
        self.config["min_area"] = 45
        self.router.left_keys, self.router.right_keys = {"page_down", "right", "down"}, {"page_up", "left", "up"}
        self.router.capture_action = None
        self.activate("reset_objects")
        self._set_fullscreen(False)
        self.save_config()
        self.reset_requested = True
        self.mode = "home"
        self.notify("FULL RESET COMPLETE — camera, calibration, pointer, remote, workspace and preferences restored.", 6)

    def start_calibration(self) -> None:
        """Enter calibration mode and clear any previous four-corner selection."""
        self.mode = "calibrate"
        self._stop_all_motion()
        self.calibration_points = []
        self.calibration = None
        self.calibration_message = "Click Top-Left corner (1/4)."
        self.notify("Calibration started: click TL, TR, BR, BL.", 5)

    def _camera_coordinates(self, point, rect) -> tuple[float, float]:
        """Convert a board-space UI point into native live-camera pixel coordinates."""
        height, width = self.frame_shape[:2]
        x, y = point
        rx, ry, rw, rh = rect
        return ((x - rx) * width / rw, (y - ry) * height / rh)

    def _calibration_click(self) -> None:
        """Collect the board corners, auto-order them, and calculate the homography."""
        if not inside(self.pointer, CAMERA_RECT):
            self.notify("Click inside the live camera image.")
            return
        self.calibration_points.append(self._camera_coordinates(self.pointer, CAMERA_RECT))
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        count = len(self.calibration_points)
        if count < 4:
            self.calibration_message = f"Click {labels[count]} corner ({count + 1}/4)."
            return
        try:
            height, width = self.frame_shape[:2]
            # Correct accidental click-order mistakes automatically. In image
            # coordinates TL minimizes x+y, BR maximizes it, TR maximizes x-y,
            # and BL minimizes x-y. This prevents a twisted bow-tie homography.
            points = np.asarray(self.calibration_points, dtype=float)
            order = [int(np.argmin(points.sum(axis=1))), int(np.argmax((points * [1, -1]).sum(axis=1))),
                     int(np.argmax(points.sum(axis=1))), int(np.argmin((points * [1, -1]).sum(axis=1)))]
            if len(set(order)) != 4:
                raise ValueError("corner order is ambiguous; click four outer corners")
            self.calibration_points = points[order].tolist()
            self.calibration = Calibration.from_points(
                self.calibration_points, (BOARD_WIDTH, BOARD_HEIGHT), (width, height)
            )
            self.calibration_message = f"Corners auto-ordered. Calibration saved — RMSE: {self.calibration.rmse:.6f} px"
            self.save_config()
            self.notify("Calibration complete. Verification grid is shown on the camera image.", 6)
        except ValueError as exc:
            self.calibration = None
            self.calibration_points = []
            self.calibration_message = f"Invalid calibration: {exc}. Restart and use four distinct corners."

    def _sample_color(self) -> None:
        """Sample HSV bounds from the current pointer location in the settings preview."""
        frame = getattr(self, "current_frame", None)
        if frame is None:
            self.notify("No camera frame available.")
            return
        x, y = self._camera_coordinates(self.pointer, SETTINGS_PREVIEW)
        lower, upper = sample_hsv_bounds(frame, int(x), int(y))
        self.config["hsv_lower"], self.config["hsv_upper"] = lower, upper
        self.config["detection_mode"] = "marker"
        self.sample_mode = False
        self.save_config()
        self.notify(f"Pointer color sampled: HSV {lower} to {upper}.", 5)

    @staticmethod
    def slider_specs() -> list[tuple[str, str, int, int, int, int]]:
        # key, label, x/y row index are interpreted by renderer and hit testing
        return [
            ("h_low", "Hue min", 0, 179, 300, 575),
            ("h_high", "Hue max", 0, 179, 680, 575),
            ("s_min", "Saturation min", 0, 255, 300, 630),
            ("v_min", "Value min", 0, 255, 680, 630),
            ("area", "Minimum area", 5, 600, 1070, 575),
        ]

    def _slider_click(self, point) -> bool:
        for key, _, low, high, x, y in self.slider_specs():
            width = 260 if key != "area" else 160
            if inside(point, (x - 5, y - 15, width + 10, 35)):
                ratio = float(np.clip((point[0] - x) / width, 0, 1))
                value = int(round(low + ratio * (high - low)))
                if key == "h_low":
                    self.config["hsv_lower"][0] = value
                elif key == "h_high":
                    self.config["hsv_upper"][0] = value
                elif key == "s_min":
                    self.config["hsv_lower"][1] = value
                elif key == "v_min":
                    self.config["hsv_lower"][2] = value
                elif key == "area":
                    self.config["min_area"] = value
                self.save_config()
                return True
        return False
