"""High-contrast projected interface rendered into a NumPy framebuffer."""
from __future__ import annotations

import time
import cv2
import numpy as np

from core.calibration import inverse_project
from ui.primitives import circle, crosshair, fit_contain, line, rectangle, resize_bilinear, zoom_pan
from ui.interaction import BOARD_HEIGHT, BOARD_WIDTH, CAMERA_RECT, CONTENT, PEN_BACKGROUND, SETTINGS_PREVIEW, Button, inside

# Pixel-art study-room palette: ink navy, notebook cream, cyan and coral.
INK = (64, 32, 37)
MUTED = (123, 91, 102)
NAVY = (42, 18, 20)
BLUE = (255, 102, 91)
TEAL = (185, 214, 60)
PINK = (138, 82, 240)
YELLOW = (61, 197, 255)
WHITE = (224, 247, 255)
PANEL = (235, 250, 255)
BACKGROUND = (222, 239, 245)
BORDER = (202, 174, 185)


def put_text(image, value, point, scale=0.55, color=INK, thickness=1, center=False):
    """Draw one OpenCV text line, with optional center alignment."""
    if center:
        (width, _), _ = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        point = (int(point[0] - width / 2), point[1])
    cv2.putText(image, str(value), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_8)


def wrapped_text(image, value, x, y, max_width=850, scale=0.62, color=INK, line_height=29):
    """Wrap text by measured OpenCV width and draw each resulting line."""
    words, lines, current = value.split(), [], ""
    for word in words:
        trial = (current + " " + word).strip()
        width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        if width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    for index, value_line in enumerate(lines):
        put_text(image, value_line, (x, y + index * line_height), scale, color)


class Renderer:
    """Convert InteractionState and live camera data into one 1280x720 image."""

    def render(self, state, camera_frame, mask, detection, remote) -> np.ndarray:
        """Dispatch to the active screen renderer and draw the shared overlays."""
        image = np.full((BOARD_HEIGHT, BOARD_WIDTH, 3), BACKGROUND, dtype=np.uint8)
        if state.mode == "presentation" and state.presentation.active:
            self._presentation(image, state)
            for button in state.buttons():
                self._button(image, button, state.pointer, state)
            self._ripples(image, state)
            crosshair(image, state.pointer, ink_color=state.pen_color if not state.eraser else None)
            return image
        # Notebook graph-paper backdrop and chunky 8-bit study-room header.
        for gy in range(92, 675, 24):
            for gx in range(250, 1270, 24):
                image[gy, gx] = (205, 225, 234)
        rectangle(image, (0, 0), (1279, 70), NAVY)
        rectangle(image, (0, 68), (1279, 73), TEAL)
        # Pixel graduation cap icon.
        rectangle(image, (17, 26), (55, 33), TEAL)
        rectangle(image, (25, 20), (47, 26), TEAL)
        rectangle(image, (28, 34), (45, 43), YELLOW)
        rectangle(image, (52, 34), (56, 48), PINK)
        put_text(image, "STUDY BOARD.EXE", (70, 39), 0.69, WHITE, 2)
        put_text(image, "HAND TRACKING  |  LEARN  |  CREATE", (70, 57), 0.34, (181, 157, 151))
        locked = getattr(state, "tracking_status", "") == "POINTER LOCKED"
        rectangle(image, (568, 27), (578, 38), TEAL if locked else PINK)
        put_text(image, getattr(state, "hand_gesture", "READY")[:20], (588, 40), 0.39, WHITE)

        if state.mode == "home":
            self._home(image, state)
        elif state.mode == "calibrate":
            self._calibration(image, state, camera_frame)
        else:
            self._page(image, state)
        for button in state.buttons():
            self._button(image, button, state.pointer, state)
        if state.show_debug and state.mode not in {"calibrate", "home"} and camera_frame is not None:
            self._debug_pip(image, camera_frame, mask, detection)
        self._ripples(image, state)
        cursor_ink = state.pen_color if state.mode in {"draw", "presentation"} and not state.eraser else None
        crosshair(image, state.pointer, ink_color=cursor_ink)
        rectangle(image, (0, 679), (1279, 719), (235, 250, 255))
        rectangle(image, (0, 679), (1279, 682), NAVY)
        status = state.status if time.monotonic() < state.status_until else \
            f"{state.mode.upper()}  |  {state.pointer_source}  |  {getattr(state, 'mapping_status', '')}"
        put_text(image, status[:145], (22, 704), 0.47, INK)
        camera_status = state.camera.status()
        put_text(image, "SYNTHETIC" if camera_status.synthetic else f"CAM {camera_status.index}", (1145, 704), 0.45, MUTED)
        return image

    def _button(self, image, button: Button, pointer, state) -> None:
        """Render one clickable control and highlight it when hovered or active."""
        x, y, w, h = button.rect
        hovered = inside(pointer, button.rect)
        if button.accent == "thumbnail":
            index = int(button.action.rsplit("_", 1)[1])
            rectangle(image, (x + 3, y + 4), (x + w + 3, y + h + 4), NAVY)
            rectangle(image, (x, y), (x + w, y + h), PANEL)
            image[y + 3:y + 85, x + 3:x + 187] = state.presentation.thumbnails[index]
            rectangle(image, (x, y), (x + w, y + h), TEAL if button.active else BORDER, 3 if button.active else 2)
            rectangle(image, (x + w - 34, y + h - 23), (x + w - 4, y + h - 4), NAVY)
            put_text(image, button.label, (x + w - 19, y + h - 8), 0.40, WHITE, center=True)
            return
        if button.accent == "card":
            accent = {"draw": PINK, "images": TEAL, "presentation": YELLOW, "lesson": BLUE,
                      "objects": (180, 90, 180), "settings": YELLOW, "calibrate": PINK}.get(button.action, BLUE)
            rectangle(image, (x + 5, y + 6), (x + w + 5, y + h + 6), NAVY)
            rectangle(image, (x, y), (x + w, y + h), WHITE if not hovered else (210, 248, 245))
            rectangle(image, (x, y), (x + w, y + h), accent, 3)
            rectangle(image, (x + 18, y + 15), (x + 54, y + 51), accent)
            # Blocky pencil/book/shape icons.
            if button.action in {"images", "lesson", "presentation"}:
                rectangle(image, (x + 26, y + 23), (x + 47, y + 43), WHITE, 2)
                rectangle(image, (x + 32, y + 29), (x + 41, y + 34), WHITE)
            elif button.action == "objects" or button.action == "settings":
                rectangle(image, (x + 26, y + 23), (x + 46, y + 43), WHITE, 3)
            else:
                line(image, (x + 26, y + 43), (x + 46, y + 23), WHITE, 5, False)
            put_text(image, button.label.upper(), (x + 70, y + 41), 0.62, NAVY, 2)
            subtitles = {"draw": "PIXEL PENS + ERASER", "images": "IMPORT / ZOOM / PAN",
                         "presentation": "PPTX + PER-SLIDE INK", "lesson": "SLIDES + LIVE NOTES",
                         "objects": "BUILD YOUR OWN SHAPES", "settings": "HAND / CAMERA / REMOTE",
                         "calibrate": "FOUR-CORNER MAPPING"}
            put_text(image, subtitles.get(button.action, ""), (x + 20, y + 70), 0.39, MUTED)
            put_text(image, "START >", (x + w - 91, y + 90), 0.38, accent, 2)
            return
        fill = (215, 242, 238) if hovered else ((193, 235, 219) if button.active else PANEL)
        rectangle(image, (x + 3, y + 4), (x + w + 3, y + h + 4), NAVY)
        rectangle(image, (x, y), (x + w, y + h), fill)
        rectangle(image, (x, y), (x + w, y + h), TEAL if button.active else BORDER, 2)
        swatch = state._action_color(button.action)
        swatch = {"object_red": (70, 70, 230), "object_blue": (220, 130, 60),
                "object_green": (80, 175, 95), "object_yellow": (55, 195, 240),
                "object_purple": (180, 90, 170)}.get(button.action, swatch)
        if button.action == "pen_next":
            swatch = state.pen_color
        if swatch is not None:
            rectangle(image, (x + 10, y + 9), (x + 32, y + h - 10), NAVY)
            rectangle(image, (x + 13, y + 12), (x + 29, y + h - 13), swatch)
            put_text(image, button.label, (x + 42, y + h / 2 + 6), 0.42, NAVY)
        else:
            put_text(image, button.label, (x + w / 2, y + h / 2 + 6), 0.45, NAVY, 1, center=True)

    def _home(self, image, state) -> None:
        """Render the home card navigation screen."""
        put_text(image, "Interactive teaching, controlled from the board.", (280, 120), 0.75, NAVY, 2)
        put_text(image, "Choose a workspace. All configuration is available inside the application.", (280, 147), 0.49, MUTED)
        calibration = "CALIBRATED" if state.calibration else "CALIBRATION NEEDED"
        put_text(image, calibration, (1030, 120), 0.47, (50, 145, 90) if state.calibration else (50, 90, 190), 2)

    def _page(self, image, state) -> None:
        """Render the shared page header and current status bar."""
        titles = {"draw": "Free Drawing", "images": "Image Viewer", "presentation": "Lesson Presentation",
                "lesson": "Interactive Mini-Lesson", "objects": "Object Playground", "settings": "Settings"}
        put_text(image, titles.get(state.mode, state.mode.title()), (25, 93), 0.62, NAVY, 2)
        if state.mode == "draw":
            self._draw(image, state)
        elif state.mode == "images":
            self._images(image, state)
        elif state.mode == "presentation":
            self._presentation(image, state)
        elif state.mode == "lesson":
            self._lesson(image, state)
        elif state.mode == "objects":
            self._objects(image, state)
        elif state.mode == "settings":
            self._settings(image, state)

    def _canvas(self, image, color=PEN_BACKGROUND) -> None:
        """Fill the main workspace and draw its fixed content boundary."""
        x, y, w, h = CONTENT
        rectangle(image, (x, y), (x + w, y + h), color)
        rectangle(image, (x, y), (x + w, y + h), (198, 208, 218), 2)

    def _strokes(self, image, strokes) -> None:
        """Render normalized or board-space recorded ink strokes."""
        for stroke in strokes:
            if len(stroke.points) == 1:
                circle(image, stroke.points[0], max(1, stroke.thickness // 2), stroke.color, -1)
            else:
                for start, end in zip(stroke.points, stroke.points[1:]):
                    if inside(start, CONTENT) and inside(end, CONTENT):
                        line(image, start, end, stroke.color, stroke.thickness)

    def _draw(self, image, state) -> None:
        """Render the freehand drawing workspace."""
        self._canvas(image)
        self._strokes(image, state.strokes)
        if not state.strokes:
            put_text(image, "Click in the canvas to start ink. Move the pointer. Click again to stop.", (430, 365), 0.60, MUTED)
            put_text(image, "Right-click = undo  |  Hold mode is available in the left menu", (500, 400), 0.5, MUTED)
        if state.drawing:
            put_text(image, "INK ACTIVE", (1040, 183), 0.48, (50, 150, 90), 2)

    def _images(self, image, state) -> None:
        """Render the selected imported image with zoom, pan, and fit controls."""
        self._canvas(image, (250, 250, 250))
        if not state.images or state.images[state.image_index] is None:
            put_text(image, "Choose + Import Image to load a PNG, JPG or JPEG.", (430, 360), 0.65, MUTED)
            return
        x, y, w, h = CONTENT
        view = zoom_pan(state.images[state.image_index], (w, h), state.image_zoom, state.image_pan)
        image[y:y + h, x:x + w] = view
        rectangle(image, (x, y), (x + w, y + h), (198, 208, 218), 2)
        rectangle(image, (758, 595), (918, 643), PANEL)
        rectangle(image, (758, 595), (918, 643), NAVY, 2)
        put_text(image, f"{state.image_zoom:.2f}x", (775, 627), 0.60, NAVY, 2)
        put_text(image, "WHEEL", (844, 624), 0.34, MUTED)

    def _presentation(self, image, state) -> None:
        """Render the active or editor PowerPoint slide and live annotations."""
        deck = state.presentation
        if not deck.slides:
            self._canvas(image, (245, 247, 250))
            put_text(image, "IMPORT YOUR POWERPOINT FILE", (480, 275), 0.85, NAVY, 2)
            put_text(image, "Choose + Import PPTX. LibreOffice converts it in the background.", (390, 325), 0.57, MUTED)
            put_text(image, "Then use thumbnails, Previous / Next, Present and live pinch ink.", (405, 365), 0.55, MUTED)
            put_text(image, "Your camera tracking stays responsive during conversion.", (440, 405), 0.50, TEAL)
            return
        x, y, w, h = state.presentation_viewport()
        image[y:y + h, x:x + w] = fit_contain(deck.slides[deck.index], (w, h), (22, 25, 34))
        sx, sy, sw, sh = state.presentation_slide_rect()
        rectangle(image, (sx, sy), (sx + sw, sy + sh), NAVY, 2)
        self._presentation_strokes(image, state)
        if deck.active:
            put_text(image, f"SLIDE {deck.index + 1} / {len(deck.slides)}", (640, 684), 0.48, WHITE, 2, center=True)
        else:
            put_text(image, deck.name[:60], (700, 93), 0.43, MUTED)
            put_text(image, f"SLIDE {deck.index + 1} / {len(deck.slides)}   |   SELECT A THUMBNAIL   |   PINCH TO ANNOTATE",
                     (742, 563), 0.38, NAVY, 1, center=True)

    @staticmethod
    def _presentation_strokes(image, state) -> None:
        """Render normalized per-slide presentation annotations."""
        x, y, w, h = state.presentation_slide_rect()
        for stroke in state.presentation.annotations[state.presentation.index]:
            points = [(x + px * w, y + py * h) for px, py in stroke.points]
            if len(points) == 1:
                circle(image, points[0], max(1, stroke.thickness // 2), stroke.color, -1)
            else:
                for start, end in zip(points, points[1:]):
                    line(image, start, end, stroke.color, stroke.thickness)

    def _lesson(self, image, state) -> None:
        """Render the built-in lesson slide and its separate annotation layer."""
        self._canvas(image)
        slide = state.slides[state.slide_index]
        x, y, w, h = CONTENT
        put_text(image, f"{state.slide_index + 1} / {len(state.slides)}", (x + w - 95, y + 38), 0.52, MUTED)
        put_text(image, slide["title"], (x + 42, y + 60), 0.92, NAVY, 2)
        put_text(image, slide["subtitle"], (x + 42, y + 94), 0.53, BLUE)
        for index, body in enumerate(slide["body"]):
            circle(image, (x + 52, y + 145 + index * 53), 5, TEAL, -1)
            put_text(image, body, (x + 70, y + 152 + index * 53), 0.55, INK)
        self._lesson_diagram(image, slide["diagram"], x + 600, y + 150)
        put_text(image, "Annotate anywhere on this slide; ink is kept separately for each page.",
                 (x + 42, y + h - 38), 0.5, MUTED)
        self._strokes(image, state.slide_annotations[state.slide_index])

    def _lesson_diagram(self, image, kind, x, y) -> None:
        """Render the small teaching diagram associated with a lesson slide."""
        if kind == "homography":
            points = [(x, y), (x + 270, y + 15), (x + 245, y + 190), (x + 20, y + 175)]
            for i in range(4):
                line(image, points[i], points[(i + 1) % 4], BLUE, 3)
                circle(image, points[i], 8, TEAL, -1)
            put_text(image, "H", (x + 120, y + 100), 1.1, NAVY, 2)
        else:
            labels = {"pipeline": ["CAMERA", "HSV", "BOARD"], "blob": ["MASK", "BLOB", "POINT"],
                      "interaction": ["POINT", "CLICK", "ACTION"]}.get(kind, ["CAMERA", "MAP", "DRAW"])
            for index, label in enumerate(labels):
                cx = x + index * 120
                circle(image, (cx, y + 75), 38, (220, 230, 238), -1)
                circle(image, (cx, y + 75), 38, BLUE, 2)
                put_text(image, label, (cx, y + 140), 0.42, NAVY, center=True)
                if index < 2:
                    line(image, (cx + 42, y + 75), (cx + 78, y + 75), TEAL, 4)

    def _objects(self, image, state) -> None:
        """Render shapes, selection handles, grab handles, and object names."""
        self._canvas(image)
        put_text(image, "Left-click: grab / release    Right-click: cycle color + size", (420, 130), 0.53, MUTED)
        for index, obj in enumerate(state.objects):
            x, y = obj.center
            if obj.kind == "circle":
                circle(image, (x, y), obj.size, obj.color, -1)
            elif obj.kind == "square":
                rectangle(image, (int(x - obj.size), int(y - obj.size)),
                          (int(x + obj.size), int(y + obj.size)), obj.color)
            else:
                points = [(x, y - obj.size), (x + obj.size, y), (x, y + obj.size), (x - obj.size, y)]
                for i in range(4):
                    line(image, points[i], points[(i + 1) % 4], obj.color, obj.size // 2)
            if index == state.selected_object:
                rectangle(image, (int(x - obj.size - 8), int(y - obj.size - 8)),
                          (int(x + obj.size + 8), int(y + obj.size + 8)), NAVY, 2)
                put_text(image, "SELECTED", (x - 36, y - obj.size - 16), 0.38, NAVY, 2)
            if index == state.grabbed_object:
                for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                    rectangle(image, (int(x + dx * obj.size - 6), int(y + dy * obj.size - 6)),
                              (int(x + dx * obj.size + 6), int(y + dy * obj.size + 6)), PINK)

    def _settings(self, image, state) -> None:
        """Dispatch rendering for the selected settings tab."""
        self._canvas(image)
        if state.settings_tab == "camera":
            put_text(image, "Camera source and orientation", (280, 145), 0.72, NAVY, 2)
            put_text(image, "Cycle Synthetic -> 0 -> 1 -> 2. Switching requires recalibration.", (280, 420), 0.52, MUTED)
            wrapped_text(image, "Synthetic mode is a complete offline demonstration: it generates a projected board and a moving neon-green pointer. Use it to verify mapping and every interface feature without hardware.",
                         280, 470, 830, 0.52, MUTED)
            if state.camera.last_error:
                wrapped_text(image, state.camera.last_error, 280, 555, 830, 0.5, (50, 90, 190))
        elif state.settings_tab == "pointer":
            self._pointer_settings(image, state)
        elif state.settings_tab == "remote":
            self._remote_settings(image, state)
        elif state.settings_tab == "calibrate":
            put_text(image, "Board calibration", (280, 145), 0.72, NAVY, 2)
            wrapped_text(
                image,
                "Calibration maps the camera image to the projected board. "
                "You will click the four board corners in this order: top-left, "
                "top-right, bottom-right, bottom-left.",
                280,
                185,
                830,
                0.53,
                MUTED,
                28,
            )
            if state.calibration:
                put_text(image, "Status: calibrated", (280, 330), 0.58, (45, 140, 85), 2)
                put_text(image, f"RMSE: {state.calibration.rmse:.6f} px", (280, 370), 0.52, INK)
            else:
                put_text(image, "Status: calibration needed", (280, 330), 0.58, (50, 90, 190), 2)

    @staticmethod
    def _draw_hand(image, detection, frame_shape, rect) -> None:
        """Draw MediaPipe 21-point landmarks inside the settings camera preview."""
        if detection is None or not getattr(detection, "is_hand", False):
            return
        x, y, w, h = rect
        height, width = frame_shape[:2]
        points = np.asarray(detection.landmarks) * [w / width, h / height] + [x, y]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                     (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                     (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]:
            line(image, points[a], points[b], TEAL, 2, False)
        for index, point in enumerate(points):
            size = 5 if index == 8 else 3
            rectangle(image, (int(point[0] - size), int(point[1] - size)),
                      (int(point[0] + size), int(point[1] + size)), YELLOW if index == 8 else PINK)

    def _pointer_settings(self, image, state) -> None:
        """Render hand/marker preview and HSV marker-tuning controls."""
        if state.current_frame is not None:
            x, y, w, h = SETTINGS_PREVIEW
            preview = resize_bilinear(state.current_frame, (w, h))
            image[y:y + h, x:x + w] = preview
            self._draw_hand(image, getattr(state, "current_detection", None), state.frame_shape, SETTINGS_PREVIEW)
            rectangle(image, (x, y), (x + w, y + h), BLUE if state.sample_mode else NAVY, 3)

        # Show a visible marker-color preview based on the current HSV settings.
        lower = state.config["hsv_lower"]
        upper = state.config["hsv_upper"]
        hue_offset = (upper[0] - lower[0]) % 180
        hue = (lower[0] + hue_offset // 2) % 180
        saturation = max(lower[1], min(255, (lower[1] + 255) // 2))
        value = max(lower[2], min(255, (lower[2] + 255) // 2))
        hsv_preview = np.uint8([[[hue, saturation, value]]])
        marker_bgr = tuple(int(channel) for channel in cv2.cvtColor(hsv_preview, cv2.COLOR_HSV2BGR)[0, 0])
        rectangle(image, (280, 205), (470, 257), NAVY)
        rectangle(image, (283, 208), (467, 254), PANEL)
        put_text(image, "MARKER COLOR", (298, 228), 0.38, MUTED, 2)
        rectangle(image, (395, 219), (453, 243), marker_bgr)
        rectangle(image, (395, 219), (453, 243), NAVY, 1)

        put_text(image, "21 LANDMARKS | INDEX FINGER = CURSOR | PINCH = CLICK", (280, 180), 0.45, MUTED)
        put_text(image, getattr(state, "hand_gesture", "SHOW YOUR HAND"), (25, 420), 0.50, NAVY, 2)
        put_text(image, f"Fingers: {getattr(state, 'hand_fingers', [])}", (25, 450), 0.40, MUTED)
        put_text(image, "Marker tuning controls", (25, 490), 0.42, MUTED)
        values = {"h_low": state.config["hsv_lower"][0], "h_high": state.config["hsv_upper"][0],
                  "s_min": state.config["hsv_lower"][1], "v_min": state.config["hsv_lower"][2],
                  "area": state.config["min_area"]}
        for key, label, low, high, x, y in state.slider_specs():
            width = 260 if key != "area" else 160
            value = values[key]
            ratio = (value - low) / (high - low)
            put_text(image, f"{label}: {value}", (x, y - 12), 0.41, INK)
            rectangle(image, (x, y), (x + width, y + 6), (205, 212, 220))
            rectangle(image, (x, y), (int(x + ratio * width), y + 6), TEAL)
            circle(image, (x + ratio * width, y + 3), 7, NAVY, -1)

    def _remote_settings(self, image, state) -> None:
        """Render remote-key events and active left/right click bindings."""
        put_text(image, "Remote Key Inspector & Remapper", (280, 145), 0.72, NAVY, 2)
        put_text(image, "Press either remote button. Its OS key name and codes appear below.", (280, 185), 0.53, MUTED)
        event = getattr(state, "last_remote_event", None)
        rectangle(image, (280, 215), (1190, 325), (226, 233, 240))
        if event:
            put_text(image, f"Last key: {event.name}", (305, 255), 0.72, NAVY, 2)
            put_text(image, f"{event.code}   {'PRESS' if event.pressed else 'RELEASE'}", (305, 295), 0.62, BLUE)
        else:
            put_text(image, "Waiting for a key…", (305, 275), 0.65, MUTED)
        put_text(image, "Next / Left click: " + ", ".join(sorted(state.router.left_keys)), (290, 460), 0.54, INK)
        put_text(image, "Previous / Right click: " + ", ".join(sorted(state.router.right_keys)), (290, 495), 0.54, INK)
        put_text(image, "Observed codes are appended to remote_keycodes.log", (290, 535), 0.48, MUTED)
        if state.router.capture_action:
            put_text(image, f"CAPTURING {state.router.capture_action.upper()} CLICK — PRESS A REMOTE KEY", (290, 590), 0.56, (50, 100, 210), 2)

    def _calibration(self, image, state, camera_frame) -> None:
        """Render the calibration instructions, live camera, corners, and grid."""
        put_text(image, "Four-Corner Calibration", (25, 93), 0.62, NAVY, 2)
        put_text(image, "TL -> TR -> BR -> BL", (25, 270), 0.52, MUTED)
        wrapped_text(image, state.calibration_message, 25, 320, 210, 0.46, INK, 24)
        if camera_frame is None:
            return
        working = camera_frame.copy()
        height, width = working.shape[:2]
        if state.calibration:
            grid = []
            for gx in range(0, BOARD_WIDTH, 160):
                grid.append(([gx, 0], [gx, BOARD_HEIGHT - 1]))
            for gy in range(0, BOARD_HEIGHT, 120):
                grid.append(([0, gy], [BOARD_WIDTH - 1, gy]))
            for start, end in grid:
                a, b = inverse_project([start, end], state.calibration.matrix)
                line(working, a, b, (60, 210, 90), 1)
        x, y, w, h = CAMERA_RECT
        image[y:y + h, x:x + w] = resize_bilinear(working, (w, h))
        rectangle(image, (x, y), (x + w, y + h), BLUE, 2)
        labels = ["TL", "TR", "BR", "BL"]
        for index, (cx, cy) in enumerate(state.calibration_points):
            px, py = x + cx * w / width, y + cy * h / height
            circle(image, (px, py), 10, (50, 80, 230), -1)
            put_text(image, f"{index + 1} {labels[index]}", (px + 13, py - 10), 0.48, NAVY, 2)
        if state.calibration:
            put_text(image, f"RMSE {state.calibration.rmse:.6f} px", (970, 108), 0.48, (45, 140, 85), 2)

    def _debug_pip(self, image, camera_frame, mask, detection) -> None:
        """Render the optional camera, segmentation-mask, and pointer debug picture."""
        x, y, w, h = 970, 452, 250, 150
        rectangle(image, (x - 5, y - 26), (x + w + 5, y + h + 5), NAVY)
        put_text(image, "CAMERA / MASK", (x, y - 8), 0.42, WHITE)
        half = w // 2
        image[y:y + h, x:x + half] = resize_bilinear(camera_frame, (half, h))
        mask_frame = np.repeat((mask[..., None] * 255), 3, axis=2) if mask is not None else np.zeros_like(camera_frame)
        image[y:y + h, x + half:x + w] = resize_bilinear(mask_frame, (w - half, h))
        if detection:
            self._draw_hand(image, detection, camera_frame.shape, (x, y, half, h))
            self._draw_hand(image, detection, camera_frame.shape, (x + half, y, w - half, h))
            sx, sy = half / camera_frame.shape[1], h / camera_frame.shape[0]
            x0, y0, x1, y1 = detection.bbox
            rectangle(image, (int(x + x0 * sx), int(y + y0 * sy)),
                      (int(x + x1 * sx), int(y + y1 * sy)), (50, 80, 240), 1)
            circle(image, (x + detection.centroid[0] * sx, y + detection.centroid[1] * sy), 4, (50, 240, 80), -1)

    def _ripples(self, image, state) -> None:
        """Draw short-lived visual ripples for left and right input clicks."""
        now = time.monotonic()
        state.ripples = [r for r in state.ripples if now - r[0] < 0.55]
        for started, x, y, kind in state.ripples:
            progress = (now - started) / 0.55
            color = (55, 145, 225) if kind == "left" else (170, 75, 190)
            circle(image, (x, y), int(12 + progress * 28), color, 2)
