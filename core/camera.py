"""Cross-platform camera acquisition and a deterministic synthetic camera."""
from __future__ import annotations

from dataclasses import dataclass
import time
import cv2
import numpy as np

from .calibration import compute_homography, inverse_project, map_points
from ui.primitives import circle, line


SYNTHETIC_CORNERS = np.array([[78, 32], [562, 45], [594, 326], [43, 310]], dtype=np.float64)


class SyntheticCamera:
    """Offline camera showing a projected board and a moving neon marker.

    The board itself is projectively rendered with the same NumPy homography
    code used by the application, making this mode useful for end-to-end demos.
    """

    def __init__(self, width: int = 640, height: int = 360):
        # Precompute the projective board and a moving pointer demo frame.
        self.width, self.height = width, height
        self.board_size = (1280, 720)
        self.homography = compute_homography(
            SYNTHETIC_CORNERS,
            [[0, 0], [1279, 0], [1279, 719], [0, 719]],
        )
        self.started = time.monotonic()
        self.frame_number = 0
        self.background = self._make_background()

    def _make_background(self) -> np.ndarray:
        """Generate the static projected classroom and grid used for demos."""
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = np.clip(30 + yy * 0.08, 0, 255)
        image[..., 1] = np.clip(23 + yy * 0.055, 0, 255)
        image[..., 2] = np.clip(20 + xx * 0.018, 0, 255)
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))
        board = map_points(coordinates, self.homography)
        bx = np.clip(np.rint(board[:, 0]).astype(int), 0, 1279)
        by = np.clip(np.rint(board[:, 1]).astype(int), 0, 719)
        inside = ((board[:, 0] >= 0) & (board[:, 0] < 1280) &
                  (board[:, 1] >= 0) & (board[:, 1] < 720))
        pixels = image.reshape(-1, 3)
        # A subdued projected interface visible in the synthetic classroom.
        board_blue = 225 - (bx // 12)
        pixels[inside, 0] = board_blue[inside]
        pixels[inside, 1] = 220 - (by[inside] // 18)
        pixels[inside, 2] = 205 - (by[inside] // 24)
        top_strip = inside & (board[:, 1] < 70)
        pixels[top_strip] = (105, 62, 35)
        for grid_x in range(0, 1280, 160):
            points = inverse_project([[grid_x, 0], [grid_x, 719]], self.homography)
            line(image, points[0], points[1], (188, 194, 190), 1)
        for grid_y in range(0, 720, 120):
            points = inverse_project([[0, grid_y], [1279, grid_y]], self.homography)
            line(image, points[0], points[1], (188, 194, 190), 1)
        border = inverse_project([[0, 0], [1279, 0], [1279, 719], [0, 719], [0, 0]], self.homography)
        for index in range(4):
            line(image, border[index], border[index + 1], (245, 245, 245), 3)
        return image

    def read(self):
        """Return the next synthetic frame and advance the moving marker."""
        elapsed = time.monotonic() - self.started
        board_point = [[
            640 + 440 * np.sin(elapsed * 0.72),
            365 + 235 * np.sin(elapsed * 1.07 + 0.4),
        ]]
        camera_point = inverse_project(board_point, self.homography)[0]
        frame = self.background.copy()
        x, y = camera_point
        circle(frame, (x + 3, y + 4), 14, (65, 65, 65), -1)
        circle(frame, (x, y), 11, (35, 245, 70), -1)
        circle(frame, (x - 2, y - 2), 4, (90, 255, 115), -1)
        cv2.putText(frame, "SYNTHETIC CAMERA", (14, 344), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (200, 210, 220), 1, cv2.LINE_AA)
        self.frame_number += 1
        return True, frame

    def release(self) -> None:
        """Release resources held by the synthetic source (a no-op)."""
        pass


@dataclass
class CameraStatus:
    """Small immutable snapshot describing an active camera source."""

    index: int
    synthetic: bool
    width: int
    height: int


class CameraManager:
    """Live device switcher with non-destructive NumPy image flipping."""

    def __init__(self, index: int = -1, mirror: bool = False, flip_vertical: bool = False):
        # Open the requested source once and remember non-destructive flips.
        self.index = index
        self.mirror = mirror
        self.flip_vertical = flip_vertical
        self.capture = None
        self.last_error = ""
        self.open(index)

    def open(self, index: int) -> bool:
        """Switch to a real device or the synthetic fallback."""
        self.release()
        self.index = int(index)
        if self.index < 0:
            self.capture = SyntheticCamera()
            self.last_error = ""
            return True
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.last_error = f"Camera {self.index} unavailable; switched to synthetic mode."
            self.index = -1
            self.capture.release()
            self.capture = SyntheticCamera()
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.last_error = ""
        return True

    def cycle(self) -> None:
        """Rotate through synthetic and the available demo device indexes."""
        choices = [-1, 0, 1, 2]
        next_position = (choices.index(self.index) + 1) if self.index in choices else 0
        self.open(choices[next_position % len(choices)])

    def read(self):
        """Read a frame, apply orientation flips, and return a contiguous array."""
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        if self.mirror:
            frame = np.flip(frame, axis=1)
        if self.flip_vertical:
            frame = np.flip(frame, axis=0)
        return True, np.ascontiguousarray(frame)

    def status(self) -> CameraStatus:
        """Report source index and current reported frame dimensions."""
        frame = getattr(self.capture, "background", None)
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.capture else 0
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.capture else 0
        return CameraStatus(self.index, self.index < 0, width, height)

    def release(self) -> None:
        """Release the OpenCV capture object and clear it."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
