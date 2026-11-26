"""Real hand recognition with 21 landmarks, finger states, and pinch gestures.

MediaPipe performs the learned hand/landmark inference. This module owns the
coordinate restoration, distant-hand crop search, fingertip selection, gesture
geometry, and conversion to the application's NumPy detection structure.
"""
from __future__ import annotations

import numpy as np
from core.vision import Detection


class HandTracker:
    """MediaPipe hand tracker that converts landmarks into UI pointer detections."""

    def __init__(self, far_search: bool = True):
        # MediaPipe is optional at runtime if learned-hand support is unavailable.
        self.available = False
        self.error = ""
        self.hands = None
        self.far_search = far_search
        self.misses = 0
        self.frame_index = 0
        self.pinch_active = False
        try:
            import mediapipe as mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=1,
                min_detection_confidence=0.38, min_tracking_confidence=0.38,
            )
            self.available = True
        except Exception as exc:
            self.error = f"Hand recognition unavailable: {exc}"

    @staticmethod
    def _geometry(landmarks: np.ndarray, handedness: float, previous_pinch: bool):
        """Derive finger states, pinch hysteresis, and a readable gesture label."""
        wrist = landmarks[0]
        hand_size = max(8.0, float(np.linalg.norm(landmarks[9] - wrist)))
        fingers = []
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers.append(bool(np.linalg.norm(landmarks[tip] - wrist) >
                                np.linalg.norm(landmarks[pip] - wrist) * 1.10))
        thumb_extended = bool(np.linalg.norm(landmarks[4] - landmarks[17]) >
                              np.linalg.norm(landmarks[3] - landmarks[17]) * 1.08)
        fingers.insert(0, thumb_extended)
        pinch_ratio = float(np.linalg.norm(landmarks[4] - landmarks[8]) / hand_size)
        # Hysteresis prevents a stream of accidental clicks near the threshold.
        pinch = pinch_ratio < (0.52 if previous_pinch else 0.38)
        if pinch:
            gesture = "PINCH / CLICK"
        elif fingers[1] and sum(fingers[2:]) == 0:
            gesture = "POINTING"
        elif sum(fingers) == 5:
            gesture = "OPEN HAND"
        elif sum(fingers) == 0:
            gesture = "FIST"
        else:
            gesture = f"{sum(fingers)} FINGERS"
        return fingers, pinch, pinch_ratio, gesture, handedness

    def _make_detection(self, result, frame_shape, offset=(0, 0), crop_shape=None):
        """Convert a MediaPipe hand result into the application's Detection model."""
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        crop_h, crop_w = crop_shape if crop_shape else frame_shape[:2]
        ox, oy = offset
        points = np.array([[ox + landmark.x * crop_w, oy + landmark.y * crop_h]
                        for landmark in hand.landmark], dtype=float)
        points[:, 0] = np.clip(points[:, 0], 0, frame_shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, frame_shape[0] - 1)
        confidence = float(result.multi_handedness[0].classification[0].score) if result.multi_handedness else 0.0
        fingers, pinch, ratio, gesture, confidence = self._geometry(points, confidence, self.pinch_active)
        self.pinch_active = pinch
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad = max(3, (x1 - x0 + y1 - y0) * 0.025)
        bbox = (int(max(0, x0 - pad)), int(max(0, y0 - pad)),
                int(min(frame_shape[1] - 1, x1 + pad)), int(min(frame_shape[0] - 1, y1 + pad)))
        detection = Detection((float(points[8, 0]), float(points[8, 1])), bbox,
                              int((x1 - x0) * (y1 - y0)), np.zeros(frame_shape[:2], dtype=np.uint8))
        detection.landmarks = points
        detection.fingers = fingers
        detection.gesture = gesture
        detection.confidence = confidence
        detection.pinch = pinch
        detection.pinch_ratio = ratio
        detection.is_hand = True
        return detection

    def _infer(self, frame, offset=(0, 0)):
        """Run MediaPipe inference and restore crop-relative landmark coordinates."""
        rgb = np.ascontiguousarray(frame[..., ::-1])
        result = self.hands.process(rgb)
        return self._make_detection(result, frame.shape, offset=(0, 0)) if offset == (0, 0) else result

    def detect(self, frame: np.ndarray) -> Detection | None:
        """Return the index fingertip and hand metadata, not a color centroid.

        If the full-frame detector loses a distant hand, overlapping quadrant
        crops provide a larger effective hand size without modifying camera
        settings. The expensive search runs only occasionally during loss.
        """
        if not self.available:
            return None
        self.frame_index += 1
        detection = self._infer(frame)
        if detection is None and self.far_search and self.misses >= 2 and self.frame_index % 4 == 0:
            height, width = frame.shape[:2]
            crop_w, crop_h = int(width * 0.62), int(height * 0.62)
            xs = [0, width - crop_w]
            ys = [0, height - crop_h]
            for oy in ys:
                for ox in xs:
                    crop = np.ascontiguousarray(frame[oy:oy + crop_h, ox:ox + crop_w])
                    result = self.hands.process(np.ascontiguousarray(crop[..., ::-1]))
                    detection = self._make_detection(result, frame.shape, (ox, oy), (crop_h, crop_w))
                    if detection is not None:
                        break
                if detection is not None:
                    break
        self.misses = 0 if detection is not None else self.misses + 1
        if detection is None:
            self.pinch_active = False
        return detection

    def close(self):
        if self.hands is not None:
            self.hands.close()
