"""Vectorized color segmentation, morphology, blobs, and stabilization.

No OpenCV image-processing functions are used in this module. OpenCV's HSV
convention is used: H in [0, 179], S and V in [0, 255].
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to uint8 HSV using vectorized NumPy arithmetic."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 uint8 BGR image")
    b, g, r = np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)
    maximum = np.maximum(np.maximum(r, g), b)
    minimum = np.minimum(np.minimum(r, g), b)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-7
    red = nonzero & (maximum == r)
    green = nonzero & (maximum == g) & ~red
    blue = nonzero & ~(red | green)
    hue[red] = (60.0 * ((g[red] - b[red]) / delta[red])) % 360.0
    hue[green] = 60.0 * ((b[green] - r[green]) / delta[green] + 2.0)
    hue[blue] = 60.0 * ((r[blue] - g[blue]) / delta[blue] + 4.0)
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-7)
    h = np.rint(hue / 2.0) % 180
    s = np.clip(np.rint(saturation * 255.0), 0, 255)
    v = np.clip(np.rint(maximum * 255.0), 0, 255)
    return np.stack((h, s, v), axis=-1).astype(np.uint8)


def bgr_to_ycbcr(frame: np.ndarray) -> np.ndarray:
    """Full-range BGR to Y/Cb/Cr transform for the secondary skin mode."""
    b, g, r = np.moveaxis(frame.astype(np.float32), -1, 0)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.clip(np.stack((y, cb, cr), axis=-1), 0, 255).astype(np.uint8)


def threshold_mask(channels: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Multi-channel inclusive threshold; supports circular HSV hue ranges."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    mask = np.ones(channels.shape[:2], dtype=bool)
    for channel in range(channels.shape[2]):
        lo, hi = int(lower[channel]), int(upper[channel])
        values = channels[..., channel]
        if channel == 0 and channels.shape[2] == 3 and lo > hi:
            mask &= (values >= lo) | (values <= hi)
        else:
            mask &= (values >= lo) & (values <= hi)
    return mask.astype(np.uint8)


def _morphology(mask: np.ndarray, kernel_size: int, maximum: bool) -> np.ndarray:
    """Separable square morphology using only vectorized shifted slices.

    A square min/max filter equals a horizontal pass followed by a vertical
    pass. This avoids a large 3-D sliding-window allocation and is much faster
    on camera frames while remaining entirely NumPy-owned.
    """
    if kernel_size % 2 != 1 or kernel_size < 1:
        raise ValueError("Kernel size must be a positive odd number")
    pad = kernel_size // 2
    binary = (mask > 0).astype(np.uint8)
    padded = np.pad(binary, ((0, 0), (pad, pad)), mode="constant")
    horizontal = binary if pad == 0 else (np.zeros_like(binary) if maximum else np.ones_like(binary))
    for offset in range(kernel_size):
        shifted = padded[:, offset:offset + binary.shape[1]]
        horizontal = np.maximum(horizontal, shifted) if maximum else np.minimum(horizontal, shifted)
    padded = np.pad(horizontal, ((pad, pad), (0, 0)), mode="constant")
    result = horizontal if pad == 0 else (np.zeros_like(binary) if maximum else np.ones_like(binary))
    for offset in range(kernel_size):
        shifted = padded[offset:offset + binary.shape[0], :]
        result = np.maximum(result, shifted) if maximum else np.minimum(result, shifted)
    return result


def erode(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Binary erosion implemented with separable NumPy shifted slices."""
    return _morphology(mask, kernel_size, maximum=False)


def dilate(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Binary dilation implemented with separable NumPy shifted slices."""
    return _morphology(mask, kernel_size, maximum=True)


def opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return dilate(erode(mask, kernel_size), kernel_size)


def closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return erode(dilate(mask, kernel_size), kernel_size)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove isolated speckles, then fill small holes."""
    return closing(opening(mask, 3), 3)


@dataclass
class Detection:
    """Connected-component pointer result in camera pixel coordinates."""

    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    mask: np.ndarray


def _components(mask: np.ndarray):
    """Run-length union-find; Python iterates over runs, never pixels."""
    binary = (mask > 0).astype(np.uint8)
    height, _ = binary.shape
    runs: list[tuple[int, int, int]] = []
    row_ranges: list[tuple[int, int]] = []
    for y in range(height):
        first = len(runs)
        padded = np.pad(binary[y], (1, 1))
        changes = np.diff(padded.astype(np.int16))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        runs.extend((y, int(start), int(end)) for start, end in zip(starts, ends))
        row_ranges.append((first, len(runs)))
    if not runs:
        return binary, runs, [], np.array([], dtype=np.int64)
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for y in range(1, height):
        current_start, current_end = row_ranges[y]
        previous_start, previous_end = row_ranges[y - 1]
        i, j = current_start, previous_start
        while i < current_end and j < previous_end:
            _, cs, ce = runs[i]
            _, ps, pe = runs[j]
            if cs <= pe + 1 and ps <= ce + 1:
                root_a, root_b = find(i), find(j)
                if root_a != root_b:
                    parent[root_b] = root_a
            if ce < pe:
                i += 1
            else:
                j += 1
    areas = np.zeros(len(runs), dtype=np.int64)
    for index, (_, start, end) in enumerate(runs):
        areas[find(index)] += end - start + 1
    return binary, runs, parent, areas


def find_blobs(mask: np.ndarray, min_area: int = 1, limit: int = 8) -> list[Detection]:
    """Return up to ``limit`` components, largest first, with exact moments."""
    binary, runs, parent, areas = _components(mask)
    if not runs:
        return []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    roots = [int(root) for root in np.argsort(areas)[::-1] if areas[root] >= min_area]
    # argsort contains every run label; only roots own an accumulated area.
    unique_roots = []
    for root in roots:
        if find(root) == root and root not in unique_roots:
            unique_roots.append(root)
        if len(unique_roots) >= limit:
            break
    detections = []
    for root in unique_roots:
        component = np.zeros_like(binary)
        for index, (y, start, end) in enumerate(runs):
            if find(index) == root:
                component[y, start:end + 1] = 1
        ys, xs = np.nonzero(component)
        detections.append(Detection(
            (float(xs.mean()), float(ys.mean())),
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            int(xs.size), component,
        ))
    return detections


def largest_blob(mask: np.ndarray, min_area: int = 1) -> Detection | None:
    """Compatibility wrapper returning the largest connected component."""
    blobs = find_blobs(mask, min_area, limit=1)
    return blobs[0] if blobs else None


def _skin_tip(detection: Detection) -> Detection:
    """Estimate the raised fingertip from the upper narrow part of a hand blob.

    A whole-hand centroid lies near the palm and moves sluggishly. Averaging
    only the upper 16% gives a responsive pointing tip while remaining more
    stable than selecting a single topmost pixel.
    """
    ys, xs = np.nonzero(detection.mask)
    if not len(xs):
        return detection
    x0, y0, x1, y1 = detection.bbox
    band = max(3.0, (y1 - y0 + 1) * 0.16)
    selected = ys <= y0 + band
    if selected.sum() >= 2:
        return Detection((float(xs[selected].mean()), float(ys[selected].mean())),
                        detection.bbox, detection.area, detection.mask)
    return detection


def segment_pointer(
    frame: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mode: str = "marker",
    min_area: int = 45,
    preferred_point: tuple[float, float] | None = None,
) -> tuple[Detection | None, np.ndarray, np.ndarray]:
    """Detect a pointer with temporal candidate selection and sparse-mask fallback."""
    hsv = bgr_to_hsv(frame)
    if mode == "skin":
        ycbcr = bgr_to_ycbcr(frame)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
        b, g, r = np.moveaxis(frame.astype(np.int16), -1, 0)
        hsv_skin = ((h <= 32) | (h >= 170)) & (s >= 18) & (v >= 35)
        ycc_skin = (y >= 35) & (cb >= 72) & (cb <= 138) & (cr >= 128) & (cr <= 185)
        rgb_skin = ((r > 45) & (g > 25) & (b > 15) & (r > g) & (r > b) &
                    ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 12) &
                    (np.abs(r - g) > 6))
        raw = (ycc_skin & (hsv_skin | rgb_skin)).astype(np.uint8)
        min_area = max(min_area, 80)
    else:
        raw = threshold_mask(hsv, lower, upper)
    cleaned = clean_mask(raw)

    # Opening alone preserves thin tips that open-close can remove. Raw mask is
    # a final fallback for a distant/small marker; min_area still rejects dust.
    candidates = find_blobs(cleaned, min_area)
    if not candidates:
        candidates = find_blobs(opening(raw, 3), min_area)
    if not candidates:
        candidates = find_blobs(raw, min_area)
    if not candidates:
        return None, cleaned, hsv
    if preferred_point is not None:
        preferred = np.asarray(preferred_point)
        def score(detection):
            distance = np.linalg.norm(np.asarray(detection.centroid) - preferred)
            return np.log1p(detection.area) - distance / 55.0
        detection = max(candidates, key=score)
    else:
        detection = candidates[0]
    if mode == "skin":
        detection = _skin_tip(detection)
    return detection, cleaned, hsv


def _bounds_from_pixels(pixels: np.ndarray) -> tuple[list[int], list[int]]:
    """Robust circular-hue interval with generous lighting tolerance."""
    if not len(pixels):
        raise ValueError("No pixels available for color sampling")
    saturation_cutoff = np.percentile(pixels[:, 1], 45)
    selected = pixels[pixels[:, 1] >= saturation_cutoff]
    if not len(selected):
        selected = pixels
    angles = selected[:, 0].astype(float) * (2 * np.pi / 180.0)
    hue = (np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) * 180 / (2 * np.pi)) % 180
    center = int(round(hue)) % 180
    lower_h, upper_h = (center - 14) % 180, (center + 14) % 180
    s_low = int(min(150, max(30, np.percentile(selected[:, 1], 5) - 35)))
    v_low = int(min(140, max(25, np.percentile(selected[:, 2], 5) - 45)))
    return [lower_h, s_low, v_low], [upper_h, 255, 255]


def sample_hsv_bounds(frame: np.ndarray, x: int, y: int, radius: int = 6) -> tuple[list[int], list[int]]:
    """Build safe HSV bounds from a patch selected in the GUI."""
    hsv = bgr_to_hsv(frame)
    height, width = hsv.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    return _bounds_from_pixels(hsv[y0:y1, x0:x1].reshape(-1, 3))


def auto_sample_marker(frame: np.ndarray):
    """Find a compact saturated marker and return bounds plus its detection.

    Large colorful background regions are excluded. This provides a one-click
    starting point which the user can refine with the live HSV/area sliders.
    """
    hsv = bgr_to_hsv(frame)
    raw = ((hsv[..., 1] >= 75) & (hsv[..., 2] >= 65)).astype(np.uint8)
    cleaned = clean_mask(raw)
    candidates = find_blobs(cleaned, max(12, min(45, frame.shape[0] * frame.shape[1] // 5000)))
    frame_area = frame.shape[0] * frame.shape[1]
    best, best_score = None, -np.inf
    for detection in candidates:
        x0, y0, x1, y1 = detection.bbox
        box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        if detection.area > frame_area * 0.025 or min(x1 - x0 + 1, y1 - y0 + 1) < 3:
            continue
        pixels = hsv[detection.mask > 0]
        compactness = detection.area / box_area
        score = float(pixels[:, 1].mean()) * compactness * np.log1p(detection.area)
        if score > best_score:
            best, best_score = detection, score
    if best is None:
        return None
    return (*_bounds_from_pixels(hsv[best.mask > 0]), best)


class PointerStabilizer:
    """Responsive adaptive EMA with jitter deadzone and jump rejection."""

    def __init__(self, base_alpha: float = 0.55, deadzone: float = 1.35, max_jump: float = 230.0):
        # A candidate point for a valid jump must remain in place before switching.
        self.base_alpha = base_alpha
        self.deadzone = deadzone
        self.max_jump = max_jump
        self.position: np.ndarray | None = None
        self.missing = 0
        self._jump_candidate: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the last position and treat the next sample as a fresh start."""
        self.position = None
        self.missing = 0
        self._jump_candidate = None

    def update(self, point: tuple[float, float] | None) -> tuple[float, float] | None:
        """Smooth one camera point while rejecting dust and accidental teleports."""
        if point is None:
            self.missing += 1
            return None if self.missing > 10 else (tuple(self.position) if self.position is not None else None)
        self.missing = 0
        measured = np.asarray(point, dtype=np.float64)
        if self.position is None:
            self.position = measured
            return tuple(measured)
        distance = float(np.linalg.norm(measured - self.position))
        if distance > self.max_jump:
            if self._jump_candidate is not None and np.linalg.norm(measured - self._jump_candidate) < 35:
                self.position = measured
                self._jump_candidate = None
            else:
                self._jump_candidate = measured
            return tuple(self.position)
        self._jump_candidate = None
        if distance < self.deadzone:
            return tuple(self.position)
        alpha = self.base_alpha + (1.0 - self.base_alpha) * min(distance / 65.0, 1.0)
        self.position = alpha * measured + (1.0 - alpha) * self.position
        return tuple(self.position)
