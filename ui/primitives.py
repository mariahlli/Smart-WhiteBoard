"""Pure NumPy drawing and image-resampling primitives.

Functions modify an HxWx3 uint8 framebuffer in place. Local regions of
interest keep thick-line and antialiasing costs proportional to stroke size.
"""
from __future__ import annotations

import math
import numpy as np


def _blend_region(image: np.ndarray, x0: int, y0: int, color, coverage: np.ndarray) -> None:
    """Alpha-blend one tightly bounded rectangle into the framebuffer."""
    height, width = image.shape[:2]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1 = min(width, x0 + coverage.shape[1])
    sy1 = min(height, y0 + coverage.shape[0])
    if sx0 >= sx1 or sy0 >= sy1:
        return
    alpha = coverage[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, None].astype(np.float32)
    region = image[sy0:sy1, sx0:sx1].astype(np.float32)
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    image[sy0:sy1, sx0:sx1] = np.clip(region * (1.0 - alpha) + color_array * alpha, 0, 255).astype(np.uint8)


def alpha_blend(image: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    """Blend a full overlay image into a destination framebuffer in place."""
    """Blend an equally sized overlay into an image using NumPy."""
    image[:] = np.clip(image.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def line(image: np.ndarray, p0, p1, color=(255, 255, 255), thickness: int = 1, antialias: bool = True) -> None:
    """Draw an antialiased vector segment using bounded NumPy vector operations."""
    """Draw a thick antialiased segment via vectorized point-line distance."""
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    radius = max(0.5, thickness / 2.0)
    margin = radius + 1.5
    left, top = int(math.floor(min(x0, x1) - margin)), int(math.floor(min(y0, y1) - margin))
    right, bottom = int(math.ceil(max(x0, x1) + margin)), int(math.ceil(max(y0, y1) + margin))
    if right <= 0 or bottom <= 0 or left >= image.shape[1] or top >= image.shape[0]:
        return
    yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
    dx, dy = x1 - x0, y1 - y0
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-9:
        distance = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_squared, 0.0, 1.0)
        distance = np.sqrt((xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2)
    if antialias:
        coverage = np.clip(radius + 0.65 - distance, 0.0, 1.0)
    else:
        coverage = (distance <= radius).astype(np.float32)
    _blend_region(image, left, top, color, coverage)


def circle(image: np.ndarray, center, radius: int, color=(255, 255, 255), thickness: int = -1) -> None:
    """Draw a filled or outlined circle at a framebuffer coordinate."""
    """Draw a filled or outlined antialiased circle."""
    cx, cy = map(float, center)
    outer = abs(radius) + 1.5
    left, top = int(math.floor(cx - outer)), int(math.floor(cy - outer))
    right, bottom = int(math.ceil(cx + outer)), int(math.ceil(cy + outer))
    if right <= 0 or bottom <= 0 or left >= image.shape[1] or top >= image.shape[0]:
        return
    yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if thickness < 0:
        coverage = np.clip(radius + 0.6 - distance, 0.0, 1.0)
    else:
        half = max(0.5, thickness / 2.0)
        coverage = np.clip(half + 0.6 - np.abs(distance - radius), 0.0, 1.0)
    _blend_region(image, left, top, color, coverage)


def rectangle(image: np.ndarray, p0, p1, color=(255, 255, 255), thickness: int = -1) -> None:
    """Draw a clipped filled or outlined axis-aligned rectangle."""
    """Draw a filled or outlined axis-aligned rectangle."""
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)
    x0, x1 = sorted((max(0, x0), min(image.shape[1] - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(image.shape[0] - 1, y1)))
    if x1 < x0 or y1 < y0:
        return
    if thickness < 0:
        image[y0:y1 + 1, x0:x1 + 1] = np.asarray(color, dtype=np.uint8)
    else:
        line(image, (x0, y0), (x1, y0), color, thickness)
        line(image, (x1, y0), (x1, y1), color, thickness)
        line(image, (x1, y1), (x0, y1), color, thickness)
        line(image, (x0, y1), (x0, y0), color, thickness)


def rounded_rectangle(image: np.ndarray, p0, p1, radius: int = 10,
                    color=(255, 255, 255), thickness: int = -1) -> None:
    """Draw a clipped rounded panel using a signed-distance coverage mask."""
    """Antialiased rounded rectangle evaluated from a vectorized SDF."""
    x0, y0 = map(int, p0)
    x1, y1 = map(int, p1)
    left, top = max(0, x0 - 1), max(0, y0 - 1)
    right, bottom = min(image.shape[1] - 1, x1 + 1), min(image.shape[0] - 1, y1 + 1)
    if right <= left or bottom <= top:
        return
    radius = min(radius, abs(x1 - x0) // 2, abs(y1 - y0) // 2)
    yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    qx = np.abs(xx - cx) - (abs(x1 - x0) / 2 - radius)
    qy = np.abs(yy - cy) - (abs(y1 - y0) / 2 - radius)
    distance = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + \
        np.minimum(np.maximum(qx, qy), 0) - radius
    if thickness < 0:
        coverage = np.clip(0.6 - distance, 0, 1)
    else:
        coverage = np.clip(thickness / 2 + 0.6 - np.abs(distance), 0, 1)
    _blend_region(image, left, top, color, coverage)


def crosshair(image: np.ndarray, point, active: bool = True, ink_color=None) -> None:
    """Draw the visible pointer/cursor with a crisp square target frame."""
    if point is None:
        return
    color = ink_color if ink_color is not None else ((185, 214, 60) if active else (130, 130, 130))
    x, y = map(int, point)
    # Crisp 8-bit targeting cursor with square corner brackets.
    rectangle(image, (x - 11, y - 11), (x + 11, y + 11), (42, 18, 20), 3)
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        rectangle(image, (x + dx * 14 - 3, y + dy * 14 - 3),
                  (x + dx * 14 + 3, y + dy * 14 + 3), color)
    rectangle(image, (x - 3, y - 3), (x + 3, y + 3), color)


def resize_bilinear(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height) with explicit bilinear interpolation."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Output dimensions must be positive")
    source_height, source_width = image.shape[:2]
    if (width, height) == (source_width, source_height):
        return image.copy()
    x = np.linspace(0, source_width - 1, width, dtype=np.float32)
    y = np.linspace(0, source_height - 1, height, dtype=np.float32)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx, wy = x - x0, y - y0
    top = image[y0[:, None], x0[None, :]].astype(np.float32) * (1 - wx)[None, :, None] + \
          image[y0[:, None], x1[None, :]].astype(np.float32) * wx[None, :, None]
    bottom = image[y1[:, None], x0[None, :]].astype(np.float32) * (1 - wx)[None, :, None] + \
             image[y1[:, None], x1[None, :]].astype(np.float32) * wx[None, :, None]
    return np.clip(top * (1 - wy)[:, None, None] + bottom * wy[:, None, None], 0, 255).astype(np.uint8)


def contained_rect(source_size: tuple[int, int], destination: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return the aspect-correct image rectangle inside ``(x, y, w, h)``."""
    source_width, source_height = source_size
    x, y, width, height = destination
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("Image and destination dimensions must be positive")
    scale = min(width / source_width, height / source_height)
    fitted_width = max(1, min(width, round(source_width * scale)))
    fitted_height = max(1, min(height, round(source_height * scale)))
    return (x + (width - fitted_width) // 2, y + (height - fitted_height) // 2,
            fitted_width, fitted_height)


def fit_contain(image: np.ndarray, output_size: tuple[int, int], background=(28, 32, 40)) -> np.ndarray:
    """Letterbox an image without distortion and return ``output_size`` pixels."""
    out_width, out_height = output_size
    output = np.full((out_height, out_width, 3), background, dtype=np.uint8)
    x, y, width, height = contained_rect((image.shape[1], image.shape[0]), (0, 0, out_width, out_height))
    output[y:y + height, x:x + width] = resize_bilinear(image, (width, height))
    return output


def zoom_pan(image: np.ndarray, output_size: tuple[int, int], zoom: float, pan: tuple[float, float]) -> np.ndarray:
    """Aspect-correct fit, then crop a zoomed window with bilinear scaling."""
    out_width, out_height = output_size
    fitted = fit_contain(image, output_size, background=(250, 250, 250))
    source_height, source_width = fitted.shape[:2]
    zoom = max(1.0, float(zoom))
    crop_width, crop_height = source_width / zoom, source_height / zoom
    center_x = np.clip(source_width / 2 + pan[0], crop_width / 2, source_width - crop_width / 2)
    center_y = np.clip(source_height / 2 + pan[1], crop_height / 2, source_height - crop_height / 2)
    x0 = int(np.clip(round(center_x - crop_width / 2), 0, source_width - 1))
    y0 = int(np.clip(round(center_y - crop_height / 2), 0, source_height - 1))
    x1 = int(np.clip(round(center_x + crop_width / 2), x0 + 1, source_width))
    y1 = int(np.clip(round(center_y + crop_height / 2), y0 + 1, source_height))
    return resize_bilinear(fitted[y0:y1, x0:x1], output_size)
