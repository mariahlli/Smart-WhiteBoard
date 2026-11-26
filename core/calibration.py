"""Four-point projective calibration implemented with NumPy only.

The camera coordinates are (u, v), in pixels. Board coordinates are (X, Y),
with (0, 0) at the upper-left board corner. A 3x3 homography H maps column
vectors [u, v, 1] to homogeneous board coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


def _normalization_transform(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hartley-normalized points and their similarity transform.

    Hartley normalization centers points at the origin and gives them an
    average distance of sqrt(2). This improves the conditioning of the DLT
    design matrix and is important when coordinates are hundreds of pixels.
    """
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    if mean_distance < 1e-12:
        raise ValueError("Calibration points are degenerate")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def compute_homography(
    source_points: Iterable[Iterable[float]],
    destination_points: Iterable[Iterable[float]],
) -> np.ndarray:
    """Estimate H using four or more correspondences and normalized DLT.

    Each correspondence contributes two linear equations. The solution is the
    right-singular vector associated with the smallest singular value of A.
    """
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Source and destination must have matching Nx2 shapes")
    if len(source) < 4:
        raise ValueError("At least four correspondences are required")
    if np.linalg.matrix_rank(source) < 2 or np.linalg.matrix_rank(destination) < 2:
        raise ValueError("Calibration points must span a non-degenerate plane")

    src, Ts = _normalization_transform(source)
    dst, Td = _normalization_transform(destination)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    normalized_h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(Td) @ normalized_h @ Ts
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("Calibration produced an invalid homography")
    return h / h[2, 2]


def map_points(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    """Project an Nx2 array through H, including homogeneous division."""
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 1
    array = np.atleast_2d(array)
    homogeneous = np.column_stack((array, np.ones(len(array))))
    projected = (np.asarray(homography) @ homogeneous.T).T
    denominator = projected[:, 2:3]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Point maps to infinity")
    result = projected[:, :2] / denominator
    return result[0] if single else result


def inverse_project(points: Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    """Map board points back to camera coordinates using H inverse."""
    return map_points(points, np.linalg.inv(homography))


def reprojection_rmse(
    source_points: Iterable[Iterable[float]],
    destination_points: Iterable[Iterable[float]],
    homography: np.ndarray,
) -> float:
    """Round-trip RMSE in camera pixels (an intuitive validation metric)."""
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    reconstructed = inverse_project(destination, homography)
    return float(np.sqrt(np.mean(np.sum((source - reconstructed) ** 2, axis=1))))


@dataclass
class Calibration:
    """Serializable camera-to-board calibration and its validation score."""

    matrix: np.ndarray
    camera_points: list[list[float]]
    board_points: list[list[float]]
    rmse: float
    camera_size: tuple[int, int]

    @classmethod
    def from_points(
        cls,
        camera_points: Iterable[Iterable[float]],
        board_size: tuple[int, int],
        camera_size: tuple[int, int],
    ) -> "Calibration":
        """Build a calibrated homography from four camera-to-board correspondences."""
        width, height = board_size
        board_points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        matrix = compute_homography(camera_points, board_points)
        rmse = reprojection_rmse(camera_points, board_points, matrix)
        return cls(matrix, np.asarray(camera_points).tolist(), board_points, rmse, camera_size)

    def to_dict(self) -> dict:
        """Serialize calibration values for config.json."""
        return {
            "matrix": np.asarray(self.matrix).tolist(),
            "camera_points": self.camera_points,
            "board_points": self.board_points,
            "rmse": self.rmse,
            "camera_size": list(self.camera_size),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Calibration | None":
        if not data:
            return None
        return cls(
            np.asarray(data["matrix"], dtype=np.float64),
            data["camera_points"],
            data["board_points"],
            float(data["rmse"]),
            tuple(data["camera_size"]),
        )
