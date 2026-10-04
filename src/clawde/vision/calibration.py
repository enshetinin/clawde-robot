"""Table-plane calibration: pixel -> millimetre homography.

Assumptions: the table is a plane, lens distortion is small (or already
corrected) and the camera does not move after calibrating. The homography
gives X/Y on the table only; height (Z) and grasp pose do NOT come from it.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clawde.errors import ConfigError, require

Point = tuple[float, float]


class CalibrationError(ConfigError):
    pass


def _collinear(a: Point, b: Point, c: Point, tol: float) -> bool:
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    scale = max(math.dist(a, b), math.dist(a, c), math.dist(b, c), 1e-9)
    return area2 / scale < tol


def _check_points(points: Sequence[Point], name: str, tol: float) -> None:
    for p in points:
        if len(p) != 2 or not all(math.isfinite(float(v)) for v in p):
            raise CalibrationError(f"punto no válido en {name}: {p}")
    for a, b in itertools.combinations(points, 2):
        if math.dist(a, b) < tol:
            raise CalibrationError(f"puntos duplicados en {name}")
    for quad in itertools.combinations(points, 4):
        if not any(_collinear(*tri, tol) for tri in itertools.combinations(quad, 3)):
            return
    raise CalibrationError(
        f"correspondencias degeneradas en {name}: necesito 4 puntos sin 3 alineados"
    )


def _apply(h: Sequence[Sequence[float]], x: float, y: float) -> Point:
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-12:
        raise CalibrationError("punto en el infinito para esta homografía")
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def fit_homography(
    pixel_points: Sequence[Point], world_points_mm: Sequence[Point], tol: float = 1.0
) -> tuple[list[list[float]], float]:
    """Normalised DLT. Returns (H, RMS reprojection error in mm)."""
    np: Any = require("numpy", "vision")
    if len(pixel_points) != len(world_points_mm):
        raise CalibrationError("número distinto de puntos en imagen y mesa")
    if len(pixel_points) < 4:
        raise CalibrationError("se necesitan al menos 4 correspondencias")
    if len(pixel_points) > 50:
        raise CalibrationError("demasiadas correspondencias (máx. 50)")
    _check_points(pixel_points, "imagen", tol)
    _check_points(world_points_mm, "mesa", tol)

    def normaliser(pts: Any) -> Any:
        mean = pts.mean(axis=0)
        scale = math.sqrt(2) / max(np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean(), 1e-12)
        return np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]])

    src = np.asarray(pixel_points, dtype=float)
    dst = np.asarray(world_points_mm, dtype=float)
    t_src, t_dst = normaliser(src), normaliser(dst)
    src_h = (t_src @ np.column_stack([src, np.ones(len(src))]).T).T
    dst_h = (t_dst @ np.column_stack([dst, np.ones(len(dst))]).T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(src_h, dst_h, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[-2] / singular[0] < 1e-8:
        raise CalibrationError("sistema mal condicionado: correspondencias degeneradas")
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) < 1e-12 or abs(np.linalg.det(h)) < 1e-12:
        raise CalibrationError("homografía singular")
    h = h / h[2, 2]
    matrix = h.tolist()
    errors = [
        math.dist(_apply(matrix, *p), w) for p, w in zip(pixel_points, world_points_mm, strict=True)
    ]
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    return matrix, rms


class TableCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    homography: list[list[float]] = Field(min_length=3, max_length=3)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_index: int = Field(ge=0)
    camera_name: str | None = None
    created_at: str
    rms_error_mm: float = Field(ge=0)
    num_points: int = Field(ge=4)
    notes: str = "plano de mesa; Z y agarre no proceden de la homografía"

    @classmethod
    def from_points(
        cls,
        pixel_points: Sequence[Point],
        world_points_mm: Sequence[Point],
        image_size: tuple[int, int],
        camera_index: int,
        max_rms_mm: float,
        camera_name: str | None = None,
    ) -> TableCalibration:
        matrix, rms = fit_homography(pixel_points, world_points_mm)
        if rms > max_rms_mm:
            raise CalibrationError(f"error de ajuste {rms:.2f} mm > {max_rms_mm} mm")
        return cls(
            homography=matrix,
            image_width=image_size[0],
            image_height=image_size[1],
            camera_index=camera_index,
            camera_name=camera_name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            rms_error_mm=rms,
            num_points=len(pixel_points),
        )

    def pixel_to_mm(self, x: float, y: float) -> Point:
        return _apply(self.homography, x, y)

    def ensure_resolution(self, width: int, height: int, allow_mismatch: bool = False) -> None:
        if (width, height) != (self.image_width, self.image_height) and not allow_mismatch:
            raise CalibrationError(
                f"calibración hecha a {self.image_width}x{self.image_height}, cámara a "
                f"{width}x{height}: recalibra o valida explícitamente"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TableCalibration:
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise CalibrationError(f"no se pudo cargar la calibración {path}: {exc}") from exc
