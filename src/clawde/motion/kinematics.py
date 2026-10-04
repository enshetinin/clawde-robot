"""Kinematics gate for the real arm.

The exact Adeept model, link lengths and servo calibration are unknown, so no
geometry is supported yet. This module refuses instead of inventing angles.
Add a solver to SUPPORTED_GEOMETRIES only after measuring and verifying it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from clawde.config import GeometryConfig
from clawde.errors import KinematicsUnavailable

IKSolver = Callable[[GeometryConfig, Sequence[float]], Mapping[str, float]]

# kind -> solver. Intentionally empty until a geometry is measured and verified.
SUPPORTED_GEOMETRIES: dict[str, IKSolver] = {}


def require_solver(geometry: GeometryConfig | None) -> IKSolver:
    if geometry is None:
        raise KinematicsUnavailable("geometría del brazo sin definir en robot.yaml")
    solver = SUPPORTED_GEOMETRIES.get(geometry.kind)
    if solver is None:
        raise KinematicsUnavailable(
            f"no hay cinemática verificada para la geometría '{geometry.kind}'"
        )
    return solver


def solve_ik(geometry: GeometryConfig | None, point_mm: Sequence[float]) -> Mapping[str, float]:
    """Return joint targets in degrees, or raise KinematicsUnavailable."""
    return require_solver(geometry)(geometry, point_mm)  # type: ignore[arg-type]
