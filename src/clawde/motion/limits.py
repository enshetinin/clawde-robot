"""Safety validation: joint limits, speed, trajectory, workspace and scene freshness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from clawde.config import ServoConfig, WorkspaceConfig
from clawde.contracts import Scene
from clawde.errors import SafetyError


def _finite_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SafetyError(f"{what}: tipo no numérico ({type(value).__name__})")
    number = float(value)
    if not math.isfinite(number):
        raise SafetyError(f"{what}: valor no finito")
    return number


def validate_joint_targets(
    targets: Mapping[str, object], servos: Sequence[ServoConfig]
) -> dict[str, float]:
    """Check every servo has a finite target inside its configured range."""
    if not servos:
        raise SafetyError("límites de servos sin configurar")
    by_name = {s.name: s for s in servos}
    unknown = set(targets) - set(by_name)
    if unknown:
        raise SafetyError(f"articulaciones desconocidas: {sorted(unknown)}")
    missing = set(by_name) - set(targets)
    if missing:
        raise SafetyError(f"faltan articulaciones: {sorted(missing)}")
    checked: dict[str, float] = {}
    for name, raw in targets.items():
        value = _finite_number(raw, name)
        servo = by_name[name]
        if not servo.min_deg <= value <= servo.max_deg:
            raise SafetyError(
                f"{name}={value:.1f}° fuera de límites [{servo.min_deg}, {servo.max_deg}]"
            )
        checked[name] = value
    return checked


def validate_trajectory(
    waypoints: Sequence[Mapping[str, object]],
    durations_s: Sequence[object],
    servos: Sequence[ServoConfig],
) -> None:
    """Validate each waypoint and the joint speed between consecutive waypoints."""
    if len(waypoints) < 2 or len(durations_s) != len(waypoints) - 1:
        raise SafetyError("trayectoria mal formada")
    checked = [validate_joint_targets(w, servos) for w in waypoints]
    by_name = {s.name: s for s in servos}
    for i, raw_dt in enumerate(durations_s):
        dt = _finite_number(raw_dt, "duración")
        if dt <= 0:
            raise SafetyError("duración de tramo no positiva")
        for name, servo in by_name.items():
            speed = abs(checked[i + 1][name] - checked[i][name]) / dt
            if speed > servo.max_speed_dps:
                raise SafetyError(f"{name}: velocidad {speed:.0f}°/s > {servo.max_speed_dps}")


def validate_workspace(point_mm: Sequence[object], workspace: WorkspaceConfig | None) -> None:
    if workspace is None:
        raise SafetyError("espacio de trabajo sin configurar")
    if len(point_mm) != 3:
        raise SafetyError("se esperan coordenadas x, y, z")
    x, y, z = (_finite_number(v, "coordenada") for v in point_mm)
    if not (
        workspace.x_min_mm <= x <= workspace.x_max_mm
        and workspace.y_min_mm <= y <= workspace.y_max_mm
        and workspace.z_min_mm <= z <= workspace.z_max_mm
    ):
        raise SafetyError(f"punto ({x:.0f}, {y:.0f}, {z:.0f}) mm fuera del espacio de trabajo")


def validate_scene_fresh(scene: Scene, max_age_s: float, now: float | None = None) -> None:
    age = scene.age(now)
    if age > max_age_s:
        raise SafetyError(f"escena caducada ({age:.1f} s > {max_age_s:.1f} s); vuelve a observar")
    if age < -1.0:
        raise SafetyError("marca de tiempo de escena en el futuro")
    if scene.tracking_ambiguous:
        raise SafetyError("seguimiento ambiguo de objetos; vuelve a observar")
