"""Turn validated actions into explicit, symbolic plans.

Simulation executes the symbolic steps. Real execution would additionally need
verified kinematics, calibration and limits; until then it is refused here,
before anything is sent to the hardware.
"""

from __future__ import annotations

from typing import Any

from clawde.config import RobotConfig
from clawde.contracts import ActionKind, Scene
from clawde.errors import ActionRejected, HardwareNotConfigured
from clawde.motion.kinematics import require_solver
from clawde.motion.limits import validate_scene_fresh
from clawde.robot.base import ArmStatus, Plan, PlanStep


def check_real_motion_ready(robot: RobotConfig, scene: Scene | None = None) -> None:
    """Raise HardwareNotConfigured unless every hardware prerequisite is satisfied."""
    pending = robot.pending_hardware()
    if pending:
        raise HardwareNotConfigured("movimiento real bloqueado. Pendiente: " + "; ".join(pending))
    require_solver(robot.geometry)
    if scene is not None and not scene.calibrated:
        raise HardwareNotConfigured("la cámara no tiene calibración de mesa (homografía)")


class Planner:
    def __init__(self, robot: RobotConfig, scene_max_age_s: float, real: bool = False) -> None:
        self.robot = robot
        self.scene_max_age_s = scene_max_age_s
        self.real = real

    def plan(self, action: Any, scene: Scene, arm: ArmStatus, now: float | None = None) -> Plan:
        if self.real:
            check_real_motion_ready(
                self.robot, scene if action.action == ActionKind.PICK_PLACE else None
            )
        if action.action == ActionKind.HOME:
            return Plan("home", (PlanStep("move", "home", "volver a casa"),))
        if action.action == ActionKind.GRIPPER:
            verb = "abrir" if action.state.value == "open" else "cerrar"
            return Plan("gripper", (PlanStep("gripper", action.state.value, f"{verb} pinza"),))
        if action.action == ActionKind.PICK_PLACE:
            return self._pick_place(action.object_id, action.destination_id, scene, arm, now)
        raise ActionRejected(f"la acción '{action.action}' no genera movimiento")

    def _pick_place(
        self, object_id: str, destination_id: str, scene: Scene, arm: ArmStatus, now: float | None
    ) -> Plan:
        validate_scene_fresh(scene, self.scene_max_age_s, now)
        obj = scene.object(object_id)
        if obj is None:
            raise ActionRejected(f"el objeto '{object_id}' ya no está en la escena")
        dest = scene.destination(destination_id)
        if dest is None:
            raise ActionRejected(f"destino desconocido '{destination_id}'")
        if arm.holding is not None:
            raise ActionRejected(f"la pinza ya sostiene '{arm.holding}'")
        if obj.zone == dest.zone:
            raise ActionRejected(f"'{object_id}' ya está en la zona {dest.label}")
        steps = (
            PlanStep("gripper", "open", "abrir pinza"),
            PlanStep("move", f"above:{object_id}", f"aproximación sobre {object_id}"),
            PlanStep("move", f"at:{object_id}", f"descenso a {object_id}"),
            PlanStep("gripper", "close", "cerrar pinza"),
            PlanStep("move", f"above:{object_id}", "elevar"),
            PlanStep("move", f"above:{destination_id}", f"traslado a {dest.label}"),
            PlanStep("move", f"at:{destination_id}", "descenso"),
            PlanStep("gripper", "open", "liberar"),
            PlanStep("move", f"above:{destination_id}", "retirada"),
        )
        return Plan("pick_place", steps, object_id=object_id, destination_id=destination_id)
