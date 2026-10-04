"""Simulated arm with observable state. Never opens a serial port."""

from __future__ import annotations

import logging
import threading

from clawde.contracts import GripperState
from clawde.errors import Stopped
from clawde.robot.base import ArmDriver, ArmState, ArmStatus, CancelToken, Plan

log = logging.getLogger(__name__)


class SimulatedArm(ArmDriver):
    name = "simulator"
    simulated = True

    def __init__(self, step_delay_s: float = 0.15, hold_position_on_stop: bool = True) -> None:
        self.step_delay_s = step_delay_s
        # Simulator profile: servos are assumed to hold position after STOP.
        self.hold_position_on_stop = hold_position_on_stop
        self._lock = threading.Lock()
        self.state = ArmState.IDLE
        self.gripper = GripperState.OPEN
        self.location: str | None = "home"
        self.holding: str | None = None
        self.history: list[str] = []
        self.stop_count = 0

    def status(self) -> ArmStatus:
        with self._lock:
            notes = ["brazo simulado: no hay movimiento físico"]
            if self.state == ArmState.STOPPED:
                notes.append("detenido: requiere rearme explícito")
            return ArmStatus(
                driver=self.name,
                state=self.state,
                gripper=self.gripper,
                location=self.location,
                holding=self.holding,
                simulated=True,
                notes=notes,
            )

    def execute(self, plan: Plan, token: CancelToken) -> None:
        with self._lock:
            if self.state == ArmState.STOPPED:
                raise Stopped("el brazo simulado está detenido; di «rearmar» primero")
            if token.cancelled:
                self.state = ArmState.STOPPED
                raise Stopped("ejecución cancelada por STOP")
            self.state = ArmState.MOVING
        self.history.append(f"plan:{plan.name}")
        try:
            for step in plan.steps:
                token.check()
                token.wait(self.step_delay_s)
                with self._lock:
                    if self.state == ArmState.STOPPED:
                        raise Stopped("parada durante la trayectoria")
                    self._apply(step.kind, step.target)
                self.history.append(f"{step.kind}:{step.target}")
                log.info("sim step %s %s", step.kind, step.target)
        except Stopped:
            with self._lock:
                self.state = ArmState.STOPPED
            raise
        with self._lock:
            self.state = ArmState.IDLE

    def _apply(self, kind: str, target: str) -> None:
        if kind == "move":
            self.location = target
        elif kind == "gripper":
            self.gripper = GripperState(target)
            if (
                self.gripper == GripperState.CLOSE
                and self.location
                and self.location.startswith("at:")
            ):
                self.holding = self.location.removeprefix("at:")
            elif self.gripper == GripperState.OPEN:
                self.holding = None

    def stop(self) -> None:
        with self._lock:
            self.state = ArmState.STOPPED
            self.stop_count += 1
        self.history.append("stop")
        log.warning("simulator stopped (latched)")

    def rearm(self) -> None:
        with self._lock:
            if self.state != ArmState.STOPPED:
                return
            # Revalidation: in simulation the state is fully known.
            self.state = ArmState.IDLE
        self.history.append("rearm")
