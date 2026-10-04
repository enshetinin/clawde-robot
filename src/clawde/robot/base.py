"""Arm driver interface, cancellation token and observable state."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from clawde.contracts import GripperState
from clawde.errors import Stopped


class ArmState(StrEnum):
    IDLE = "idle"
    MOVING = "moving"
    STOPPED = "stopped"  # latched: needs explicit rearm
    UNKNOWN = "unknown"  # e.g. lost DONE or disconnected
    DISCONNECTED = "disconnected"


class CancelToken:
    """Per-generation cancellation flag. STOP sets it; a new generation gets a new one."""

    def __init__(self, generation: int = 0) -> None:
        self.generation = generation
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def event(self) -> threading.Event:
        return self._event

    def check(self) -> None:
        if self._event.is_set():
            raise Stopped("ejecución cancelada por STOP")

    def wait(self, seconds: float) -> None:
        """Sleep that wakes immediately on cancellation."""
        if self._event.wait(seconds):
            raise Stopped("ejecución cancelada por STOP")


@dataclass
class ArmStatus:
    driver: str
    state: ArmState
    gripper: GripperState | None
    location: str | None
    holding: str | None
    simulated: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlanStep:
    """Symbolic step. Real joint targets require verified kinematics (pending)."""

    kind: str  # "move" | "gripper"
    target: str  # named location (e.g. "above:red_1", "home") or gripper state
    description: str


@dataclass(frozen=True)
class Plan:
    name: str
    steps: tuple[PlanStep, ...]
    object_id: str | None = None
    destination_id: str | None = None


class ArmDriver(ABC):
    """Interface implemented by the simulator and by real drivers."""

    name: str = "base"
    simulated: bool = True

    @abstractmethod
    def status(self) -> ArmStatus: ...

    @abstractmethod
    def execute(self, plan: Plan, token: CancelToken) -> None:
        """Run a plan step by step, honouring the token. Raises Stopped on cancel."""

    @abstractmethod
    def stop(self) -> None:
        """Immediate stop: cancel trajectory and latch STOPPED."""

    @abstractmethod
    def rearm(self) -> None:
        """Explicit re-arm after a stop. Must revalidate state."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources. Never moves the arm."""
