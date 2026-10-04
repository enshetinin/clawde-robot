"""Strict data contracts: actions the brain may request and the scene it sees.

The LLM can only produce one of the actions in `LlmAction`. It never supplies
pins, angles, code or shell commands.
"""

from __future__ import annotations

import json
import math
import time
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from clawde.errors import InterpreterError

ID_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------- actions


class ActionKind(StrEnum):
    OBSERVE = "observe"
    STATUS = "status"
    HOME = "home"
    GRIPPER = "gripper"
    PICK_PLACE = "pick_place"
    STOP = "stop"
    CLARIFY = "clarify"
    EXIT = "exit"
    REARM = "rearm"  # local control only, never offered to the LLM


class GripperState(StrEnum):
    OPEN = "open"
    CLOSE = "close"


class ObserveAction(Strict):
    action: Literal[ActionKind.OBSERVE]


class StatusAction(Strict):
    action: Literal[ActionKind.STATUS]


class HomeAction(Strict):
    action: Literal[ActionKind.HOME]


class GripperAction(Strict):
    action: Literal[ActionKind.GRIPPER]
    state: GripperState


class PickPlaceAction(Strict):
    action: Literal[ActionKind.PICK_PLACE]
    object_id: Annotated[str, Field(pattern=ID_PATTERN)]
    destination_id: Annotated[str, Field(pattern=ID_PATTERN)]


class StopAction(Strict):
    action: Literal[ActionKind.STOP]


class ClarifyAction(Strict):
    action: Literal[ActionKind.CLARIFY]
    question: Annotated[str, Field(min_length=1, max_length=300)]


class ExitAction(Strict):
    action: Literal[ActionKind.EXIT]


class RearmAction(Strict):
    action: Literal[ActionKind.REARM]


LlmAction = Annotated[
    ObserveAction
    | StatusAction
    | HomeAction
    | GripperAction
    | PickPlaceAction
    | StopAction
    | ClarifyAction
    | ExitAction,
    Field(discriminator="action"),
]

Action = Annotated[
    ObserveAction
    | StatusAction
    | HomeAction
    | GripperAction
    | PickPlaceAction
    | StopAction
    | ClarifyAction
    | ExitAction
    | RearmAction,
    Field(discriminator="action"),
]

MOTION_KINDS = frozenset({ActionKind.HOME, ActionKind.GRIPPER, ActionKind.PICK_PLACE})

_llm_adapter: TypeAdapter[Any] = TypeAdapter(LlmAction)


def llm_action_schema() -> dict[str, Any]:
    """JSON schema passed to Ollama `format=` (union of allowed actions)."""
    return _llm_adapter.json_schema()


def parse_llm_action(raw: str) -> Any:
    """Parse and strictly validate an LLM JSON reply. Raises InterpreterError."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InterpreterError(f"respuesta JSON inválida: {exc}") from exc
    if not isinstance(data, dict):
        raise InterpreterError("la respuesta debe ser un objeto JSON")
    try:
        return _llm_adapter.validate_python(data)
    except ValidationError as exc:
        raise InterpreterError(f"acción no válida: {exc.errors()[0]['msg']}") from exc


def is_motion(action: Any) -> bool:
    return action.action in MOTION_KINDS


# ------------------------------------------------------------------------- scene


def _finite(values: tuple[float, ...]) -> tuple[float, ...]:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("valores no finitos")
    return values


class DetectedObject(Strict):
    id: Annotated[str, Field(pattern=ID_PATTERN)]
    color: str
    center_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]  # x, y, w, h
    area_px: float = Field(ge=0)
    timestamp: float
    zone: Literal["left", "center", "right"] | None = None
    position_mm: tuple[float, float] | None = None

    @field_validator("center_px", "position_mm")
    @classmethod
    def _check_finite(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        return None if value is None else _finite(value)


class Destination(Strict):
    id: Annotated[str, Field(pattern=ID_PATTERN)]
    label: str
    zone: Literal["left", "center", "right"]


DEFAULT_DESTINATIONS: tuple[Destination, ...] = (
    Destination(id="left", label="izquierda", zone="left"),
    Destination(id="center", label="centro", zone="center"),
    Destination(id="right", label="derecha", zone="right"),
)


class Scene(Strict):
    source: Literal["demo", "camera", "fixture", "empty"]
    timestamp: float
    objects: tuple[DetectedObject, ...] = ()
    destinations: tuple[Destination, ...] = DEFAULT_DESTINATIONS
    frame_size: tuple[int, int] | None = None
    calibrated: bool = False
    tracking_ambiguous: bool = False

    def object(self, object_id: str) -> DetectedObject | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def destination(self, destination_id: str) -> Destination | None:
        return next((d for d in self.destinations if d.id == destination_id), None)

    def by_color(self, color: str) -> list[DetectedObject]:
        return [o for o in self.objects if o.color == color]

    def age(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.timestamp

    def to_prompt_dict(self) -> dict[str, Any]:
        """Minimal scene view for the LLM prompt."""
        return {
            "objects": [{"id": o.id, "color": o.color, "zone": o.zone} for o in self.objects],
            "destinations": [{"id": d.id, "label": d.label} for d in self.destinations],
        }
