import json
import math

import pytest
from pydantic import ValidationError

from clawde.contracts import (
    ActionKind,
    DetectedObject,
    PickPlaceAction,
    Scene,
    llm_action_schema,
    parse_llm_action,
)
from clawde.errors import InterpreterError


@pytest.mark.parametrize(
    "raw",
    [
        '{"action": "fly"}',
        '{"action": "shell", "command": "rm -rf /"}',
        '{"action": "observe", "extra": 1}',
        '{"action": "pick_place", "object_id": "red_1"}',
        '{"action": "pick_place", "object_id": "red_1", "destination_id": "left", "angles": [1]}',
        '{"action": "gripper", "state": "half"}',
        '{"action": "gripper", "state": "open", "pin": 9}',
        '{"action": "clarify", "question": ""}',
        '{"action": "rearm"}',
        "[]",
        "null",
        "not json",
        '{"action": "pick_place", "object_id": "../etc", "destination_id": "left"}',
        '{"action": "pick_place", "object_id": "RED_1", "destination_id": "left"}',
        '{"action": "pick_place", "object_id": "red 1; ls", "destination_id": "left"}',
    ],
)
def test_invalid_llm_replies_are_rejected(raw: str) -> None:
    with pytest.raises(InterpreterError):
        parse_llm_action(raw)


def test_valid_pick_place_parses() -> None:
    action = parse_llm_action(
        json.dumps({"action": "pick_place", "object_id": "red_1", "destination_id": "left"})
    )
    assert isinstance(action, PickPlaceAction)
    assert action.action == ActionKind.PICK_PLACE


def test_schema_offers_only_allowed_actions() -> None:
    schema = json.dumps(llm_action_schema())
    for kind in ["observe", "status", "home", "gripper", "pick_place", "stop", "clarify", "exit"]:
        assert f'"{kind}"' in schema
    assert "rearm" not in schema
    assert '"additionalProperties": false' in schema


def test_scene_rejects_non_finite_positions() -> None:
    with pytest.raises(ValidationError):
        DetectedObject(
            id="red_1",
            color="red",
            center_px=(math.nan, 1.0),
            bbox_px=(0, 0, 1, 1),
            area_px=1.0,
            timestamp=0.0,
        )


def test_scene_lookup(fixture_scene: Scene) -> None:
    assert fixture_scene.object("red_2").zone == "right"
    assert fixture_scene.object("green_9") is None
    assert [o.id for o in fixture_scene.by_color("red")] == ["red_1", "red_2"]
    assert fixture_scene.destination("left").label == "izquierda"
    prompt = fixture_scene.to_prompt_dict()
    assert set(prompt) == {"objects", "destinations"}
