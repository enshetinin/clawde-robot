"""Opt-in integration with the real local Ollama (`pytest --integration`). No motion."""

import pytest

from clawde.brain.interpreter import Interpreter
from clawde.brain.ollama_client import OllamaBrain
from clawde.contracts import ActionKind
from clawde.vision.scene import DemoSceneProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def live_interpreter(settings) -> Interpreter:
    brain = OllamaBrain(settings.app.ollama)
    probe = brain.probe()
    if not probe.reachable or not probe.has_model(settings.app.ollama.model):
        pytest.skip(f"Ollama local o modelo {settings.app.ollama.model} no disponible")
    return Interpreter("ollama", brain)


def test_live_pick_place_uses_existing_ids(settings, live_interpreter) -> None:
    scene = DemoSceneProvider(settings.robot.demo_profile).get_scene()
    result = live_interpreter.interpret("coge el objeto rojo y ponlo a la izquierda", scene)
    assert result.action.action in (ActionKind.PICK_PLACE, ActionKind.CLARIFY)
    if result.action.action == ActionKind.PICK_PLACE:
        assert result.action.object_id == "red_1" and result.action.destination_id == "left"


def test_live_ambiguous_blue_never_moves(settings, live_interpreter) -> None:
    scene = DemoSceneProvider(settings.robot.demo_profile).get_scene()
    result = live_interpreter.interpret("coge el objeto azul y ponlo en el centro", scene)
    assert result.action.action == ActionKind.CLARIFY
