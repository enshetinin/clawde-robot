import json
import threading

import pytest
from pydantic import ValidationError

from clawde.brain.interpreter import Interpreter, interpret_rules, is_stop_command, normalize
from clawde.brain.ollama_client import OllamaBrain
from clawde.config import OllamaConfig
from clawde.contracts import ActionKind, GripperState
from clawde.errors import InterpreterError, Stopped
from clawde.vision.scene import DemoSceneProvider


@pytest.fixture
def demo_scene(settings):
    return DemoSceneProvider(settings.robot.demo_profile).get_scene()


def pick(obj: str, dest: str) -> str:
    return json.dumps({"action": "pick_place", "object_id": obj, "destination_id": dest})


def test_normalize_accents_case_punctuation_and_wake_word() -> None:
    assert normalize("Clawde, ¿Qué VES?") == "que ves"
    assert normalize("¡Ábrela!  Pinza...") == "abrela pinza"


@pytest.mark.parametrize(
    "text", ["Para", "¡ALTO!", "detente", "Stop.", "Clawde, para ya", "para, para"]
)
def test_explicit_stop_words(text: str) -> None:
    assert is_stop_command(text)


@pytest.mark.parametrize(
    "text",
    ["no pares", "No te detengas", "coge el rojo para ponerlo a la izquierda", "", "parada"],
)
def test_not_stop(text: str) -> None:
    assert not is_stop_command(text)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Clawde, ¿qué ves?", ActionKind.OBSERVE),
        ("Vuelve a casa", ActionKind.HOME),
        ("Estado", ActionKind.STATUS),
        ("Salir", ActionKind.EXIT),
        ("Para", ActionKind.STOP),
        ("rearmar", ActionKind.REARM),
    ],
)
def test_minimum_phrases(demo_scene, text, kind) -> None:
    assert interpret_rules(text, demo_scene).action.action == kind


def test_gripper_open_close(demo_scene) -> None:
    assert interpret_rules("Abre la pinza", demo_scene).action.state == GripperState.OPEN
    assert interpret_rules("cierra la pinza", demo_scene).action.state == GripperState.CLOSE


@pytest.mark.parametrize("text", ["No cierres la pinza", "no abras la pinza", "no vuelvas a casa"])
def test_negated_orders_do_nothing(demo_scene, text) -> None:
    outcome = interpret_rules(text, demo_scene)
    assert outcome.final
    assert outcome.action.action == ActionKind.CLARIFY


def test_pick_place_on_existing_id(demo_scene) -> None:
    action = interpret_rules("Coge el objeto rojo y ponlo a la izquierda", demo_scene).action
    assert action.action == ActionKind.PICK_PLACE
    assert (action.object_id, action.destination_id) == ("red_1", "left")


def test_two_objects_same_color_ask_for_clarification(fixture_scene) -> None:
    action = interpret_rules("coge el objeto rojo y ponlo en el centro", fixture_scene).action
    assert action.action == ActionKind.CLARIFY
    assert "red_1" in action.question and "red_2" in action.question


def test_zone_qualifier_disambiguates(fixture_scene) -> None:
    action = interpret_rules(
        "coge el rojo de la derecha y ponlo a la izquierda", fixture_scene
    ).action
    assert (action.object_id, action.destination_id) == ("red_2", "left")
    action = interpret_rules("coge el rojo 1 y llévalo al centro", fixture_scene).action
    assert (action.object_id, action.destination_id) == ("red_1", "center")


@pytest.mark.parametrize(
    "text",
    [
        "coge el objeto verde y ponlo a la izquierda",  # no green object
        "coge el objeto amarillo y ponlo a la izquierda",  # unknown colour
        "coge el objeto azul",  # no destination
    ],
)
def test_missing_object_or_destination_never_invents_ids(fixture_scene, text) -> None:
    assert interpret_rules(text, fixture_scene).action.action == ActionKind.CLARIFY


# --------------------------------------------------------------- Ollama path


def test_rules_mode_unknown_text_clarifies(demo_scene) -> None:
    result = Interpreter("rules").interpret("cuéntame un chiste", demo_scene)
    assert result.action.action == ActionKind.CLARIFY
    assert result.source == "rules"


def test_stop_never_reaches_ollama(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([])  # raises if called
    result = Interpreter("ollama", brain).interpret("¡Alto!", demo_scene)
    assert result.action.action == ActionKind.STOP
    assert brain.calls == []


def test_hybrid_uses_rules_when_confident(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([])
    result = Interpreter("hybrid", brain).interpret("abre la pinza", demo_scene)
    assert result.source == "rules" and brain.calls == []


def test_ollama_valid_action(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("red_1", "right")])
    result = Interpreter("ollama", brain).interpret(
        "lleva la pieza roja hacia la derecha", demo_scene
    )
    assert result.source == "ollama"
    assert result.action.object_id == "red_1"
    assert "red_1" in brain.calls[0]  # scene included in prompt


def test_invalid_json_bounded_retry_then_no_action(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls(["{bad", '{"action": "dance"}', pick("red_1", "left")])
    with pytest.raises(InterpreterError):
        Interpreter("ollama", brain, max_retries=1).interpret(
            "haz algo raro con el rojo", demo_scene
        )
    assert len(brain.calls) == 2  # one try + one retry, never a loop


def test_ollama_error_on_motion_has_no_silent_fallback(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([InterpreterError("timeout"), InterpreterError("timeout")])
    with pytest.raises(InterpreterError):
        Interpreter("ollama", brain).interpret(
            "coge el objeto rojo y ponlo a la izquierda", demo_scene
        )


def test_ollama_error_on_query_falls_back_to_rules_with_note(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([InterpreterError("timeout"), InterpreterError("timeout")])
    result = Interpreter("ollama", brain).interpret("¿qué ves?", demo_scene)
    assert result.action.action == ActionKind.OBSERVE
    assert result.source == "rules" and "Ollama" in result.note


def test_ollama_invented_id_rejected(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("green_7", "left"), pick("green_7", "left")])
    result = Interpreter("ollama", brain).interpret(
        "coge el verde y déjalo a la izquierda", demo_scene
    )
    assert result.action.action == ActionKind.CLARIFY  # rule clarification, never green_7
    assert "no existe" in result.note


def test_ollama_invented_id_without_rule_fallback_raises(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("green_7", "left"), pick("green_7", "left")])
    with pytest.raises(InterpreterError, match="no existe"):
        Interpreter("ollama", brain).interpret("haz lo de antes con la pieza", demo_scene)


def test_ollama_unknown_destination_rejected(demo_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("red_1", "kitchen"), pick("red_1", "kitchen")])
    result = Interpreter("ollama", brain).interpret("lleva el rojo a la cocina", demo_scene)
    assert result.action.action == ActionKind.CLARIFY


def test_ollama_ambiguous_choice_becomes_clarify(fixture_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("red_1", "center")])
    result = Interpreter("ollama", brain).interpret("el rojo, al centro", fixture_scene)
    assert result.action.action == ActionKind.CLARIFY


def test_ollama_wrong_color_rejected(fixture_scene, fake_brain_cls) -> None:
    brain = fake_brain_cls([pick("red_1", "center"), pick("red_1", "center")])
    with pytest.raises(InterpreterError):
        Interpreter("ollama", brain).interpret("el azul, al centro por favor", fixture_scene)


def test_cancelled_interpretation_raises_stopped(demo_scene, fake_brain_cls) -> None:
    cancel = threading.Event()

    def reply() -> str:
        cancel.set()  # STOP arrives while the model is "thinking"
        return pick("red_1", "left")

    brain = fake_brain_cls([reply])
    with pytest.raises(Stopped):
        Interpreter("ollama", brain).interpret("haz algo con el rojo", demo_scene, cancel)


# --------------------------------------------------------- Ollama config/client


@pytest.mark.parametrize(
    "host", ["http://192.168.1.10:11434", "https://ollama.com", "http://example.com"]
)
def test_only_loopback_hosts(host: str) -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(host=host)


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3:cloud", ""])
def test_cloud_models_rejected(model: str) -> None:
    with pytest.raises(ValidationError):
        OllamaConfig(model=model)


def test_probe_lists_local_models_only() -> None:
    class Model:
        def __init__(self, name: str) -> None:
            self.model = name

    class Listing:
        models = [Model("qwen3.5:4b"), Model("gpt-oss:120b-cloud")]

    class Client:
        def list(self) -> Listing:
            return Listing()

    probe = OllamaBrain(OllamaConfig(), client=Client()).probe()
    assert probe.reachable and probe.models == ("qwen3.5:4b",)
    assert probe.has_model("qwen3.5:4b")


def test_probe_unreachable_is_reported() -> None:
    class Client:
        def list(self) -> None:
            raise ConnectionError("refused")

    probe = OllamaBrain(OllamaConfig(), client=Client()).probe()
    assert not probe.reachable and "refused" in probe.error


def test_chat_uses_schema_low_temperature_and_no_cloud() -> None:
    captured = {}

    class Message:
        content = '{"action": "status"}'

    class Response:
        message = Message()

    class Client:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return Response()

    brain = OllamaBrain(OllamaConfig(), client=Client())
    assert brain.chat_json("s", "u", {"type": "object"}) == '{"action": "status"}'
    assert captured["format"] == {"type": "object"}
    assert captured["options"]["temperature"] == 0.0
    assert captured["model"] == "qwen3.5:4b"
