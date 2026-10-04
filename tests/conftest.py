"""Shared fixtures. Default tests use no network, weights, hardware or services."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from clawde.app import ClawdeApp
from clawde.brain.interpreter import Interpreter
from clawde.config import Settings, load_settings
from clawde.contracts import Scene
from clawde.motion.planner import Planner
from clawde.robot.simulator import SimulatedArm
from clawde.vision.scene import DemoSceneProvider, StaticSceneProvider

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--integration", action="store_true", help="ejecutar pruebas reales opt-in")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="prueba de integración: usa --integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def fixture_scene() -> Scene:
    data = json.loads((FIXTURES / "scene.json").read_text())
    now = time.time()
    data["timestamp"] = now
    for obj in data["objects"]:
        obj["timestamp"] = now
    return Scene.model_validate(data)


class FakeBrain:
    """Stands in for OllamaBrain: returns scripted replies, counts calls."""

    def __init__(self, replies: list[Any] | None = None, model: str = "fake:1b") -> None:
        from clawde.config import OllamaConfig

        self.replies = list(replies or [])
        self.calls: list[str] = []
        self.config = OllamaConfig(model=model)

    def chat_json(self, system: str, user: str, schema: dict) -> str:
        self.calls.append(user)
        if not self.replies:
            raise AssertionError("Ollama no debería haberse llamado")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply()
        return reply


@pytest.fixture
def fake_brain_cls() -> type[FakeBrain]:
    return FakeBrain


def make_app(
    settings: Settings,
    scene_provider: Any = None,
    interpreter: Interpreter | None = None,
    arm: SimulatedArm | None = None,
    real: bool = False,
) -> ClawdeApp:
    scene_provider = scene_provider or DemoSceneProvider(settings.robot.demo_profile)
    return ClawdeApp(
        settings,
        scene_provider,
        arm or SimulatedArm(step_delay_s=0.0),
        interpreter or Interpreter("rules"),
        Planner(settings.robot, settings.app.scene_max_age_s, real=real),
    )


@pytest.fixture
def app_factory(settings: Settings):  # noqa: ANN201
    created: list[ClawdeApp] = []

    def factory(**kwargs: Any) -> ClawdeApp:
        app = make_app(settings, **kwargs)
        created.append(app)
        return app

    yield factory
    for app in created:
        app.shutdown()


@pytest.fixture
def static_scene(fixture_scene: Scene) -> StaticSceneProvider:
    return StaticSceneProvider(fixture_scene)
