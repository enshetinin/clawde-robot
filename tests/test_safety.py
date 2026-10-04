"""STOP, latching, stale results, real-mode refusal and limit validation."""

import math
import threading
import time

import pytest

from clawde.brain.interpreter import Interpreter
from clawde.cli import build_arm
from clawde.config import (
    GeometryConfig,
    RobotConfig,
    ServoConfig,
    WorkspaceConfig,
)
from clawde.contracts import ActionKind, GripperState
from clawde.errors import HardwareNotConfigured, KinematicsUnavailable, SafetyError
from clawde.motion.kinematics import solve_ik
from clawde.motion.limits import (
    validate_joint_targets,
    validate_scene_fresh,
    validate_trajectory,
    validate_workspace,
)
from clawde.motion.planner import Planner, check_real_motion_ready
from clawde.robot.base import ArmState
from clawde.robot.simulator import SimulatedArm
from clawde.vision.scene import DemoSceneProvider, StaticSceneProvider


def test_stop_needs_no_ollama(app_factory, fake_brain_cls) -> None:
    brain = fake_brain_cls([])  # fails if called
    app = app_factory(interpreter=Interpreter("ollama", brain))
    reply = app.submit("¡para!")
    assert reply.action == "stop"
    assert brain.calls == []
    assert app.stopped
    assert "No es una parada de emergencia física" in reply.text


def test_stop_clears_queue(app_factory) -> None:
    app = app_factory()  # worker not started: jobs stay queued
    assert app.submit("abre la pinza") is None
    assert app.submit("cierra la pinza") is None
    app.submit("alto")
    assert app._queue.qsize() == 0


def test_queue_is_bounded(app_factory, settings) -> None:
    app = app_factory()
    for _ in range(settings.app.queue_maxsize):
        assert app.submit("estado") is None
    reply = app.submit("estado")
    assert reply is not None and not reply.ok and "Cola llena" in reply.text


def test_late_ollama_result_discarded_after_stop(app_factory, fake_brain_cls) -> None:
    release = threading.Event()
    started = threading.Event()

    def slow_reply() -> str:
        started.set()
        release.wait(5)
        return '{"action": "gripper", "state": "close"}'

    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(interpreter=Interpreter("ollama", fake_brain_cls([slow_reply])), arm=arm)
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(r=app._process(app_job(app, "aprieta la mano")))
    )
    worker.start()
    assert started.wait(5)
    app.emergency_stop("teclado")
    release.set()
    worker.join(5)
    assert result["r"] is None  # discarded
    assert arm.gripper == GripperState.OPEN
    assert not any(h.startswith("gripper") for h in arm.history)


def app_job(app, text):
    from clawde.app import Job

    return Job(text, app.generation, "test")


def test_new_order_does_not_rearm(app_factory) -> None:
    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(arm=arm)
    app.handle("para")
    reply = app.handle("cierra la pinza")
    assert not reply.ok and "rearmar" in reply.text
    assert arm.gripper == GripperState.OPEN
    assert app.stopped and arm.state == ArmState.STOPPED
    assert app.handle("rearmar").ok
    assert not app.stopped
    assert app.handle("cierra la pinza").ok
    assert arm.gripper == GripperState.CLOSE


def test_negated_close_does_not_close(app_factory) -> None:
    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(arm=arm)
    reply = app.handle("No cierres la pinza")
    assert reply.action == ActionKind.CLARIFY.value
    assert arm.gripper == GripperState.OPEN and arm.history == []


def test_stop_interrupts_simulated_trajectory(app_factory, settings) -> None:
    arm = SimulatedArm(step_delay_s=0.2)
    scene = DemoSceneProvider(settings.robot.demo_profile)
    app = app_factory(arm=arm, scene_provider=scene)
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(r=app.handle("coge el objeto rojo y ponlo a la izquierda"))
    )
    worker.start()
    time.sleep(0.5)
    app.emergency_stop("teclado")
    worker.join(5)
    assert "interrumpido" in result["r"].text
    assert arm.state == ArmState.STOPPED
    assert scene.get_scene().object("red_1").zone == "center"  # scene unchanged


def test_stale_scene_rejected_before_motion(app_factory, fixture_scene) -> None:
    old = fixture_scene.model_copy(update={"timestamp": time.time() - 60})
    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(arm=arm, scene_provider=StaticSceneProvider(old))
    reply = app.handle("coge el objeto azul y ponlo a la izquierda")
    assert not reply.ok and "caducada" in reply.text
    assert arm.history == []


def test_ambiguous_tracking_rejected(fixture_scene) -> None:
    scene = fixture_scene.model_copy(update={"tracking_ambiguous": True})
    with pytest.raises(SafetyError, match="ambiguo"):
        validate_scene_fresh(scene, 3.0)


# ------------------------------------------------------------------ real mode


def full_robot_config(**overrides) -> RobotConfig:
    data = dict(
        driver="serial",
        hardware_enabled=True,
        calibrated=True,
        board="unknown-board",
        port="/dev/null-test",
        servos=[
            ServoConfig(name="base", pin=1, min_deg=0, max_deg=180, home_deg=90, max_speed_dps=60)
        ],
        geometry=GeometryConfig(kind="adeept-unverified", link_lengths_mm=[100.0]),
        workspace=WorkspaceConfig(
            x_min_mm=-1, x_max_mm=1, y_min_mm=-1, y_max_mm=1, z_min_mm=0, z_max_mm=1
        ),
        hold_position_on_stop=True,
    )
    data.update(overrides)
    return RobotConfig(**data)


def test_real_without_profile_fails_before_opening_port(settings, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(
        "clawde.robot.serial_transport.open_serial_port", lambda *a, **k: opened.append(a)
    )
    with pytest.raises(HardwareNotConfigured) as info:
        build_arm(settings, real=True)
    assert "calibración" in str(info.value) and "placa" in str(info.value)
    assert opened == []


def test_real_with_complete_config_still_needs_verified_kinematics(settings, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(
        "clawde.robot.serial_transport.open_serial_port", lambda *a, **k: opened.append(a)
    )
    real_settings = settings.model_copy(update={"robot": full_robot_config()})
    with pytest.raises(KinematicsUnavailable):
        build_arm(real_settings, real=True)
    assert opened == []


@pytest.mark.parametrize("missing", ["calibrated", "board", "workspace", "hold_position_on_stop"])
def test_each_missing_item_blocks_real_motion(missing) -> None:
    value = False if missing == "calibrated" else None
    with pytest.raises(HardwareNotConfigured):
        check_real_motion_ready(full_robot_config(**{missing: value}))


def test_real_planner_refuses_home_and_gripper(settings, fixture_scene) -> None:
    from clawde.contracts import GripperAction, HomeAction

    planner = Planner(settings.robot, 3.0, real=True)
    status = SimulatedArm().status()
    for action in (HomeAction(action="home"), GripperAction(action="gripper", state="open")):
        with pytest.raises(HardwareNotConfigured):
            planner.plan(action, fixture_scene, status)


def test_kinematics_never_invents_angles() -> None:
    with pytest.raises(KinematicsUnavailable):
        solve_ik(None, (0.0, 100.0, 50.0))
    with pytest.raises(KinematicsUnavailable):
        solve_ik(GeometryConfig(kind="adeept", link_lengths_mm=[100, 100]), (0.0, 100.0, 50.0))


# --------------------------------------------------------------------- limits

SERVOS = [
    ServoConfig(name="base", pin=1, min_deg=0, max_deg=180, home_deg=90, max_speed_dps=60),
    ServoConfig(name="elbow", pin=2, min_deg=20, max_deg=160, home_deg=90, max_speed_dps=60),
]


@pytest.mark.parametrize(
    "targets",
    [
        {"base": math.nan, "elbow": 90},
        {"base": math.inf, "elbow": 90},
        {"base": "90", "elbow": 90},
        {"base": True, "elbow": 90},
        {"base": 200, "elbow": 90},
        {"base": 90},
        {"base": 90, "elbow": 90, "wrist": 10},
    ],
)
def test_joint_targets_rejected(targets) -> None:
    with pytest.raises(SafetyError):
        validate_joint_targets(targets, SERVOS)


def test_joint_limits_unconfigured() -> None:
    with pytest.raises(SafetyError):
        validate_joint_targets({"base": 90}, [])


def test_trajectory_speed_limit() -> None:
    a, b = {"base": 0, "elbow": 90}, {"base": 90, "elbow": 90}
    validate_trajectory([a, b], [2.0], SERVOS)
    with pytest.raises(SafetyError, match="velocidad"):
        validate_trajectory([a, b], [0.5], SERVOS)
    with pytest.raises(SafetyError):
        validate_trajectory([a, b], [math.nan], SERVOS)


def test_workspace() -> None:
    ws = WorkspaceConfig(
        x_min_mm=-100, x_max_mm=100, y_min_mm=0, y_max_mm=200, z_min_mm=0, z_max_mm=150
    )
    validate_workspace((0, 100, 50), ws)
    for point in [(500, 100, 50), (0, 100, math.nan), (0, 100)]:
        with pytest.raises(SafetyError):
            validate_workspace(point, ws)
    with pytest.raises(SafetyError):
        validate_workspace((0, 0, 0), None)
