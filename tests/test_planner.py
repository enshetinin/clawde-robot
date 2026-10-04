import time

import pytest

from clawde.contracts import ActionKind, PickPlaceAction
from clawde.errors import ActionRejected, Stopped
from clawde.motion.planner import Planner
from clawde.robot.base import ArmState, CancelToken
from clawde.robot.simulator import SimulatedArm
from clawde.vision.scene import DemoSceneProvider


def pp(obj: str, dest: str) -> PickPlaceAction:
    return PickPlaceAction(action=ActionKind.PICK_PLACE, object_id=obj, destination_id=dest)


@pytest.fixture
def planner(settings) -> Planner:
    return Planner(settings.robot, scene_max_age_s=3.0)


def test_pick_place_sequence_is_explicit(planner, fixture_scene) -> None:
    plan = planner.plan(pp("blue_1", "left"), fixture_scene, SimulatedArm().status())
    targets = [s.target for s in plan.steps]
    i_approach = targets.index("above:blue_1")
    i_grip = targets.index("close")
    i_transfer = targets.index("above:left")
    i_release = len(targets) - 1 - targets[::-1].index("open")
    assert i_approach < targets.index("at:blue_1") < i_grip < i_transfer < i_release


def test_rejections(planner, fixture_scene) -> None:
    status = SimulatedArm().status()
    with pytest.raises(ActionRejected):
        planner.plan(pp("green_1", "left"), fixture_scene, status)
    with pytest.raises(ActionRejected, match="ya está"):
        planner.plan(pp("red_1", "left"), fixture_scene, status)
    status.holding = "red_2"
    with pytest.raises(ActionRejected, match="sostiene"):
        planner.plan(pp("blue_1", "left"), fixture_scene, status)


def test_simulator_runs_plan_and_tracks_state(planner, fixture_scene) -> None:
    arm = SimulatedArm(step_delay_s=0.0)
    plan = planner.plan(pp("blue_1", "right"), fixture_scene, arm.status())
    arm.execute(plan, CancelToken())
    status = arm.status()
    assert status.state == ArmState.IDLE
    assert status.location == "above:right" and status.holding is None
    assert "gripper:close" in arm.history


def test_cancelled_token_stops_simulator(planner, fixture_scene) -> None:
    arm = SimulatedArm(step_delay_s=0.0)
    token = CancelToken()
    token.cancel()
    with pytest.raises(Stopped):
        arm.execute(planner.plan(pp("blue_1", "right"), fixture_scene, arm.status()), token)
    assert arm.state == ArmState.STOPPED


def test_demo_scene_updates_only_after_completed_pick_place(app_factory, settings) -> None:
    scene = DemoSceneProvider(settings.robot.demo_profile)
    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(scene_provider=scene, arm=arm)
    reply = app.handle("Coge el objeto rojo y ponlo a la izquierda")
    assert reply.ok, reply.text
    assert scene.get_scene().object("red_1").zone == "left"
    assert arm.status().location == "above:left"
    # Second pick of the same object to the same zone is rejected, scene unchanged.
    reply = app.handle("coge el objeto rojo y ponlo a la izquierda")
    assert not reply.ok
    assert scene.get_scene().object("red_1").zone == "left"


def test_demo_observe_and_status_report(app_factory) -> None:
    app = app_factory()
    observe = app.handle("Clawde, ¿qué ves?")
    assert "FICTICIA" in observe.text and "red_1" in observe.text and "blue_2" in observe.text
    status = app.handle("estado")
    assert "simulación" in status.text and "reglas" in status.text and "idle" in status.text


def test_home_and_gripper_in_simulation(app_factory) -> None:
    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(arm=arm)
    assert app.handle("cierra la pinza").ok
    assert app.handle("Vuelve a casa").ok
    assert arm.location == "home" and arm.gripper.value == "close"


def test_exit_action(app_factory) -> None:
    assert app_factory().handle("salir").exit


def test_worker_processes_queue(app_factory) -> None:
    replies = []
    app = app_factory()
    app.on_reply = replies.append
    app.start()
    app.submit("abre la pinza")
    deadline = time.time() + 5
    while not replies and time.time() < deadline:
        time.sleep(0.01)
    assert replies and replies[0].ok
