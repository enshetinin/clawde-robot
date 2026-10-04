"""Detection, tracking and calibration on synthetic, reproducible images."""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from clawde.errors import DeviceError  # noqa: E402
from clawde.vision.calibration import (  # noqa: E402
    CalibrationError,
    TableCalibration,
    fit_homography,
)
from clawde.vision.detector import ColorDetector, Detection, ObjectTracker, zone_for_x  # noqa: E402
from clawde.vision.scene import CameraSceneProvider  # noqa: E402


@pytest.fixture
def detector(settings) -> ColorDetector:
    v = settings.vision
    return ColorDetector(v.colors, v.min_area_px, v.max_area_px, v.morph_kernel)


def synthetic_frame():
    frame = np.full((480, 640, 3), 200, np.uint8)  # light grey table
    cv2.rectangle(frame, (40, 200), (100, 260), (0, 0, 255), -1)  # red, hue ~0
    cv2.rectangle(frame, (300, 300), (360, 360), (40, 0, 255), -1)  # red, hue ~175
    cv2.rectangle(frame, (500, 100), (570, 170), (255, 0, 0), -1)  # blue
    cv2.circle(frame, (320, 120), 30, (0, 200, 0), -1)  # green
    cv2.rectangle(frame, (600, 400), (604, 404), (0, 0, 255), -1)  # tiny red noise
    return frame


def test_detects_colours_including_both_red_bands(detector) -> None:
    detections = detector.detect(synthetic_frame())
    by_color: dict[str, list[Detection]] = {}
    for d in detections:
        by_color.setdefault(d.color, []).append(d)
    assert len(by_color["red"]) == 2  # noise filtered by area
    assert len(by_color["blue"]) == 1 and len(by_color["green"]) == 1
    red_low = min(by_color["red"], key=lambda d: d.center_px[0])
    assert math.dist(red_low.center_px, (70, 230)) < 2
    x, y, w, h = red_low.bbox_px
    assert (x, y) == (40, 200) and abs(w - 61) <= 1 and abs(h - 61) <= 1


def test_empty_table_has_no_detections(detector) -> None:
    assert detector.detect(np.full((480, 640, 3), 200, np.uint8)) == []


def det(color: str, x: float, y: float) -> Detection:
    return Detection(color, (x, y), (int(x) - 5, int(y) - 5, 10, 10), 100.0)


def test_tracker_keeps_ids_stable_when_objects_move_a_little() -> None:
    tracker = ObjectTracker(max_distance_px=50)
    first, amb = tracker.update([det("red", 100, 100), det("red", 400, 100), det("blue", 250, 300)])
    ids = {d.center_px[0]: i for i, d in first}
    assert not amb and set(ids.values()) == {"red_1", "red_2", "blue_1"}
    second, amb = tracker.update([det("red", 410, 105), det("red", 110, 95), det("blue", 255, 300)])
    moved = {i: d.center_px[0] for i, d in second}
    assert not amb
    assert moved["red_1"] == 110 and moved["red_2"] == 410 and moved["blue_1"] == 255


def test_tracker_new_id_when_far_and_flags_ambiguity() -> None:
    tracker = ObjectTracker(max_distance_px=50)
    tracker.update([det("red", 100, 100)])
    out, _ = tracker.update([det("red", 400, 400)])
    assert out[0][0] == "red_2"
    tracker = ObjectTracker(max_distance_px=50)
    tracker.update([det("red", 100, 100)])
    _, amb = tracker.update([det("red", 110, 100), det("red", 90, 100)])
    assert amb


def test_zones() -> None:
    assert [zone_for_x(x, 900) for x in (10, 450, 890)] == ["left", "center", "right"]


class FakeCamera:
    def __init__(self, frame=None, fail=False):
        self.frame, self.fail, self.closed = frame, fail, False

    def read(self):
        if self.fail:
            raise DeviceError("sin imagen")
        return self.frame

    def close(self):
        self.closed = True


def test_camera_scene_provider_builds_scene(detector) -> None:
    camera = FakeCamera(synthetic_frame())
    provider = CameraSceneProvider(camera, detector, ObjectTracker())
    scene = provider.get_scene()
    assert scene.source == "camera" and not scene.calibrated
    assert {o.color for o in scene.objects} == {"red", "blue", "green"}
    assert all(o.position_mm is None for o in scene.objects)
    assert provider.apply_pick_place("red_1", "left") is False  # real scene never altered
    provider.close()
    assert camera.closed


def test_camera_command_releases_camera_on_failure(settings, monkeypatch) -> None:
    import argparse

    import clawde.vision.camera as camera_module
    from clawde.cli import cmd_camera

    instances = []

    class BrokenCamera(FakeCamera):
        def __init__(self, *a):
            super().__init__(fail=True)
            instances.append(self)

        def open(self):
            return self

    monkeypatch.setattr(camera_module, "Camera", BrokenCamera)
    args = argparse.Namespace(index=0, seconds=0.1, no_preview=True, save=False)
    with pytest.raises(DeviceError):
        cmd_camera(args, settings)
    assert instances[0].closed


# ---------------------------------------------------------------- calibration

PX = [(100.0, 100.0), (500.0, 110.0), (520.0, 400.0), (90.0, 390.0)]
MM = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]


def test_homography_fits_and_maps() -> None:
    cal = TableCalibration.from_points(PX, MM, (640, 480), camera_index=0, max_rms_mm=1.0)
    assert cal.rms_error_mm < 1e-6
    x, y = cal.pixel_to_mm(100, 100)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    x, y = cal.pixel_to_mm(520, 400)
    assert abs(x - 400) < 1e-6 and abs(y - 300) < 1e-6


@pytest.mark.parametrize(
    ("px", "mm"),
    [
        (PX[:3], MM[:3]),  # too few
        ([(0, 0), (100, 0), (200, 0), (300, 0)], MM),  # collinear
        ([(0, 0), (0, 0), (100, 100), (0, 100)], MM),  # duplicate
        (PX, [(0, 0), (1, 1), (2, 2), (3, 3)]),  # collinear world
        ([(0, 0), (100, 0), (math.nan, 1), (0, 100)], MM),
    ],
)
def test_degenerate_correspondences_rejected(px, mm) -> None:
    with pytest.raises(CalibrationError):
        fit_homography(px, mm)


def test_bad_fit_rejected() -> None:
    noisy_mm = MM + [(200.0, 400.0)]  # inconsistent 5th point
    with pytest.raises(CalibrationError, match="error de ajuste"):
        TableCalibration.from_points(PX + [(300.0, 250.0)], noisy_mm, (640, 480), 0, max_rms_mm=1.0)


def test_save_load_and_resolution_check(tmp_path) -> None:
    cal = TableCalibration.from_points(PX, MM, (640, 480), camera_index=1, max_rms_mm=1.0)
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = TableCalibration.load(path)
    assert loaded.camera_index == 1 and loaded.created_at
    loaded.ensure_resolution(640, 480)
    with pytest.raises(CalibrationError, match="recalibra"):
        loaded.ensure_resolution(1280, 720)
    loaded.ensure_resolution(1280, 720, allow_mismatch=True)
