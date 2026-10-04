"""Scene providers: fictitious demo scene and live camera scene."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from clawde.config import DemoProfile
from clawde.contracts import DetectedObject, Scene
from clawde.vision.calibration import TableCalibration
from clawde.vision.detector import ColorDetector, ObjectTracker, zone_for_x

log = logging.getLogger(__name__)

DEMO_WARNING = "MODO DEMO: escena FICTICIA (los objetos y posiciones no son reales)"
DEMO_FRAME = (1280, 720)
_ZONE_X = {"left": 213.0, "center": 640.0, "right": 1066.0}
_ZONE_MM = {"left": -120.0, "center": 0.0, "right": 120.0}


class SceneProvider(Protocol):
    warning: str | None

    def get_scene(self) -> Scene: ...
    def apply_pick_place(self, object_id: str, destination_id: str) -> bool: ...
    def close(self) -> None: ...


class DemoSceneProvider:
    """Artificial positions, used only with --demo."""

    warning = DEMO_WARNING

    def __init__(self, profile: DemoProfile) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, dict[str, Any]] = {
            o.id: {"color": o.color, "zone": o.zone, "x_mm": o.x_mm, "y_mm": o.y_mm}
            for o in profile.objects
        }

    def get_scene(self) -> Scene:
        now = time.time()
        with self._lock:
            objects = []
            per_zone: dict[str, int] = {}
            for object_id, data in self._objects.items():
                slot = per_zone.get(data["zone"], 0)
                per_zone[data["zone"]] = slot + 1
                cx = _ZONE_X[data["zone"]] + 40.0 * slot
                cy = 400.0 + 30.0 * slot
                objects.append(
                    DetectedObject(
                        id=object_id,
                        color=data["color"],
                        center_px=(cx, cy),
                        bbox_px=(int(cx) - 30, int(cy) - 30, 60, 60),
                        area_px=3600.0,
                        timestamp=now,
                        zone=data["zone"],
                        position_mm=(data["x_mm"], data["y_mm"]),
                    )
                )
        return Scene(source="demo", timestamp=now, objects=tuple(objects), frame_size=DEMO_FRAME)

    def apply_pick_place(self, object_id: str, destination_id: str) -> bool:
        """Update the fictitious scene after a COMPLETED simulated pick/place."""
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None or destination_id not in _ZONE_X:
                return False
            obj["zone"] = destination_id
            obj["x_mm"] = _ZONE_MM[destination_id]
            return True

    def close(self) -> None:
        pass


class CameraSceneProvider:
    """Builds scenes from live frames. The physical scene is never altered by simulation."""

    warning: str | None = None

    def __init__(
        self,
        camera: Any,
        detector: ColorDetector,
        tracker: ObjectTracker,
        calibration: TableCalibration | None = None,
        save_dir: Path | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.calibration = calibration
        self.save_dir = save_dir
        self._lock = threading.Lock()

    def get_scene(self) -> Scene:
        with self._lock:
            frame = self.camera.read()
            now = time.time()
            height, width = int(frame.shape[0]), int(frame.shape[1])
            calibration = self.calibration
            if calibration is not None:
                calibration.ensure_resolution(width, height)
            labelled, ambiguous = self.tracker.update(self.detector.detect(frame))
            objects = []
            for object_id, det in labelled:
                position = calibration.pixel_to_mm(*det.center_px) if calibration else None
                objects.append(
                    DetectedObject(
                        id=object_id,
                        color=det.color,
                        center_px=det.center_px,
                        bbox_px=det.bbox_px,
                        area_px=det.area_px,
                        timestamp=now,
                        zone=zone_for_x(det.center_px[0], width),
                        position_mm=position,
                    )
                )
            if self.save_dir is not None:
                from clawde.vision.camera import draw_labels, save_capture

                path = save_capture(draw_labels(frame.copy(), labelled), self.save_dir)
                log.info("captura guardada en %s", path)
            return Scene(
                source="camera",
                timestamp=now,
                objects=tuple(objects),
                frame_size=(width, height),
                calibrated=calibration is not None,
                tracking_ambiguous=ambiguous,
            )

    def apply_pick_place(self, object_id: str, destination_id: str) -> bool:
        return False  # real objects did not move: the arm is simulated

    def close(self) -> None:
        self.camera.close()


class StaticSceneProvider:
    """Fixed scene (tests and fixtures)."""

    warning: str | None = None

    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    def get_scene(self) -> Scene:
        return self.scene

    def apply_pick_place(self, object_id: str, destination_id: str) -> bool:
        return False

    def close(self) -> None:
        pass


def load_scene_file(path: Path) -> Scene:
    return Scene.model_validate(json.loads(path.read_text(encoding="utf-8")))
