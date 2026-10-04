"""Camera capture (OpenCV/AVFoundation) and an optional labelled preview."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from clawde.errors import DeviceError, require


class Camera:
    """Context-managed webcam. The device is only opened by `open()`."""

    def __init__(self, index: int, width: int, height: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Any = None

    def open(self) -> Camera:
        cv2: Any = require("cv2", "vision")
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            raise DeviceError(
                f"No se pudo abrir la cámara {self.index}. Comprueba la conexión USB y el "
                "permiso de Cámara para Terminal/Zed en Ajustes del Sistema > Privacidad."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return self

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def read(self) -> Any:
        if self._cap is None:
            raise DeviceError("la cámara no está abierta")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise DeviceError("la cámara no devolvió imagen")
        return frame

    def frame_size(self) -> tuple[int, int]:
        frame = self.read()
        return int(frame.shape[1]), int(frame.shape[0])

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


def draw_labels(frame: Any, labelled: list[tuple[str, Any]]) -> Any:
    cv2: Any = require("cv2", "vision")
    colors = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 200, 0)}
    for object_id, det in labelled:
        x, y, w, h = det.bbox_px
        bgr = colors.get(det.color, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        cv2.putText(frame, object_id, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
    return frame


def save_capture(frame: Any, directory: Path, prefix: str = "capture") -> Path:
    cv2: Any = require("cv2", "vision")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise DeviceError(f"no se pudo guardar {path}")
    return path


def destroy_windows() -> None:
    try:
        cv2: Any = require("cv2", "vision")
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:  # noqa: S110 - best effort cleanup
        pass
