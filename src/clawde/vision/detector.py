"""HSV colour segmentation and simple spatial tracking.

This finds coloured blobs; it does not recognise object shapes (a red blob is
not necessarily a cube).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from clawde.config import HSVRange
from clawde.errors import require


@dataclass(frozen=True)
class Detection:
    color: str
    center_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]
    area_px: float


class ColorDetector:
    def __init__(
        self,
        colors: dict[str, list[HSVRange]],
        min_area_px: int = 400,
        max_area_px: int = 200000,
        morph_kernel: int = 5,
    ) -> None:
        if not colors:
            raise ValueError("no hay rangos de color configurados")
        self.colors = colors
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.morph_kernel = morph_kernel

    def masks(self, frame_bgr: Any) -> dict[str, Any]:
        cv2: Any = require("cv2", "vision")
        np: Any = require("numpy", "vision")
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        result = {}
        for color, ranges in self.colors.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for rng in ranges:  # red uses two hue bands
                band = cv2.inRange(
                    hsv, np.array(rng.lower, np.uint8), np.array(rng.upper, np.uint8)
                )
                mask = cv2.bitwise_or(mask, band)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            result[color] = mask
        return result

    def detect(self, frame_bgr: Any) -> list[Detection]:
        cv2: Any = require("cv2", "vision")
        detections: list[Detection] = []
        for color, mask in self.masks(frame_bgr).items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not self.min_area_px <= area <= self.max_area_px:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(Detection(color, (cx, cy), (x, y, w, h), area))
        detections.sort(key=lambda d: (d.color, d.center_px[0], d.center_px[1]))
        return detections


class ObjectTracker:
    """Keep IDs stable across frames by bounded nearest-neighbour association."""

    def __init__(self, max_distance_px: float = 60.0) -> None:
        self.max_distance_px = max_distance_px
        self._tracks: dict[str, tuple[str, tuple[float, float]]] = {}
        self._counters: dict[str, int] = {}

    def _new_id(self, color: str) -> str:
        self._counters[color] = self._counters.get(color, 0) + 1
        return f"{color}_{self._counters[color]}"

    def update(self, detections: list[Detection]) -> tuple[list[tuple[str, Detection]], bool]:
        """Return (id, detection) pairs and whether association was ambiguous."""
        ambiguous = False
        assigned: list[tuple[str, Detection]] = []
        used_tracks: set[str] = set()
        pairs: list[tuple[float, int, str]] = []
        for i, det in enumerate(detections):
            near = []
            for track_id, (color, center) in self._tracks.items():
                if color != det.color:
                    continue
                dist = math.dist(center, det.center_px)
                if dist <= self.max_distance_px:
                    near.append(track_id)
                    pairs.append((dist, i, track_id))
            if len(near) > 1:
                ambiguous = True
        track_hits: dict[str, int] = {}
        for _, _, track_id in pairs:
            track_hits[track_id] = track_hits.get(track_id, 0) + 1
        if any(count > 1 for count in track_hits.values()):
            ambiguous = True

        matched: dict[int, str] = {}
        for _, i, track_id in sorted(pairs):
            if i in matched or track_id in used_tracks:
                continue
            matched[i] = track_id
            used_tracks.add(track_id)

        new_tracks: dict[str, tuple[str, tuple[float, float]]] = {}
        for i, det in enumerate(detections):
            track_id = matched.get(i) or self._new_id(det.color)
            new_tracks[track_id] = (det.color, det.center_px)
            assigned.append((track_id, det))
        self._tracks = new_tracks
        return assigned, ambiguous


def zone_for_x(x: float, width: int) -> str:
    third = width / 3.0
    if x < third:
        return "left"
    if x < 2 * third:
        return "center"
    return "right"
