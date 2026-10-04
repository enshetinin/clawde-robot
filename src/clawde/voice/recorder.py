"""Bounded microphone capture with sounddevice (imported lazily)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from clawde.config import RecorderConfig
from clawde.errors import DeviceError, require


@dataclass
class Recording:
    audio: Any  # numpy float32 mono array
    sample_rate: int
    overflowed: bool = False
    cancelled: bool = False

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(self.sample_rate) if self.sample_rate else 0.0


class AudioBackend(Protocol):
    def record(
        self, seconds: float, sample_rate: int, device: int | str | None, cancel: threading.Event
    ) -> tuple[Any, bool]: ...


class SoundDeviceBackend:
    """Records into memory only; nothing is written to disk."""

    def record(
        self, seconds: float, sample_rate: int, device: int | str | None, cancel: threading.Event
    ) -> tuple[Any, bool]:
        sd: Any = require("sounddevice", "voice")
        np: Any = require("numpy", "voice")
        blocks: list[Any] = []
        overflow = threading.Event()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status and status.input_overflow:
                overflow.set()
            blocks.append(indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
                callback=callback,
            ):
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline and not cancel.wait(0.05):
                    pass
        except sd.PortAudioError as exc:
            raise DeviceError(
                f"Error de micrófono: {exc}. Revisa el permiso de Micrófono para Terminal/Zed."
            ) from exc
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        return audio, overflow.is_set()


class Recorder:
    def __init__(self, config: RecorderConfig, backend: AudioBackend | None = None) -> None:
        self.config = config
        self.backend = backend or SoundDeviceBackend()

    def record(
        self, seconds: float | None = None, cancel: threading.Event | None = None
    ) -> Recording:
        cancel = cancel or threading.Event()
        duration = seconds if seconds is not None else self.config.seconds
        audio, overflowed = self.backend.record(
            duration, self.config.sample_rate, self.config.device, cancel
        )
        return Recording(audio, self.config.sample_rate, overflowed, cancel.is_set())


def check_audio(recording: Recording, config: RecorderConfig) -> str | None:
    """Return a rejection reason, or None if the audio looks usable."""
    np: Any = require("numpy", "voice")
    if recording.cancelled:
        return "grabación cancelada"
    if recording.overflowed:
        return "desbordamiento del búfer de audio; repite la orden"
    if recording.audio is None or len(recording.audio) == 0:
        return "no se capturó audio"
    if recording.duration_s < config.min_duration_s:
        return "audio demasiado corto"
    audio = np.asarray(recording.audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        return "audio corrupto"
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < config.min_rms:
        return "solo se ha detectado silencio"
    return None


def list_input_devices() -> list[dict[str, Any]]:
    sd: Any = require("sounddevice", "voice")
    default_in = sd.default.device[0] if sd.default.device else None
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": index,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev.get("default_samplerate"),
                    "default": index == default_in,
                }
            )
    return devices
