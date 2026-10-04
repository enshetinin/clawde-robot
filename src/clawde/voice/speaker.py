"""Speech output with macOS /usr/bin/say. Argument lists only, never a shell.

The text goes through stdin, so it can never be parsed as a command-line option.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clawde.config import TTSConfig

log = logging.getLogger(__name__)

SAY = "/usr/bin/say"
PREFERRED_SPANISH = ("Mónica", "Paulina", "Jorge", "Juan", "Marisol")
_VOICE_LINE = re.compile(
    r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2,3}_[A-Za-z0-9]{2,4})\s+#\s?(?P<sample>.*)$"
)


@dataclass(frozen=True)
class Voice:
    name: str
    locale: str

    @property
    def spanish(self) -> bool:
        return self.locale.startswith("es_")


def say_available() -> bool:
    return sys.platform == "darwin" and Path(SAY).exists()


def parse_voices(output: str) -> list[Voice]:
    voices: list[Voice] = []
    seen: set[str] = set()
    for line in output.splitlines():
        match = _VOICE_LINE.match(line.strip())
        if match and match.group("name") not in seen:
            seen.add(match.group("name"))
            voices.append(Voice(match.group("name").strip(), match.group("locale")))
    return voices


def list_voices(run: Callable[..., Any] = subprocess.run) -> list[Voice]:
    if not say_available():
        return []
    result = run([SAY, "-v", "?"], capture_output=True, text=True, timeout=10, check=False)
    return parse_voices(result.stdout or "")


def choose_voice(voices: list[Voice], preferred: str | None) -> tuple[Voice | None, str | None]:
    """Pick the configured voice if installed, otherwise an installed Spanish voice."""
    if preferred:
        match = next((v for v in voices if v.name == preferred), None)
        if match:
            return match, None
        note = f"La voz '{preferred}' no está instalada."
    else:
        note = None
    spanish = [v for v in voices if v.spanish]
    for name in PREFERRED_SPANISH:
        for voice in spanish:
            if voice.name.split(" (")[0] == name:
                return voice, note
    if spanish:
        return spanish[0], note
    return None, (
        (note + " " if note else "")
        + "No hay voces en español instaladas: CLAWDE responderá solo con texto. Descarga una en "
        "Ajustes del Sistema > Accesibilidad > Contenido leído > Voz del sistema > Gestionar voces."
    )


class Speaker:
    """Non-blocking TTS; `stop()` interrupts it immediately."""

    def __init__(
        self,
        config: TTSConfig,
        voice: Voice | None,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.config = config
        self.voice = voice
        self._popen = popen
        self._proc: Any = None
        self._lock = threading.Lock()
        self._last_end = 0.0

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.voice is not None

    def command(self) -> list[str]:
        assert self.voice is not None
        return [SAY, "-v", self.voice.name, "-r", str(self.config.rate), "-f", "-"]

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        self.stop()
        with self._lock:
            try:
                proc = self._popen(
                    self.command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                proc.stdin.write(text)
                proc.stdin.close()
                self._proc = proc
            except OSError as exc:
                log.warning("say failed: %s", exc)

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def wait_idle(self, cancel: threading.Event | None = None, timeout_s: float = 30.0) -> None:
        """Block until speech ended plus the configured pause (never record our own voice)."""
        deadline = time.monotonic() + timeout_s
        while self.is_speaking and time.monotonic() < deadline:
            if cancel is not None and cancel.wait(0.05):
                return
            if cancel is None:
                time.sleep(0.05)
        with self._lock:
            if self._proc is not None:
                self._last_end = time.monotonic()
                self._proc = None
        remaining = self._last_end + self.config.post_speech_pause_s - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc is not None:
            self._last_end = time.monotonic()
