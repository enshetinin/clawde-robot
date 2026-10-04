"""Local speech-to-text with faster-whisper (CPU, int8).

Normal use loads weights with local_files_only=True. Only `clawde models
prepare` downloads them.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from clawde.config import WhisperConfig
from clawde.errors import ModelNotPrepared, require

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
KNOWN_HALLUCINATIONS = (
    "subtitulos realizados por la comunidad de amara org",
    "gracias por ver el video",
    "suscribete",
    "amara org",
)


@dataclass(frozen=True)
class Transcription:
    text: str
    avg_logprob: float
    no_speech_prob: float
    language: str | None = None
    language_probability: float | None = None


class Transcriber(Protocol):
    def transcribe(self, audio: Any) -> Transcription: ...


def _cache_dir_name(model: str) -> str:
    return f"models--Systran--faster-whisper-{model}"


def local_weights_path(config: WhisperConfig, model_dir: Path) -> Path | None:
    """Return the local snapshot dir if the weights are present, without loading them."""
    candidate = Path(config.model)
    if candidate.is_absolute() and (candidate / "model.bin").exists():
        return candidate
    snapshots = model_dir / _cache_dir_name(config.model) / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir()):
            if (snap / "model.bin").exists():
                return snap
    return None


def prepare_model(model: str, model_dir: Path) -> Path:
    """Explicit download (needs Internet once). Records where the weights landed."""
    faster_whisper: Any = require("faster_whisper", "voice")
    if model.endswith(".en"):
        raise ValueError("usa un modelo multilingüe")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = Path(faster_whisper.download_model(model, cache_dir=str(model_dir)))
    manifest = model_dir / MANIFEST
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data[model] = {
        "path": str(path),
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class WhisperTranscriber:
    """Keeps the model loaded between orders."""

    def __init__(self, config: WhisperConfig, model_dir: Path) -> None:
        self.config = config
        self.model_dir = model_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        faster_whisper: Any = require("faster_whisper", "voice")
        if local_weights_path(self.config, self.model_dir) is None:
            raise ModelNotPrepared(
                f"No hay pesos locales de Whisper '{self.config.model}' en {self.model_dir}.\n"
                "Prepáralos (descarga única) con: "
                f"clawde models prepare --whisper {self.config.model}"
            )
        try:
            self._model = faster_whisper.WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=str(self.model_dir),
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelNotPrepared(
                f"No se pudo cargar Whisper desde {self.model_dir}: {exc}\n"
                f"Ejecuta: clawde models prepare --whisper {self.config.model}"
            ) from exc

    def transcribe(self, audio: Any) -> Transcription:
        self.load()
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            segments = list(segments)
        if not segments:
            return Transcription("", -10.0, 1.0, info.language, info.language_probability)
        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        no_speech = max(s.no_speech_prob for s in segments)
        return Transcription(text, avg_logprob, no_speech, info.language, info.language_probability)


def assess_transcription(t: Transcription, config: WhisperConfig) -> str | None:
    """Heuristic rejection of empty or dubious transcriptions. Not a guarantee."""
    from clawde.brain.interpreter import normalize

    normalized = normalize(t.text)
    if len(re.sub(r"\s", "", normalized)) < config.min_chars:
        return "no se ha entendido ninguna palabra"
    if t.no_speech_prob > config.max_no_speech_prob:
        return "parece que no había voz"
    if t.avg_logprob < config.min_avg_logprob:
        return "transcripción poco fiable; repite la orden más despacio"
    if any(h in normalized for h in KNOWN_HALLUCINATIONS):
        return "transcripción dudosa (posible alucinación del modelo)"
    return None
