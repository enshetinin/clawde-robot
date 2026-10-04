"""Configuration models and YAML loading.

Relative paths are resolved against the project root (the parent of the config
directory), never against the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from clawde.errors import ConfigError

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- app


class OllamaConfig(StrictModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0, le=1)
    think: bool = False

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
            raise ValueError("solo se admite Ollama local por loopback (http://127.0.0.1:11434)")
        return value

    @field_validator("model")
    @classmethod
    def _local_model_name(cls, value: str) -> str:
        if not value or "cloud" in value.split(":")[-1] or value.endswith("-cloud"):
            raise ValueError("los modelos cloud de Ollama no están permitidos")
        return value


class AppConfig(StrictModel):
    ollama: OllamaConfig = OllamaConfig()
    interpreter: Literal["auto", "rules", "ollama"] = "auto"
    queue_maxsize: int = Field(default=4, ge=1, le=32)
    scene_max_age_s: float = Field(default=3.0, gt=0)
    log_dir: Path = Path("logs")
    captures_dir: Path = Path("captures")
    save_captures: bool = False


# ------------------------------------------------------------------------- voice


class RecorderConfig(StrictModel):
    seconds: float = Field(default=5.0, gt=0, le=30)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    device: int | str | None = None
    min_duration_s: float = Field(default=0.5, ge=0)
    min_rms: float = Field(default=0.004, ge=0)


class WhisperConfig(StrictModel):
    model: str = "base"
    model_dir: Path = Path("models/whisper")
    device: Literal["cpu"] = "cpu"
    compute_type: str = "int8"
    language: str = "es"
    beam_size: int = Field(default=1, ge=1, le=10)
    max_no_speech_prob: float = Field(default=0.6, ge=0, le=1)
    min_avg_logprob: float = Field(default=-1.0, le=0)
    min_chars: int = Field(default=2, ge=1)

    @field_validator("model")
    @classmethod
    def _multilingual(cls, value: str) -> str:
        if value.endswith(".en"):
            raise ValueError("usa un modelo multilingüe (base, small), no '*.en'")
        return value


class TTSConfig(StrictModel):
    enabled: bool = True
    voice: str | None = None
    rate: int = Field(default=185, ge=80, le=400)
    post_speech_pause_s: float = Field(default=0.3, ge=0, le=5)


class VoiceConfig(StrictModel):
    recorder: RecorderConfig = RecorderConfig()
    whisper: WhisperConfig = WhisperConfig()
    tts: TTSConfig = TTSConfig()


# ------------------------------------------------------------------------ vision


class HSVRange(StrictModel):
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    @field_validator("lower", "upper")
    @classmethod
    def _hsv_bounds(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        h, s, v = value
        if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
            raise ValueError("HSV fuera de rango (H 0-179, S/V 0-255)")
        return value


class CameraConfig(StrictModel):
    index: int = Field(default=0, ge=0)
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)


class VisionConfig(StrictModel):
    camera: CameraConfig = CameraConfig()
    colors: dict[str, list[HSVRange]] = Field(default_factory=dict)
    min_area_px: int = Field(default=400, gt=0)
    max_area_px: int = Field(default=200000, gt=0)
    morph_kernel: int = Field(default=5, ge=1, le=31)
    tracking_max_distance_px: float = Field(default=60.0, gt=0)
    calibration_file: Path = Path("config/table_calibration.json")
    max_calibration_rms_mm: float = Field(default=5.0, gt=0)


# ------------------------------------------------------------------------- robot


class ServoConfig(StrictModel):
    name: str
    pin: int = Field(ge=0)
    min_deg: float
    max_deg: float
    home_deg: float
    max_speed_dps: float = Field(gt=0)


class WorkspaceConfig(StrictModel):
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float


class GeometryConfig(StrictModel):
    kind: str
    link_lengths_mm: list[float] = Field(min_length=1)


class DemoObject(StrictModel):
    id: str
    color: str
    zone: Literal["left", "center", "right"]
    x_mm: float
    y_mm: float


class DemoProfile(StrictModel):
    step_delay_s: float = Field(default=0.15, ge=0, le=5)
    objects: list[DemoObject] = Field(default_factory=list)


class RobotConfig(StrictModel):
    driver: Literal["simulator", "serial"] = "simulator"
    hardware_enabled: bool = False
    calibrated: bool = False
    board: str | None = None
    port: str | None = None
    baudrate: int = 115200
    protocol_version: int = 1
    servos: list[ServoConfig] = Field(default_factory=list)
    geometry: GeometryConfig | None = None
    workspace: WorkspaceConfig | None = None
    hold_position_on_stop: bool | None = None
    demo_profile: DemoProfile = DemoProfile()

    def pending_hardware(self) -> list[str]:
        """List what is still missing before real motion could even be considered."""
        missing: list[str] = []
        if self.driver != "serial":
            missing.append("driver distinto de 'serial'")
        if not self.hardware_enabled:
            missing.append("hardware_enabled: false")
        if not self.board:
            missing.append("modelo de placa sin definir")
        if not self.port:
            missing.append("puerto serie sin definir")
        if not self.servos:
            missing.append("servos/pines/límites/home sin definir")
        if self.geometry is None:
            missing.append("geometría del brazo sin definir")
        if self.workspace is None:
            missing.append("espacio de trabajo sin definir")
        if self.hold_position_on_stop is None:
            missing.append("comportamiento de par tras STOP sin definir")
        if not self.calibrated:
            missing.append("calibración no verificada (calibrated: false)")
        return missing


# ---------------------------------------------------------------------- settings


class Settings(StrictModel):
    root: Path
    config_dir: Path
    app: AppConfig = AppConfig()
    voice: VoiceConfig = VoiceConfig()
    vision: VisionConfig = VisionConfig()
    robot: RobotConfig = RobotConfig()

    def path(self, value: Path) -> Path:
        """Resolve a configured path against the project root."""
        return value if value.is_absolute() else (self.root / value).resolve()


def default_config_dir() -> Path:
    env = os.environ.get("CLAWDE_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # src/clawde/config.py -> project root
    return Path(__file__).resolve().parents[2] / "config"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML inválido en {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} debe contener un mapa YAML")
    return data


def load_settings(config_dir: Path | str | None = None) -> Settings:
    cfg_dir = Path(config_dir).expanduser().resolve() if config_dir else default_config_dir()
    if not cfg_dir.is_dir():
        raise ConfigError(f"No existe el directorio de configuración: {cfg_dir}")
    try:
        return Settings(
            root=cfg_dir.parent,
            config_dir=cfg_dir,
            app=AppConfig.model_validate(_read_yaml(cfg_dir / "app.yaml")),
            voice=VoiceConfig.model_validate(_read_yaml(cfg_dir / "voice.yaml")),
            vision=VisionConfig.model_validate(_read_yaml(cfg_dir / "vision.yaml")),
            robot=RobotConfig.model_validate(_read_yaml(cfg_dir / "robot.yaml")),
        )
    except ValidationError as exc:
        raise ConfigError(f"Configuración inválida en {cfg_dir}:\n{exc}") from exc
