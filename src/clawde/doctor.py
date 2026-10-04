"""Environment diagnosis. Never opens the camera or microphone, never moves the arm."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import sys
from typing import Any

from clawde.config import Settings

OK, WARN, FAIL, UNTESTED = (
    "[green]OK[/]",
    "[yellow]AVISO[/]",
    "[red]FALLA[/]",
    "[cyan]NO PROBADO[/]",
)

OPTIONAL = [
    ("pydantic", "pydantic", "base"),
    ("yaml", "PyYAML", "base"),
    ("rich", "rich", "base"),
    ("serial", "pyserial", "base"),
    ("ollama", "ollama", "base"),
    ("numpy", "numpy", "vision/voice"),
    ("cv2", "opencv-contrib-python", "vision"),
    ("sounddevice", "sounddevice", "voice"),
    ("faster_whisper", "faster-whisper", "voice"),
    ("pytest", "pytest", "dev"),
]


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def run_doctor(settings: Settings, console: Any) -> int:
    from rich.table import Table

    table = Table(title="CLAWDE doctor", show_lines=False)
    table.add_column("Comprobación")
    table.add_column("Estado")
    table.add_column("Detalle", overflow="fold")
    failures = 0

    def row(name: str, status: str, detail: str = "") -> None:
        table.add_row(name, status, detail)

    in_venv = sys.prefix != sys.base_prefix
    row(
        "Python",
        OK if sys.version_info >= (3, 12) else FAIL,
        f"{platform.python_version()} {platform.machine()} {sys.executable}"
        + ("" if in_venv else " (fuera de un venv)"),
    )
    row("Configuración", OK, str(settings.config_dir))

    for module, dist, extra in OPTIONAL:
        found = importlib.util.find_spec(module) is not None
        if not found and extra == "base":
            failures += 1
        row(
            f"dep {dist}",
            OK if found else (FAIL if extra == "base" else WARN),
            _version(dist) if found else f"falta (extra [{extra}])",
        )
    ruff = shutil.which("ruff") or (shutil.which("ruff", path=str(sys.prefix) + "/bin"))
    row("dep ruff", OK if ruff else WARN, ruff or "falta (extra [dev])")

    from clawde.brain.ollama_client import OllamaBrain

    cfg = settings.app.ollama
    probe = OllamaBrain(cfg).probe()
    if not probe.reachable:
        row(
            "Ollama",
            WARN,
            f"no responde en {cfg.host}: abre Ollama o ejecuta `ollama serve`. "
            "Se usará el intérprete de reglas.",
        )
    else:
        row("Ollama", OK, f"{cfg.host}; modelos locales: {', '.join(probe.models) or 'ninguno'}")
        row(
            f"Modelo {cfg.model}",
            OK if probe.has_model(cfg.model) else WARN,
            "instalado (soporte de visión no asumido)"
            if probe.has_model(cfg.model)
            else f"no instalado: `ollama pull {cfg.model}`",
        )

    from clawde.voice.transcriber import local_weights_path

    wcfg = settings.voice.whisper
    weights = local_weights_path(wcfg, settings.path(wcfg.model_dir))
    row(
        f"Whisper {wcfg.model}",
        OK if weights else WARN,
        str(weights)
        if weights
        else f"sin pesos locales: `clawde models prepare --whisper {wcfg.model}`",
    )

    if importlib.util.find_spec("sounddevice"):
        try:
            from clawde.voice.recorder import list_input_devices

            devices = list_input_devices()
            names = ", ".join(
                f"{d['index']}:{d['name']}" + ("*" if d["default"] else "") for d in devices
            )
            row(
                "Micrófonos",
                OK if devices else WARN,
                (names or "ninguno") + " — captura " + "no probada",
            )
        except Exception as exc:
            row("Micrófonos", WARN, f"no se pudieron listar: {exc}")
    row("Captura de micrófono", UNTESTED, "usa `clawde voice-test --seconds 5`")
    row("Cámara", UNTESTED, "usa `clawde camera --index 0` (requiere permiso de Cámara)")

    try:
        from clawde.robot.serial_transport import list_serial_ports

        ports = list_serial_ports()
        row("Puertos serie", OK, ", ".join(p for p, _ in ports) or "ninguno (normal sin placa)")
    except Exception as exc:
        row("Puertos serie", WARN, str(exc))

    from clawde.voice.speaker import choose_voice, list_voices, say_available

    if say_available():
        voice, note = choose_voice(list_voices(), settings.voice.tts.voice)
        row(
            "Voz (say)",
            OK if voice else WARN,
            (voice.name if voice else "") + (f" {note}" if note else ""),
        )
    else:
        row("Voz (say)", WARN, "solo disponible en macOS")

    pio = shutil.which("pio") or shutil.which("platformio")
    row("PlatformIO", OK if pio else WARN, pio or "no encontrado (solo necesario para firmware)")

    robot = settings.robot
    pending = robot.pending_hardware()
    row(
        "Brazo",
        OK if robot.driver == "simulator" else WARN,
        f"driver={robot.driver}, hardware_enabled={robot.hardware_enabled}, "
        f"calibrated={robot.calibrated}",
    )
    row(
        "Hardware real",
        WARN if pending else OK,
        ("pendiente: " + "; ".join(pending) + ". En simulación no es un error.")
        if pending
        else "completo",
    )

    console.print(table)
    return 1 if failures else 0
