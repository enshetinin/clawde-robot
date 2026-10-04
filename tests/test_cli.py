"""CLI behaviour, installation and side-effect-free imports."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest

import clawde
from clawde.cli import main
from clawde.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, timeout=60, cwd=cwd, check=False
    )


def test_help_works() -> None:
    result = run("-m", "clawde", "--help")
    assert result.returncode == 0
    for cmd in ["doctor", "models", "devices", "camera", "voice-test", "run", "command"]:
        assert cmd in result.stdout


def test_main_py_wrapper(tmp_path) -> None:
    result = run(str(ROOT / "main.py"), "--version", cwd=tmp_path)
    assert result.returncode == 0 and clawde.__version__ in result.stdout


def test_editable_install_metadata() -> None:
    assert importlib.metadata.version("clawde") == clawde.__version__
    scripts = importlib.metadata.entry_points(group="console_scripts")
    assert any(ep.name == "clawde" and ep.value == "clawde.cli:main" for ep in scripts)


def test_imports_have_no_heavy_side_effects() -> None:
    code = (
        "import sys\n"
        "import clawde.cli, clawde.app, clawde.brain.interpreter\n"
        "import clawde.robot.serial_transport, clawde.vision.scene, clawde.vision.camera\n"
        "import clawde.voice.interaction, clawde.doctor\n"
        "heavy = ['cv2', 'sounddevice', 'faster_whisper', 'ctranslate2', 'serial',\n"
        "         'ollama', 'numpy']\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    result = run("-c", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_config_is_independent_of_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.root == ROOT
    assert settings.path(settings.app.log_dir) == ROOT / "logs"
    assert settings.robot.driver == "simulator" and not settings.robot.hardware_enabled
    assert not settings.robot.calibrated and settings.robot.board is None


def test_command_sim_demo_never_opens_serial(monkeypatch, capsys) -> None:
    import serial

    def forbidden(*a, **k):
        raise AssertionError("no debe abrir el puerto serie")

    monkeypatch.setattr(serial, "Serial", forbidden)
    monkeypatch.setattr("clawde.robot.serial_transport.open_serial_port", forbidden)
    code = main(
        [
            "command",
            "--sim",
            "--demo",
            "--interpreter",
            "rules",
            "coge el objeto rojo y ponlo a la izquierda",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "FICTICIA" in out and "red_1" in out and "reglas" in out


def test_command_real_fails_before_motion(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "clawde.robot.serial_transport.open_serial_port",
        lambda *a, **k: pytest.fail("no debe abrir el puerto"),
    )
    code = main(["command", "--real", "--interpreter", "rules", "vuelve a casa"])
    assert code == 2
    assert "bloqueado" in capsys.readouterr().out


def test_real_and_demo_are_incompatible(capsys) -> None:
    assert main(["command", "--real", "--demo", "--interpreter", "rules", "estado"]) == 2


def test_cli_overrides_yaml(monkeypatch, capsys) -> None:
    from clawde.brain.ollama_client import OllamaBrain, OllamaProbe

    seen = {}

    def probe(self, timeout_s=2.0):
        seen["model"] = self.config.model
        return OllamaProbe(reachable=False, models=())

    monkeypatch.setattr(OllamaBrain, "probe", probe)
    assert (
        main(["command", "--demo", "--interpreter", "auto", "--model", "llama3.2:3b", "estado"])
        == 0
    )
    assert seen["model"] == "llama3.2:3b"
    out = capsys.readouterr().out
    assert "reglas deterministas" in out  # announced fallback for auto mode
    # explicit ollama mode refuses to silently fall back
    assert main(["command", "--demo", "--interpreter", "ollama", "estado"]) == 2


def test_cloud_model_override_rejected(capsys) -> None:
    assert main(["command", "--demo", "--interpreter", "auto", "--model", "x:cloud", "estado"]) == 2


def test_run_text_session_via_stdin(tmp_path) -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clawde",
            "run",
            "--sim",
            "--demo",
            "--text",
            "--interpreter",
            "rules",
        ],
        input="¿qué ves?\ncierra la pinza\nestado\n",
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = " ".join(result.stdout.split())  # undo Rich wrapping
    assert "Veo 3 objeto(s)" in out and "pinza cerrada" in out
    assert "pinza close" in out and "CLAWDE cerrado" in out


def test_run_exits_on_salir(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "clawde", "run", "--demo", "--interpreter", "rules"],
        input="salir\n",
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0 and "Hasta luego" in result.stdout
