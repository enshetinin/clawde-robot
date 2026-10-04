"""Diagnóstico del entorno (equivale a `clawde doctor`). No abre cámara ni micrófono."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["doctor", *sys.argv[1:]]))
