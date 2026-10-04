"""Prueba MANUAL de cámara y detección de color (equivale a `clawde camera`).

Uso: python scripts/test_camera.py --index 0 [--seconds 10] [--no-preview]
Requiere permiso de Cámara para la terminal. No guarda imágenes salvo --save.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["camera", *sys.argv[1:]]))
