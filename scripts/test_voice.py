"""Prueba MANUAL de micrófono + Whisper local (equivale a `clawde voice-test`).

Uso: python scripts/test_voice.py --seconds 5 [--device N]
Requiere `clawde models prepare --whisper base` y permiso de Micrófono.
No guarda el audio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["voice-test", *sys.argv[1:]]))
