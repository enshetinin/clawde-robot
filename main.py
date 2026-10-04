"""Compatibility wrapper: prefer `clawde` or `python -m clawde`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clawde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
