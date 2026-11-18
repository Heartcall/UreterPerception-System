"""Run from a checkout with an existing Python environment."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from ureter_perception.cli import main

    raise SystemExit(main())
