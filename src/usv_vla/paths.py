"""Paths derived from the project location; data and caches may be external."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT_ROOT / "assets"
OUTPUTS = Path(os.environ.get("USV_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
