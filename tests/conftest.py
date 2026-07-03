"""Shared pytest setup — make the src/ modules importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("optimizer", "forecasting", "anomaly", "copilot"):
    sys.path.insert(0, str(ROOT / "src" / sub))
