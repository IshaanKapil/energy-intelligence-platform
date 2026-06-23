"""
run_all.py — run the full pipeline end to end.

    python run_all.py

Steps: check real data exists -> train load -> train solar -> SHAP -> battery optimizer.
Then start the API with:  uvicorn src.api.main:app --reload

If data/energy_dataset_real.csv is missing, run fetch_real_data.py first:
    python fetch_real_data.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REAL_DATA = ROOT / "data" / "energy_dataset_real.csv"


def run(path, cwd):
    print(f"\n=== running {path} ===")
    subprocess.run([PY, path], cwd=cwd, check=True)


if __name__ == "__main__":
    if not REAL_DATA.exists():
        print("ERROR: Real data not found at data/energy_dataset_real.csv")
        print("Run this first:  python fetch_real_data.py")
        sys.exit(1)

    run("train_load.py",  ROOT / "src" / "forecasting")
    run("train_solar.py", ROOT / "src" / "forecasting")
    run("shap_explain.py", ROOT / "src" / "explain")
    run("battery.py", ROOT / "src" / "optimizer")
    print("\nAll done. Start the API:  uvicorn src.api.main:app --reload")
