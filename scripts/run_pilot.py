"""Run development demonstrations and matched straight-line baselines."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = []
manifest = json.loads((root / "outputs/dataset/manifest.json").read_text())
allowed = {r["id"] for r in manifest if r["split"] in ("train", "validation", "development")}
for trajectory in sorted((root / "outputs/dataset").glob("*/trajectory_prepared.csv")):
    if trajectory.parent.name not in allowed:
        continue
    for mode in ("demonstration", "straight"):
        out = root / "outputs/evaluation" / trajectory.parent.name / mode
        subprocess.run([sys.executable, str(root / "scripts/simulate_push.py"), str(trajectory), "--output", str(out), "--mode", mode], capture_output=True, check=True)
        result = json.loads((out / "result.json").read_text())
        result["demo_id"] = trajectory.parent.name
        results.append(result)
        print(trajectory.parent.name, mode, round(result["endpoint_error_m"], 4), result["obstacle_contact_steps"], flush=True)
(root / "outputs/evaluation/results.json").write_text(json.dumps(results, indent=2))
