"""Extract local demonstrations, audit missing runs, mark short-gap repairs."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2

root = Path(__file__).resolve().parents[1]
reports = []
split_path = root / "data/splits.json"
splits = json.loads(split_path.read_text()) if split_path.exists() else {}
goal_path = root / "data/goal_annotations.json"
annotations = json.loads(goal_path.read_text()) if goal_path.exists() else {}
for video in sorted((root / "data").glob("*.mp4")):
    out = root / "outputs" / "dataset" / video.stem
    process = subprocess.run([sys.executable, str(root / "scripts/extract_pilot.py"), str(video), "--output", str(out)], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Extraction failed for {video.name}:\n{process.stderr}")
    with (out / "trajectory.csv").open() as f:
        rows = list(csv.DictReader(f))
    valid = np.array([int(r["valid"]) for r in rows], bool)
    gaps, i = [], 0
    while i < len(rows):
        if valid[i]:
            i += 1
            continue
        j = i
        while j < len(rows) and not valid[j]:
            j += 1
        gaps.append({"start_frame": i, "end_frame": j - 1, "length": j - i, "start_s": float(rows[i]["time_s"]), "end_s": float(rows[j-1]["time_s"])})
        i = j
    for r in rows:
        r["interpolated"] = 0
    for gap in gaps:
        a, b = gap["start_frame"], gap["end_frame"] + 1
        if a > 0 and b < len(rows) and b-a <= 10:
            for k in range(a, b):
                alpha = (float(rows[k]["time_s"]) - float(rows[a-1]["time_s"])) / (float(rows[b]["time_s"]) - float(rows[a-1]["time_s"]))
                for field in ("u_px", "v_px", "x_m", "y_m"):
                    rows[k][field] = (1-alpha)*float(rows[a-1][field]) + alpha*float(rows[b][field])
                rows[k]["interpolated"] = 1
    with (out / "trajectory_prepared.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads((out / "summary.json").read_text())
    split = next((name for name in ("train", "validation", "test") if video.stem in splits.get(name, [])), "unassigned")
    print(f"{video.stem}: {valid.sum()}/{len(rows)} detected, {split}", flush=True)
    goal = "unannotated"
    if annotations.get("status") == "confirmed_by_user" and video.stem in annotations.get("goals", {}):
        uv = annotations["goals"][video.stem]
        transform = cv2.getPerspectiveTransform(np.float32(summary["corners_first_frame_px"]), np.float32([[0,0],[.22,0],[.22,.13],[0,.13]]))
        metric = cv2.perspectiveTransform(np.float32([[uv]]), transform)[0,0].tolist()
        goal = {"source": "first_frame_black_marker", "pixel_xy": uv, "notebook_xy_m": metric, "approximate": True}
    reports.append({"id": video.stem, "sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "frames": len(rows), "detected": int(valid.sum()), "gaps": gaps, "interpolated": sum(r["interpolated"] for r in rows), "split": split, "goal": goal, "calibration_corners_px": summary["corners_first_frame_px"]})
(root / "outputs/dataset/manifest.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
print(json.dumps({"videos": len(reports), "frames": sum(r["frames"] for r in reports), "detected": sum(r["detected"] for r in reports), "interpolated": sum(r["interpolated"] for r in reports)}, indent=2))
