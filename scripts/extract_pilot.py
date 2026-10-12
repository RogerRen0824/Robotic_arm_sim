"""Pilot extraction for the supplied 720x1280 scene; review overlays before use.

Colour-detected first-frame notebook corners define an approximate plane in metres.
Notebook template matching estimates apparent translation, not its cause.
No interpolation of missing detections and no independent goal annotation.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(args.video)
ok, first = cap.read()
if not ok:
    raise RuntimeError("Cannot read video")
if first.shape[:2] != (1280, 720):
    raise ValueError("Pilot coordinates only support 720x1280 frames")
fps = cap.get(cv2.CAP_PROP_FPS)
# Detect the brown notebook in the first frame; reject ambiguous geometry.
b, g, r = [channel.astype(np.int16) for channel in cv2.split(first)]
brown = np.uint8((r-g > 8) & (r-b > 8) & (r < 170) & (r > 35))*255
brown[:100] = 0
brown[500:] = 0
brown[:, :150] = 0
brown[:, 580:] = 0
brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
regions, _ = cv2.findContours(brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not regions:
    raise ValueError("Notebook not found; manual calibration is required")
region = max(regions, key=cv2.contourArea)
polygon = cv2.approxPolyDP(region, .04*cv2.arcLength(region, True), True).reshape(-1, 2)
if cv2.contourArea(region) < 20000 or len(polygon) != 4:
    raise ValueError("Notebook is not an unambiguous quadrilateral; review calibration")
points = polygon.astype(np.float32)
sums = points.sum(axis=1)
differences = points[:, 1]-points[:, 0]
corners = np.float32([points[np.argmin(sums)], points[np.argmin(differences)], points[np.argmax(sums)], points[np.argmax(differences)]])
calibration_method = "colour_quadrilateral"
# Prefer alignment to the visually reviewed reference for matching notebook
# orientation/scale. Shadow boundaries can cut into the colour silhouette.
reference_path = Path(__file__).resolve().parents[1] / "data/4adc8157380f6541c351f3929c693d36.mp4"
reference_cap = cv2.VideoCapture(str(reference_path))
reference_ok, reference_frame = reference_cap.read()
reference_cap.release()
if reference_ok:
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    reference_template = reference_gray[200:390, 210:500]
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    alignment = cv2.matchTemplate(first_gray[100:550, 140:570], reference_template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, location = cv2.minMaxLoc(alignment)
    if confidence > .80:
        translation = np.float32([location[0]+140-210, location[1]+100-200])
        corners = np.float32([[221, 214], [486, 206], [490, 377], [226, 384]]) + translation
        calibration_method = "reviewed_reference_translation"
world = np.float32([[0, 0], [.22, 0], [.22, .13], [0, .13]])
gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
x0, y0 = np.floor(corners.min(axis=0)-5).astype(int)
x1, y1 = np.ceil(corners.max(axis=0)+5).astype(int)
template = gray[y0:y1, x0:x1]
writer = cv2.VideoWriter(str(out / "tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (720, 640))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
rows, trail = [], []
previous = None
frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = cv2.matchTemplate(gray[100:550, 140:570], template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(scores)
    shift = np.array([loc[0] + 140 - x0, loc[1] + 100 - y0], dtype=np.float32)
    moved = corners + shift
    homography = cv2.getPerspectiveTransform(moved, world)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([179, 85, 255]))
    mask[:20] = 0
    mask[610:] = 0
    mask[:, :125] = 0
    mask[:, 560:] = 0
    # Silver table fittings lie near the lower corners.
    cv2.circle(mask, (148, 548), 29, 0, -1)
    cv2.circle(mask, (541, 540), 29, 0, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if 1000 < area < 5000 and 30 < w < 90 and 30 < h < 95 and .55 < w / h < 1.6:
            m = cv2.moments(contour)
            center = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            candidates.append((np.linalg.norm(center - previous) if previous is not None else -area, center))
    row = {"frame": frame_id, "time_s": frame_id / fps, "template_score": score,
           "reference_dx_px": float(shift[0]), "reference_dy_px": float(shift[1]),
           "u_px": "", "v_px": "", "x_m": "", "y_m": "", "valid": 0}
    if candidates:
        distance, center = min(candidates, key=lambda item: item[0])
        if distance < 70 and score > .65:
            previous = center
            xy = cv2.perspectiveTransform(np.float32([[center]]), homography)[0, 0]
            row.update(u_px=float(center[0]), v_px=float(center[1]), x_m=float(xy[0]), y_m=float(xy[1]), valid=1)
            trail.append(tuple(center.astype(int)))
            cv2.circle(frame, trail[-1], 8, (0, 255, 0), 2)
    cv2.polylines(frame, [moved.astype(int)], True, (255, 0, 0), 2)
    if len(trail) > 1:
        cv2.polylines(frame, [np.array(trail)], False, (0, 0, 255), 2)
    cv2.putText(frame, f'{frame_id/fps:.1f}s valid={row["valid"]} ref={score:.2f}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    writer.write(frame[:640])
    if frame_id % 30 == 0:
        cv2.imwrite(str(out / "latest_overlay.jpg"), frame[:640])
    rows.append(row)
    frame_id += 1
cap.release()
writer.release()
with (out / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
    csv_writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    csv_writer.writeheader()
    csv_writer.writerows(rows)
summary = {"source": args.video, "frames": len(rows), "valid_frames": sum(r["valid"] for r in rows),
           "calibration_method": calibration_method,
           "reference_shift_range_px": np.ptp([[r["reference_dx_px"], r["reference_dy_px"]] for r in rows], axis=0).tolist(),
           "min_template_score": min(r["template_score"] for r in rows),
           "notebook_dimensions_m": [.22, .13], "corners_first_frame_px": corners.tolist(),
           "limitations": ["Colour-based approximate calibration; notebook surface height is unmeasured", "Object top centre is a proxy for tabletop contact position", "Template translation is only an apparent image displacement", "No independent goal label; no robot simulation results"]}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
