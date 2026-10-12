"""Data-driven physical pushing with a deliberately simple planar serial arm.

Three hinge joints plus a tool lift; joint servos move a collision-enabled
pusher. The free object is only initialised once and then moves under physics.
This is a development replay, not a learned policy or held-out evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("trajectory")
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=["demonstration", "straight"], default="demonstration")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
with open(args.trajectory) as f:
    rows = list(csv.DictReader(f))
xy = np.array([[float(r["x_m"]), float(r["y_m"])] for r in rows if r["x_m"] != ""])
# Table coordinates -> robot workspace, with image vertical axis reversed.
xy = np.column_stack((.34 + xy[:, 0], .10 - xy[:, 1]))
path = [xy[0]]
for point in xy[1:]:
    if np.linalg.norm(point-path[-1]) > .012:
        path.append(point)
path.append(xy[-1])
path = np.array(path)
goal = path[-1].copy()
if args.mode == "straight":
    path = np.linspace(path[0], goal, 35)
xml = '''<mujoco model="planar_push"><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/><visual><global offwidth="800" offheight="600"/></visual>
<default><joint damping="3"/><geom friction="0.5 0.005 0.0001"/><position kp="400" kv="35"/></default>
<worldbody><light pos="0.3 0 2"/><geom name="table" type="plane" size="1 1 .1" rgba=".65 .72 .70 1"/>
<geom name="obstacle" type="box" pos=".45 .035 .025" size=".11 .065 .025" rgba=".4 .22 .18 1"/>
<site name="goal" type="cylinder" pos="GOAL .002" size=".025 .001" rgba="0 .8 0 .5"/>
<body name="arm" pos="0 -.28 .12"><joint name="j1" axis="0 0 1" range="-3.14 3.14"/><geom type="capsule" fromto="0 0 0 .27 0 0" size=".014" mass="1" rgba=".2 .4 .8 1"/>
<body pos=".27 0 0"><joint name="j2" axis="0 0 1" range="-3.14 3.14"/><geom type="capsule" fromto="0 0 0 .25 0 0" size=".013" mass=".8" rgba=".2 .5 .9 1"/>
<body pos=".25 0 0"><joint name="j3" axis="0 0 1" range="-3.14 3.14"/><geom type="capsule" fromto="0 0 0 .13 0 0" size=".012" mass=".4" rgba=".3 .6 .9 1"/>
<body pos=".13 0 0"><joint name="lift" type="slide" axis="0 0 1" range="-.09 .08"/><geom name="pusher" type="cylinder" size=".012 .022" mass=".1" rgba=".9 .6 .1 1"/><site name="tip" size=".005"/></body></body></body></body>
<body name="object" pos="START .025"><freejoint/><geom name="object_geom" type="cylinder" size=".023 .025" mass=".08" rgba=".95 .95 .9 1" friction=".3 .005 .0001"/></body>
</worldbody><actuator><position joint="j1"/><position joint="j2"/><position joint="j3"/><position joint="lift" kp="800" kv="45"/></actuator></mujoco>'''
xml = xml.replace("START", f"{path[0,0]} {path[0,1]}").replace("GOAL", f"{goal[0]} {goal[1]}")
# Adjacent simplified capsules overlap at joints. Disable link collisions;
# keep physical pusher/object/obstacle collisions. Record this simplification.
xml = xml.replace('type="capsule"', 'type="capsule" contype="0" conaffinity="0"')
(out / "scene.xml").write_text(xml)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
data.qpos[:4] = [.1, 1.2, -1.3, .04]
mujoco.mj_forward(model, data)
tip = model.site("tip").id
obj = model.body("object").id
object_geom = model.geom("object_geom").id
obstacle = model.geom("obstacle").id
pusher = model.geom("pusher").id
renderer = mujoco.Renderer(model, height=600, width=800) if args.render else None
camera = mujoco.MjvCamera()
camera.lookat[:] = [.3, -.03, .02]
camera.distance = 1.05
camera.azimuth = 90
camera.elevation = -65
writer = cv2.VideoWriter(str(out / "simulation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 600)) if args.render else None
log, obstacle_steps, contact_steps = [], 0, 0
jac = np.zeros((3, model.nv))
ik_data = mujoco.MjData(model)

def advance(target, steps):
    global obstacle_steps, contact_steps
    ik_data.qpos[:] = data.qpos
    for _ in range(20):
        mujoco.mj_forward(model, ik_data)
        mujoco.mj_jacSite(model, ik_data, jac, None, tip)
        j = jac[:, :4]
        error = target-ik_data.site_xpos[tip]
        dq = j.T @ np.linalg.solve(j@j.T + .0001*np.eye(3), error)
        ik_data.qpos[:4] = np.clip(ik_data.qpos[:4] + np.clip(dq, -.1, .1), [-3.13]*3+[-.09], [3.13]*3+[.08])
    desired_q = ik_data.qpos[:4].copy()
    for _ in range(steps):
        data.ctrl[:] += np.clip(desired_q-data.ctrl, -.001, .001)
        mujoco.mj_step(model, data)
        pairs = [{c.geom1, c.geom2} for c in data.contact]
        obstacle_steps += int(any(obstacle in pair and (object_geom in pair or pusher in pair) for pair in pairs))
        contact_steps += int({object_geom, pusher} in pairs)
        if int(round(data.time/model.opt.timestep)) % 50 == 0:
            if renderer:
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            log.append([data.time, *data.xpos[obj], *data.site_xpos[tip]])

# Approach from behind the first direction while the tool is raised.
direction = path[1]-path[0]
direction /= np.linalg.norm(direction)
behind = path[0] - direction*.037
advance(np.r_[behind, .17], 1000)
advance(np.r_[behind, .025], 700)
for waypoint in path[1:]:
    center = data.xpos[obj, :2].copy()
    delta = waypoint-center
    direction = delta/max(np.linalg.norm(delta), 1e-9)
    behind = center-direction*.039
    # Lift before changing the side of contact so turns do not sweep through
    # the object. This robot controller is separate from the human path.
    advance(np.r_[data.site_xpos[tip, :2], .12], 150)
    advance(np.r_[behind, .12], 250)
    advance(np.r_[behind, .025], 200)
    for _ in range(100):
        center = data.xpos[obj, :2].copy()
        delta = waypoint-center
        distance = np.linalg.norm(delta)
        if distance < .008:
            break
        direction = delta/max(distance, 1e-9)
        # Closed-loop pushing; the waypoint sequence is supplied by human data.
        target = center-direction*.034
        advance(np.r_[target, .025], 20)
advance(np.r_[data.site_xpos[tip, :2], .17], 500)
if writer:
    writer.release()
    renderer.close()
with (out / "rollout.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "object_x", "object_y", "object_z", "tool_x", "tool_y", "tool_z"])
    w.writerows(log)
error = float(np.linalg.norm(data.xpos[obj, :2]-goal))
result = {"mode": args.mode, "waypoints": len(path), "sim_seconds": data.time, "endpoint_error_m": error,
          "replay_endpoint_reached": error < .025, "obstacle_contact_steps": obstacle_steps, "pusher_object_contact_steps": contact_steps,
          "start_xy": path[0].tolist(), "demonstration_endpoint_xy": goal.tolist(), "final_object_xy": data.xpos[obj,:2].tolist(),
          "limitations": ["Development replay; endpoint derived from the demonstration, not an independently labelled task goal", "Simplified planar arm and cylindrical proxy object; arm-link collisions disabled, pusher collisions enabled", "No learning or held-out evaluation"]}
(out / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
