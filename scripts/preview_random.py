"""Render one saved validation scene; macro snapshots, not real-time speed."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from stable_baselines3 import PPO
from random_push_env import RandomPushEnv

parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--index',type=int,default=7)
parser.add_argument('--assistance',choices=['none','planner','demo'],default='demo')
parser.add_argument('--checkpoint')
args = parser.parse_args()
root = Path(args.run)
scene = json.loads((root/'config.json').read_text())['validation_scenes'][args.index]
out = root/'previews'/args.assistance
out.mkdir(parents=True,exist_ok=True)
model = PPO.load(args.checkpoint,device='cpu') if args.checkpoint else None
env = RandomPushEnv(assistance=args.assistance)
obs,_ = env.reset(options={'scene':scene})
writer = cv2.VideoWriter(str(out/'validation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),3,(640,480))
writer.write(cv2.cvtColor(env.render(),cv2.COLOR_RGB2BGR))
for step in range(100):
    if model:
        action,_ = model.predict(obs,deterministic=True)
    elif args.assistance=='none':
        delta = env.goal-env.data.xpos[env.obj,:2]
        action = delta/max(np.linalg.norm(delta),1e-8)
    else:
        action = np.zeros(2)
    obs,_,terminated,truncated,info = env.step(action)
    frame = cv2.cvtColor(env.render(),cv2.COLOR_RGB2BGR)
    label = f"{args.assistance} {'PPO' if model else 'baseline'}: {len(scene['obstacles'])} obstacles"
    cv2.putText(frame,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,220),2)
    cv2.putText(frame,f"step {step+1}, goal error {info['distance_m']*1000:.0f} mm",(10,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,220),2)
    writer.write(frame)
    if terminated or truncated:
        break
writer.release()
cv2.imwrite(str(out/'endpoint.jpg'),frame)
(out/'result.json').write_text(json.dumps({'scene':scene,'index':args.index,'method':args.assistance,
    'checkpoint':args.checkpoint,'success':info['is_success'],'collision':info['collision'],
    'distance_mm':info['distance_m']*1000,'steps':step+1,
    'note':'Validation only; macro-action snapshots, not continuous motion'},indent=2))
env.close()
print('Preview complete',info['is_success'],info['distance_m'])
