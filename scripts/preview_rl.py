"""Preview a checkpoint on validation only; no final test loading."""
import argparse
import json
from pathlib import Path
import cv2
from stable_baselines3 import PPO
from push_env import PushEnv,ROOT

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--demo-index",type=int,default=2)
parser.add_argument("--output",default=str(ROOT/"outputs/rl_preview"))
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True,exist_ok=True)
model = PPO.load(args.checkpoint,device="cpu")
env = PushEnv(split="validation",randomize=False)
obs,_ = env.reset(seed=100+args.demo_index,options={"demo_index":args.demo_index})
writer = cv2.VideoWriter(str(out/"validation_policy.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),3,(640,480))
writer.write(cv2.cvtColor(env.render(),cv2.COLOR_RGB2BGR))
for step in range(100):
    action,_ = model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info = env.step(action)
    rgb = env.render()
    frame = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.putText(frame,f'step {step+1} error {info["distance_m"]*1000:.0f} mm',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    writer.write(frame)
    if terminated or truncated:
        break
writer.release()
cv2.imwrite(str(out/"final_frame.jpg"),frame)
env.close()
(out/"result.json").write_text(json.dumps({"checkpoint":args.checkpoint,"split":"validation","step":step+1,"info":info,"note":"Macro-action snapshots, not continuous video or real-time speed"},indent=2))
print(json.dumps(info))
