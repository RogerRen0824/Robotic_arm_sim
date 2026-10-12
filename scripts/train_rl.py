"""Small CPU PPO pilot; final test data is never loaded here."""
import argparse
import json
import time
import shutil
import hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from push_env import PushEnv,ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--steps",type=int,default=2048)
parser.add_argument("--seed",type=int,default=7)
parser.add_argument("--scratch",action="store_true")
parser.add_argument("--goal-weight",type=float,default=.85)
parser.add_argument("--residual-weight",type=float,default=.35)
args = parser.parse_args()
torch.set_num_threads(2)
out = ROOT / "outputs/rl" / ("scratch" if args.scratch else "residual") / str(args.seed)
if (out/"policy.zip").exists():
    raise FileExistsError("Checkpoint already exists; choose a new seed/run directory")
out.mkdir(parents=True,exist_ok=True)
source = out / "source"
source.mkdir(exist_ok=True)
fingerprints = {}
for file in (ROOT/"scripts/push_env.py",ROOT/"scripts/train_rl.py",ROOT/"models/planar_push.xml",ROOT/"data/splits.json"):
    shutil.copy2(file,source/file.name)
    fingerprints[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
(out/"config.json").write_text(json.dumps({"steps":args.steps,"seed":args.seed,"scratch":args.scratch,"source_hashes":fingerprints,"push_iterations":50,"goal_weight":args.goal_weight,"residual_weight":args.residual_weight},indent=2))

class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = []
        self.last = time.monotonic()
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episodes.append({"step":self.num_timesteps,**info["episode"],"success":info["is_success"],"collision":info["collision"],"distance_m":info["distance_m"]})
        if time.monotonic()-self.last > 25:
            print(f"Training {self.num_timesteps}/{args.steps}, completed episodes {len(self.episodes)}",flush=True)
            self.last = time.monotonic()
        return True

def evaluate(model=None, residual=True):
    env = PushEnv(split="validation",residual=residual,randomize=False,goal_weight=args.goal_weight,residual_weight=args.residual_weight)
    results = []
    for index in range(len(env.demos)):
        obs,_ = env.reset(seed=100+index,options={"demo_index":index})
        total = 0
        for step in range(100):
            if model is not None:
                action,_ = model.predict(obs,deterministic=True)
            elif residual:
                action = np.zeros(2)
            else:
                delta = env.goal-env.data.xpos[env.obj,:2]
                action = delta/max(np.linalg.norm(delta),1e-8)
            obs,reward,terminated,truncated,info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        results.append({"demo_id":info["demo_id"],"success":info["is_success"],"collision":info["collision"],"distance_m":info["distance_m"],"steps":step+1,"reward":total,"sim_time":info["sim_time"]})
    env.close()
    return results

raw = PushEnv(residual=not args.scratch,goal_weight=args.goal_weight,residual_weight=args.residual_weight)
check_env(raw,warn=True)
env = Monitor(raw)
callback = Progress()
model = PPO("MlpPolicy",env,device="cpu",seed=args.seed,n_steps=256,batch_size=64,n_epochs=5,learning_rate=3e-4,policy_kwargs={"net_arch":[64,64]},verbose=0)
before = {"demo_guide":evaluate(),"straight":evaluate(residual=False)}
print("Validation baselines",json.dumps(before),flush=True)
model.learn(total_timesteps=args.steps,callback=callback)
model.save(out/"policy")
after = evaluate(model,residual=not args.scratch)
report = {"seed":args.seed,"requested_steps":args.steps,"actual_steps":model.num_timesteps,"residual":not args.scratch,"before":before,"after":after,"episodes":callback.episodes,"limitations":["Three validation demonstration-conditioned layouts, not final tests", "Privileged simulator state and fixed low-level controller", "Single-seed short training pilot; no claim of convergence", "Goal for train/validation is prescribed at reset from demonstration endpoint"]}
(out/"report.json").write_text(json.dumps(report,indent=2))
env.close()
print(json.dumps({"after":after,"checkpoint":str(out/"policy.zip")}),flush=True)
