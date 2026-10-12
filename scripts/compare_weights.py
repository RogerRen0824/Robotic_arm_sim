"""Direction-weight ablation on validation only, with identical scene seeds."""
import json
import numpy as np
from push_env import PushEnv,ROOT

results = []
for weight in [0.,.5,.85,1.]:
    env = PushEnv(split="validation",randomize=False,goal_weight=weight)
    for index in range(len(env.demos)):
        obs,_ = env.reset(seed=100+index,options={"demo_index":index})
        for step in range(100):
            obs,reward,terminated,truncated,info = env.step(np.zeros(2))
            if terminated or truncated:
                break
        row = {"goal_weight":weight,"demo_id":info["demo_id"],"steps":step+1,"success":info["is_success"],"collision":info["collision"],"distance_m":info["distance_m"],"sim_time":info["sim_time"]}
        results.append(row)
        print(json.dumps(row),flush=True)
    env.close()
out = ROOT/"outputs/weight_ablation"
out.mkdir(parents=True,exist_ok=True)
(out/"results.json").write_text(json.dumps(results,indent=2))
