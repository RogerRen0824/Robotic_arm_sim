"""Learn macro-action dynamics from simulation driven by applicant demonstrations.

This is a privileged-state surrogate of the robot/controller/physics system,
not a video world model or a claim of learning real-world physics from video.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from push_env import PushEnv,DemoGuide,ROOT,segment_clear

def state(env):
    return np.r_[env.data.qpos,env.data.qvel,env.data.ctrl].astype(np.float32)

def features(states,contexts,actions,guide):
    bases = []
    for s,c in zip(states,contexts):
        center,goal,obstacle = s[4:6],c[:2],c[2:4]
        demo = guide.predict(center,goal,obstacle)
        target = goal-center
        target /= max(np.linalg.norm(target),1e-8)
        weight = .85 if segment_clear(center,goal,obstacle) else 0.
        direction = (1-weight)*demo+weight*target
        bases.append(direction/max(np.linalg.norm(direction),1e-8))
    return np.column_stack((states,contexts,np.array(bases),actions)).astype(np.float32)

def collect(split,episodes):
    env = PushEnv(split=split,randomize=True)
    rng = np.random.default_rng(41 if split=="train" else 71)
    records = []
    start = time.monotonic()
    for episode in range(episodes):
        _,meta = env.reset(seed=(2000 if split=="train" else 4000)+episode,options={"demo_index":episode%len(env.demos)})
        for step in range(80):
            action = np.clip(rng.normal(0,.4,2),-1,1).astype(np.float32)
            before = state(env)
            context = np.r_[env.goal,env.obstacle,env.model.geom_friction[env.og,0]].astype(np.float32)
            _,_,terminated,truncated,info = env.step(action)
            records.append((before,context,action,state(env),episode,meta["demo_id"],terminated or truncated))
            if terminated or truncated:
                break
        print(f"WM {split} episodes {episode+1}/{episodes}, transitions {len(records)}, elapsed {time.monotonic()-start:.0f}s",flush=True)
    env.close()
    return {"states":np.array([r[0] for r in records]),"contexts":np.array([r[1] for r in records]),"actions":np.array([r[2] for r in records]),"next_states":np.array([r[3] for r in records]),"episode":np.array([r[4] for r in records]),"demo_id":np.array([r[5] for r in records]),"done":np.array([r[6] for r in records])}

class Dynamics(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,output_size))
    def forward(self,x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes",type=int,default=24)
    parser.add_argument("--validation-episodes",type=int,default=6)
    parser.add_argument("--epochs",type=int,default=300)
    args = parser.parse_args()
    out = ROOT/"outputs/world_model"
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    datasets = {}
    for split,count in [("train",args.train_episodes),("validation",args.validation_episodes)]:
        datasets[split] = collect(split,count)
        np.savez_compressed(out/(split+".npz"),**datasets[split])
    train,val = datasets["train"],datasets["validation"]
    assert not set(train["demo_id"])&set(val["demo_id"])
    splits = json.loads((ROOT/"data/splits.json").read_text())
    assert not (set(train["demo_id"])|set(val["demo_id"]))&set(splits["test"])
    guide = DemoGuide()
    x = features(train["states"],train["contexts"],train["actions"],guide)
    y = train["next_states"]-train["states"]
    xm,xs = x.mean(0),np.maximum(x.std(0),.01)
    ym,ys = y.mean(0),np.maximum(y.std(0),.001)
    xn,yn = (x-xm)/xs,(y-ym)/ys
    weights = torch.ones(y.shape[1]); weights[4:6] = 4
    trained = {}
    for name,blind in [("action_conditioned",False),("action_blind",True)]:
        torch.manual_seed(19)
        inputs = xn.copy()
        if blind:
            inputs[:,-2:] = 0
        model = Dynamics(x.shape[1],y.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        inputs,targets = torch.tensor(inputs),torch.tensor(yn)
        for epoch in range(args.epochs):
            order = torch.randperm(len(inputs))
            for ids in order.split(128):
                loss = ((model(inputs[ids])-targets[ids])**2*weights).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        trained[name] = (model,blind)
        torch.save({"state_dict":model.state_dict(),"input_size":x.shape[1],"output_size":y.shape[1],"x_mean":torch.tensor(xm),"x_std":torch.tensor(xs),"delta_mean":torch.tensor(ym),"delta_std":torch.tensor(ys),"action_blind":blind},out/(name+".pt"))
    # Ridge baseline tests whether the MLP adds value beyond a linear surrogate.
    design = np.column_stack((xn,np.ones(len(xn))))
    ridge = np.linalg.solve(design.T@design+.1*np.eye(design.shape[1]),design.T@yn)
    np.savez_compressed(out/"normalization_and_ridge.npz",x_mean=xm,x_std=xs,delta_mean=ym,delta_std=ys,ridge=ridge)

    def predict(name,current,context,action):
        if name=="persistence":
            return current.copy()
        inp = (features(current,context,action,guide)-xm)/xs
        if name=="linear_ridge":
            delta = np.column_stack((inp,np.ones(len(inp))))@ridge
        else:
            model,blind = trained[name]
            if blind: inp[:,-2:] = 0
            with torch.no_grad(): delta = model(torch.tensor(inp,dtype=torch.float32)).numpy()
        result = current+delta*ys+ym
        # Project the predicted free-body quaternion onto unit length only.
        result[:,7:11] /= np.maximum(np.linalg.norm(result[:,7:11],axis=1,keepdims=True),1e-6)
        return result

    metrics,examples = {},{}
    for name in ["persistence","linear_ridge","action_blind","action_conditioned"]:
        metrics[name] = {}
        for horizon in [1,3,5]:
            ids = np.array([i for i in range(len(val["states"])-horizon+1) if val["episode"][i]==val["episode"][i+horizon-1] and not val["done"][i:i+horizon-1].any()])
            current = val["states"][ids].copy()
            for offset in range(horizon):
                current = predict(name,current,val["contexts"][ids+offset],val["actions"][ids+offset])
            truth = val["next_states"][ids+horizon-1]
            error = np.linalg.norm(current[:,4:6]-truth[:,4:6],axis=1)
            metrics[name][str(horizon)] = {"windows":len(ids),"position_rmse_mm":float(1000*np.sqrt(np.mean(error**2))),"position_median_mm":float(1000*np.median(error)),"position_p95_mm":float(1000*np.quantile(error,.95))}
            if name=="action_conditioned" and horizon==5:
                examples = {"start_xy":val["states"][ids,4:6].tolist(),"predicted_xy":current[:,4:6].tolist(),"actual_xy":truth[:,4:6].tolist(),"episode":val["episode"][ids].tolist()}
    (out/"rollout_examples.json").write_text(json.dumps(examples))
    report = {"train_transitions":len(train["states"]),"validation_transitions":len(val["states"]),"train_episodes":args.train_episodes,"validation_episodes":args.validation_episodes,"epochs":args.epochs,"metrics":metrics,"source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"limitations":["Learned simulated macro-action state dynamics, not real physics or generative video", "Applicant videos define guide and layout distribution; model targets are simulated", "Validation scene IDs and complete episodes excluded from model fitting", "Multi-step evaluation uses recorded future actions and fixed context, never future dynamic states", "Predicted dynamics are not used to improve the deployed policy; no final test evaluation"]}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    from pathlib import Path
    main()
