"""Curriculum pilot and matching baselines on new random validation scenes."""
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from random_push_env import RandomPushEnv,ROOT


def evaluate(model,assistance,scenes):
    env = RandomPushEnv(assistance=assistance)
    records = []
    for scene in scenes:
        obs,_ = env.reset(options={'scene':scene})
        for step in range(100):
            if model is not None:
                action,_ = model.predict(obs,deterministic=True)
            elif assistance=='none':
                delta = env.goal-env.data.xpos[env.obj,:2]
                action = delta/max(np.linalg.norm(delta),1e-8)
            else:
                action = np.zeros(2)
            obs,reward,terminated,truncated,info = env.step(action)
            if terminated or truncated:
                break
        records.append({'scene':scene,'count':len(scene['obstacles']),'success':info['is_success'],
                        'collision':info['collision'],'distance_mm':info['distance_m']*1000,
                        'steps':step+1,'sim_time':info['sim_time']})
        print('Validation',assistance,'PPO' if model else 'baseline',len(records),
              records[-1]['count'],records[-1]['success'],round(records[-1]['distance_mm']),flush=True)
    env.close()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=21)
    parser.add_argument('--assistance',choices=['none','planner','demo'],default='none')
    parser.add_argument('--stage-steps',type=int,default=1024)
    args = parser.parse_args()
    out = ROOT/'outputs/random_scene'/args.assistance/str(args.seed)
    if out.exists():
        raise FileExistsError('Choose a new seed; existing experiment is preserved')
    out.mkdir(parents=True)
    src = out/'source'
    src.mkdir()
    hashes = {}
    for file in [ROOT/'scripts/random_push_env.py',ROOT/'scripts/push_env.py',ROOT/'scripts/train_random.py',ROOT/'models/planar_push.xml']:
        shutil.copy2(file,src/file.name)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    sampler = RandomPushEnv()
    scenes = []
    for count in (0,1,2):
        sampler.counts = (count,)
        for seed in (9001,9002,9003):
            sampler.reset(seed=seed+100*count)
            scenes.append(sampler.scene)
    sampler.close()
    (out/'config.json').write_text(json.dumps({'seed':args.seed,'assistance':args.assistance,
        'stage_steps':args.stage_steps,'curriculum':[[0],[0,1],[0,1,2]],'source_hashes':hashes,
        'validation_scenes':scenes,'final_test':'Not defined or consumed for this extension'},indent=2))

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = time.monotonic()
            self.episodes = []
        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episodes.append({'step':self.num_timesteps,'count':info['obstacle_count'],
                        'success':info['is_success'],'collision':info['collision'],**info['episode']})
            if time.monotonic()-self.last>20:
                print('PPO steps',self.num_timesteps,'episodes',len(self.episodes),flush=True)
                self.last = time.monotonic()
            return True

    raw = RandomPushEnv(counts=(0,),assistance=args.assistance)
    env = Monitor(raw)
    model = PPO('MlpPolicy',env,seed=args.seed,device='cpu',n_steps=256,batch_size=64,
                n_epochs=5,learning_rate=3e-4,policy_kwargs={'net_arch':[64,64]},verbose=0)
    progress = Progress()
    baselines = {}
    for kind in ('none','planner','demo'):
        baselines[kind] = evaluate(None,kind,scenes)
        (out/'baselines.json').write_text(json.dumps(baselines,indent=2))
    for counts in [(0,),(0,1),(0,1,2)]:
        raw.counts = counts
        # Explicit reset ensures a new curriculum stage starts with its own sampler.
        model._last_obs = None
        print('Starting curriculum',counts,flush=True)
        model.learn(total_timesteps=args.stage_steps,callback=progress,reset_num_timesteps=False)
        model.save(out/f'policy_stage_{len(counts)}')
    after = evaluate(model,args.assistance,scenes)
    report = {'baselines':baselines,'after':after,'training_episodes':progress.episodes,
              'total_steps':model.num_timesteps,'limitations':['State observations, not visual recognition',
              'Short single-seed curriculum, not proof of convergence','Nine validation scenes, no final generalization test',
              'Planner/demo assistance is explicitly separate from pure RL','Fixed-size cubes and fixed object physics',
              'Start lower band and goal upper band; direction diversity not yet evaluated']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    env.close()
    print('Completed',out,flush=True)


if __name__=='__main__':
    main()
