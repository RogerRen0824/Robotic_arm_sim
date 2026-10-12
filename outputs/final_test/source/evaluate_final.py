"""One frozen evaluation: test initial positions and independently marked goals only."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from push_env import PushEnv, ROOT


def main():
    out = ROOT / 'outputs/final_test'
    if (out / 'freeze.json').exists():
        raise FileExistsError('Final evaluation already frozen; do not tune and rerun on these tests.')
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / 'outputs/dataset/manifest.json').read_text())
    ids = json.loads((ROOT / 'data/splits.json').read_text())['test']
    scenes = []
    for name in ids:
        record = next(item for item in manifest if item['id'] == name)
        assert record['goal']['source'] == 'first_frame_black_marker'
        with (ROOT / 'outputs/dataset' / name / 'trajectory_prepared.csv').open() as f:
            # Read only the first detected position, never the demonstration endpoint.
            row = next(row for row in csv.DictReader(f) if row['x_m'] != '')
        start = np.array([.34 + float(row['x_m']), .10 - float(row['y_m'])])
        target = record['goal']['notebook_xy_m']
        goal = np.array([.34 + target[0], .10 - target[1]])
        scenes.append((name, np.array([start, goal])))
    out.mkdir(parents=True)
    source = out / 'source'
    source.mkdir()
    files = [ROOT/'scripts/push_env.py', Path(__file__), ROOT/'models/planar_push.xml',
             ROOT/'data/splits.json', ROOT/'data/goal_annotations.json',
             ROOT/'outputs/dataset/manifest.json', ROOT/'outputs/rl/residual/9/policy.zip']
    hashes = {}
    for file in files:
        shutil.copy2(file, source / file.name)
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    protocol = {'source_hashes': hashes, 'goal_weight': .85, 'residual_weight': .35,
                'policy_seed': 9, 'friction': .3, 'max_macro_steps': 100,
                'scenes': [{'id': n, 'start_m': p[0].tolist(), 'goal_m': p[-1].tolist()} for n,p in scenes],
                'limitations': ['Five layouts only; no statistical generalization claim',
                    'Manual approximate goal annotations; planar camera calibration',
                    'Notebook coordinates normalize camera/obstacle rotation',
                    'Privileged simulator state; arm link collisions disabled',
                    'Same contact controller for all policies; no test trajectory replay']}
    (out/'freeze.json').write_text(json.dumps(protocol, indent=2))
    model = PPO.load(ROOT/'outputs/rl/residual/9/policy.zip', device='cpu')
    all_results = {}
    for kind in ['straight', 'demo_guide', 'weighted_guide', 'ppo_residual']:
        env = PushEnv(split='validation', residual=kind != 'straight', randomize=False,
                      goal_weight=0. if kind == 'demo_guide' else .85, residual_weight=.35)
        env.demos = scenes
        results = []
        for index in range(len(scenes)):
            obs, _ = env.reset(seed=500+index, options={'demo_index': index})
            trajectory = [env.data.xpos[env.obj,:2].copy().tolist()]
            for step in range(100):
                if kind == 'ppo_residual':
                    action, _ = model.predict(obs, deterministic=True)
                elif kind == 'straight':
                    direction = env.goal - env.data.xpos[env.obj,:2]
                    action = direction / max(np.linalg.norm(direction), 1e-8)
                else:
                    action = np.zeros(2)
                obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append(env.data.xpos[env.obj,:2].copy().tolist())
                if terminated or truncated:
                    break
            result = {'demo_id': info['demo_id'], 'success': info['is_success'],
                      'collision': info['collision'], 'distance_m': info['distance_m'],
                      'steps': step+1, 'sim_time': info['sim_time'],
                      'termination': 'success' if info['is_success'] else
                          'collision' if info['collision'] else 'timeout' if truncated else 'escape',
                      'trajectory_m': trajectory}
            results.append(result)
            print(kind, index+1, result['termination'], round(info['distance_m']*1000,1), 'mm', flush=True)
            (out/f'{kind}.json').write_text(json.dumps(results, indent=2))
        env.close()
        all_results[kind] = {'successes': sum(r['success'] for r in results),
                            'collisions': sum(r['collision'] for r in results),
                            'mean_distance_mm': float(np.mean([r['distance_m'] for r in results])*1000),
                            'mean_steps': float(np.mean([r['steps'] for r in results])),
                            'episodes': results}
    (out/'report.json').write_text(json.dumps(all_results, indent=2))
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'episodes'} for k,v in all_results.items()}), flush=True)


if __name__ == '__main__':
    main()
