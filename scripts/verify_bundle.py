"""Read-only bundle integrity and inference checks; never evaluate final policies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from stable_baselines3 import PPO
from push_env import ROOT, PushEnv, DemoGuide
from world_model import features


def main():
    manifest = ROOT/'bundle_manifest.json'
    if manifest.exists():
        for entry in json.loads(manifest.read_text()):
            file = ROOT/entry['path']
            assert file.is_file(), entry['path']
            assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    splits = json.loads((ROOT/'data/splits.json').read_text())
    assert not set(splits['train']) & set(splits['validation'])
    assert not (set(splits['train']) | set(splits['validation'])) & set(splits['test'])
    subprocess.run([sys.executable,str(ROOT/'scripts/check_rl_env.py')],check=True)
    env = PushEnv(split='validation',randomize=False)
    obs, _ = env.reset(seed=100,options={'demo_index':0})
    model = PPO.load(ROOT/'outputs/rl/residual/9/policy.zip',device='cpu')
    action,_ = model.predict(obs,deterministic=True)
    assert action.shape == (2,) and np.isfinite(action).all()
    env.close()
    val = np.load(ROOT/'outputs/world_model/validation.npz')
    assert set(val['demo_id']) <= set(splits['validation'])
    params = np.load(ROOT/'outputs/world_model/normalization_and_ridge.npz')
    x = features(val['states'],val['contexts'],val['actions'],DemoGuide())
    normalized = (x-params['x_mean'])/params['x_std']
    delta = np.column_stack((normalized,np.ones(len(x))))@params['ridge']
    pred = val['states']+delta*params['delta_std']+params['delta_mean']
    rmse = float(1000*np.sqrt(np.mean(np.sum((pred[:,4:6]-val['next_states'][:,4:6])**2,axis=1))))
    expected = json.loads((ROOT/'outputs/world_model/report.json').read_text())['metrics']['linear_ridge']['1']['position_rmse_mm']
    assert abs(rmse-expected) < .001, (rmse,expected)
    print(f'Bundle checks passed; saved linear model one-step RMSE {rmse:.4f} mm. Final test not rerun.')


if __name__ == '__main__':
    main()
