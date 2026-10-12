"""Check sampling, obstacle masks, second-obstacle collision and API behavior."""
import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env
from random_push_env import RandomPushEnv,route,clear_segment

env = RandomPushEnv()
check_env(env,warn=True)
for count in (0,1,2):
    env.counts = (count,)
    for seed in range(40):
        obs,_ = env.reset(seed=seed)
        assert obs.shape == (20,) and np.isfinite(obs).all()
        assert len(env.active)==count
        assert clear_segment(env.data.qpos[4:6],env.data.qpos[4:6],env.active)
        assert clear_segment(env.goal,env.goal,env.active)
        assert route(env.data.qpos[4:6],env.goal,env.active) is not None
        scene = env.scene
        again,_ = env.reset(seed=seed)
        assert scene==env.scene and np.array_equal(obs,again)
        assert obs[12]==int(count>=1) and obs[17]==int(count==2)
        assert all(env.model.geom_contype[g]==int(k<count) for k,g in enumerate(env.obstacle_ids))
env.reset(seed=33,options={'scene':{'start':[.4,-.18],'goal':[.4,.18],'obstacles':[[.3,0],[.5,0]]}})
before = env.obs().copy()
env.reset(options={'scene':{'start':[.4,-.18],'goal':[.4,.18],'obstacles':[[.5,0],[.3,0]]}})
assert np.array_equal(before,env.obs()), 'Obstacle permutation must preserve observations'
env.data.qpos[4:6] = env.active[1]
mujoco.mj_forward(env.model,env.data)
assert env.touching_obstacle(), 'Second obstacle contact must be measured'
_,_,terminated,_,info = env.step(np.zeros(2))
assert terminated and info['collision'] and not info['is_success']
env.close()
print('Passed API, 120 seeded scenes, masks, deterministic reset and second-obstacle collision.')
