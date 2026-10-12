"""Semantic environment checks, independent of training returns."""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from push_env import PushEnv,segment_clear

env = PushEnv(randomize=False)
check_env(env,warn=True)
obs,_ = env.reset(seed=4,options={"demo_index":0})
state = env.data.qpos.copy()
obs2,_ = env.reset(seed=4,options={"demo_index":0})
assert np.allclose(obs,obs2) and np.allclose(state,env.data.qpos)
assert all(name not in env.guide.__dict__ for name in ("test", "validation"))
# Place object in obstacle only in this diagnostic, never in a rollout policy.
env.data.qpos[4:6] = env.obstacle
import mujoco
mujoco.mj_forward(env.model,env.data)
_,reward,terminated,truncated,info = env.step(np.zeros(2))
assert info["collision"] and terminated
assert info["reward_terms"]["collision"] == -10
assert not info["is_success"]
# Goal achievement must persist over two steps, and require low object speed.
env.reset(seed=4,options={"demo_index":0})
env.goal = env.data.xpos[env.obj,:2].copy()
env.residual = False
_,_,terminated,_,info = env.step(np.zeros(2))
assert not info["is_success"]
_,_,terminated,_,info = env.step(np.zeros(2))
assert info["is_success"] and terminated
# Collision must override success even when the target is inside the obstacle.
env.reset(seed=4,options={"demo_index":0})
env.residual = True
env.data.qpos[4:6] = env.obstacle
mujoco.mj_forward(env.model,env.data)
env.goal = env.obstacle.copy()
env.hold = 1
_,_,terminated,_,info = env.step(np.zeros(2))
assert info["collision"] and not info["is_success"]
assert info["reward_terms"]["success"] == 0
assert not segment_clear(np.array([.45,-.15]),np.array([.45,.16]),np.array([.45,.035]))
assert segment_clear(np.array([.25,.16]),np.array([.50,.16]),np.array([.45,.035]))
assert not segment_clear(np.array([.45,.035]),np.array([.46,.04]),np.array([.45,.035]))
env.close()
print("Passed Gymnasium/SB3 checks, deterministic reset, obstacle penalty, and stable-goal termination")
