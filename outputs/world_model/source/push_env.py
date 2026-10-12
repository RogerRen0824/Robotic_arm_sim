"""Gymnasium macro-action environment with physical arm/object contact.

Policy selects a pushing direction; fixed IK handles raised repositioning and
contact. State observations are privileged simulator state, not VLA inputs.
"""
import csv
import json
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]

def load_demos(split):
    ids = json.loads((ROOT / "data/splits.json").read_text())[split]
    if split == "test":
        raise ValueError("Final test layouts are reserved; use a separate frozen evaluation")
    demos = []
    for name in ids:
        with (ROOT / "outputs/dataset" / name / "trajectory_prepared.csv").open() as f:
            rows = list(csv.DictReader(f))
        xy = np.array([[float(r["x_m"]),float(r["y_m"])] for r in rows if r["x_m"] != ""])
        xy = np.column_stack((.34+xy[:,0], .10-xy[:,1]))
        path = [xy[0]]
        for point in xy[1:]:
            if np.linalg.norm(point-path[-1]) > .012:
                path.append(point)
        path.append(xy[-1])
        demos.append((name,np.array(path)))
    return demos

class DemoGuide:
    """Nearest-neighbour imitation of object-path direction, training videos only."""
    def __init__(self):
        inputs, outputs = [], []
        for _, path in load_demos("train"):
            for a,b in zip(path[:-1],path[1:]):
                delta = b-a
                if np.linalg.norm(delta) < .003:
                    continue
                inputs.append(np.r_[a-[.45,.035],path[-1]-[.45,.035]])
                outputs.append(delta/np.linalg.norm(delta))
        self.inputs,self.outputs = np.array(inputs),np.array(outputs)
    def predict(self, center, goal, obstacle):
        query = np.r_[center-obstacle,goal-obstacle]
        distances = np.linalg.norm(self.inputs-query,axis=1)
        ids = np.argsort(distances)[:5]
        weights = 1/(distances[ids]+.002)**2
        direction = np.average(self.outputs[ids],axis=0,weights=weights)
        if np.linalg.norm(goal-center) < .04:
            direction = goal-center
        return direction/max(np.linalg.norm(direction),1e-8)

def segment_clear(start,goal,obstacle):
    """Check object-centre segment against a conservatively expanded rectangle."""
    lower = obstacle-np.array([.11,.065])-.028
    upper = obstacle+np.array([.11,.065])+.028
    delta = goal-start
    enter,leave = 0.,1.
    for axis in range(2):
        if abs(delta[axis]) < 1e-9:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return True
        else:
            a,b = (lower[axis]-start[axis])/delta[axis],(upper[axis]-start[axis])/delta[axis]
            enter,leave = max(enter,min(a,b)),min(leave,max(a,b))
            if enter > leave:
                return True
    return False

class PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    def __init__(self, split="train", residual=True, randomize=True, push_iterations=50, goal_weight=.85, residual_weight=.35):
        self.demos = load_demos(split)
        self.guide = DemoGuide()
        self.residual,self.randomize = residual,randomize
        self.push_iterations = push_iterations
        self.goal_weight,self.residual_weight = goal_weight,residual_weight
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "models/planar_push.xml"))
        self.data,self.ik = mujoco.MjData(self.model),mujoco.MjData(self.model)
        self.tip,self.obj = self.model.site("tip").id,self.model.body("object").id
        self.og,self.pg,self.bg = [self.model.geom(n).id for n in ("object_geom","pusher","obstacle")]
        # Approximate square proxy; real dimensions have not been measured.
        self.model.geom_type[self.og] = mujoco.mjtGeom.mjGEOM_BOX
        self.model.geom_size[self.og] = [.023,.023,.025]
        self.model.geom_rbound[self.og] = np.linalg.norm(self.model.geom_size[self.og])
        self.jac = np.zeros((3,self.model.nv))
        self.action_space = spaces.Box(-1,1,shape=(2,),dtype=np.float32)
        self.observation_space = spaces.Box(-10,10,shape=(10,),dtype=np.float32)
        self.renderer = None
        self.previous_action = np.zeros(2)
    def obs(self):
        center = self.data.xpos[self.obj,:2]
        guide = self.base_direction(center) if self.residual else np.zeros(2)
        return np.r_[center-self.obstacle,self.goal-self.obstacle,self.data.site_xpos[self.tip,:2]-center,self.data.qvel[4:6],guide].astype(np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        index = options.get("demo_index",int(self.np_random.integers(len(self.demos))))
        self.demo_id,path = self.demos[index]
        self.obstacle = np.array([.45,.035])
        self.goal = path[-1].copy()
        start = path[0].copy()
        if self.randomize:
            translation = self.np_random.uniform(-.01,.01,2)
            self.obstacle += translation
            self.goal += translation+self.np_random.uniform(-.01,.01,2)
            start += translation+self.np_random.uniform(-.01,.01,2)
        self.model.geom_pos[self.bg,:2] = self.obstacle
        self.model.site_pos[self.model.site("goal").id,:2] = self.goal
        self.model.geom_friction[self.og,0] = self.np_random.uniform(.25,.4) if self.randomize else .3
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:4] = [.1,1.2,-1.3,.04]
        self.data.qpos[4:7] = [*start,.025]
        self.data.qpos[7:11] = [1,0,0,0]
        self.data.ctrl[:] = self.data.qpos[:4]
        mujoco.mj_forward(self.model,self.data)
        self.steps,self.hold = 0,0
        self.previous_action = np.zeros(2)
        self.episode_collision = False
        return self.obs(),{"demo_id":self.demo_id}
    def advance(self,target,steps):
        desired = self.solve_ik(target)
        collision = False
        for _ in range(steps):
            self.data.ctrl[:] += np.clip(desired-self.data.ctrl,-.002,.002)
            mujoco.mj_step(self.model,self.data)
            for contact in self.data.contact:
                pair = {contact.geom1,contact.geom2}
                collision |= self.bg in pair and (self.og in pair or self.pg in pair)
        return bool(collision)
    def solve_ik(self,target):
        """Enumerate analytic 3-link IK branches; prefer nearby joint poses."""
        xy = np.asarray(target[:2])-[0,-.28]
        radial = np.arctan2(xy[1],xy[0])
        candidates = []
        current_phi = np.sum(self.data.qpos[:3])
        for phi in [current_phi,radial,radial+.5*np.pi,radial-.5*np.pi,0,np.pi/2,-np.pi/2,np.pi]:
            wrist = xy-.30*np.array([np.cos(phi),np.sin(phi)])
            cosine = (np.dot(wrist,wrist)-.27**2-.25**2)/(2*.27*.25)
            if abs(cosine) > 1:
                continue
            for sign in (-1,1):
                q2 = sign*np.arccos(np.clip(cosine,-1,1))
                q1 = np.arctan2(wrist[1],wrist[0])-np.arctan2(.25*np.sin(q2),.27+.25*np.cos(q2))
                q3 = phi-q1-q2
                q = (np.array([q1,q2,q3])+np.pi)%(2*np.pi)-np.pi
                if np.all(np.abs(q) < 3.13):
                    cost = np.linalg.norm(q-self.data.qpos[:3])
                    if abs(phi-current_phi) < 1e-6:
                        cost *= .5
                    candidates.append((cost,q))
        if not candidates:
            return self.data.qpos[:4].copy()
        q = min(candidates,key=lambda pair:pair[0])[1]
        return np.r_[q,np.clip(target[2]-.12,-.09,.08)]
    def reposition(self,target,height):
        """Finish lifting before lateral motion, then settle before lowering."""
        collision = False
        # Keep joint angles fixed while lifting; IK must not change branches.
        self.data.ctrl[:3] = self.data.qpos[:3]
        for _ in range(200):
            self.data.ctrl[3] = height-.12
            mujoco.mj_step(self.model,self.data)
        for _ in range(12):
            collision |= self.advance(np.r_[target,height],100)
            if np.linalg.norm(self.data.site_xpos[self.tip,:2]-target) < .003:
                break
        # Keep current horizontal pose while lowering.
        desired_xy = self.data.ctrl[:3].copy()
        for _ in range(300):
            self.data.ctrl[:3] = desired_xy
            mujoco.mj_step(self.model,self.data)
        for _ in range(300):
            self.data.ctrl[3] = -.09
            mujoco.mj_step(self.model,self.data)
            for contact in self.data.contact:
                pair = {contact.geom1,contact.geom2}
                collision |= self.bg in pair and (self.og in pair or self.pg in pair)
        return collision
    def step(self,action):
        action = np.clip(np.asarray(action,float),-1,1)
        before = self.data.xpos[self.obj,:2].copy()
        old_distance = np.linalg.norm(before-self.goal)
        base = self.base_direction(before) if self.residual else np.zeros(2)
        direction = base+self.residual_weight*action if self.residual else action
        length = np.linalg.norm(direction)
        collision = False
        if length > .05:
            direction /= length
            behind = before-direction*(self.contact_offset(direction)+.005)
            collision |= self.reposition(behind,.12)
            for _ in range(self.push_iterations):
                center = self.data.xpos[self.obj,:2].copy()
                if np.linalg.norm(center-self.goal) < .018:
                    break
                collision |= self.advance(np.r_[center-direction*(self.contact_offset(direction)-.001),.03],20)
            # Stop pushing and let the object settle before scoring success.
            self.data.ctrl[:3] = self.data.qpos[:3]
            collision |= self.advance(self.data.site_xpos[self.tip].copy(),100)
        else:
            collision |= self.advance(self.data.site_xpos[self.tip].copy(),100)
        center = self.data.xpos[self.obj,:2]
        distance = float(np.linalg.norm(center-self.goal))
        speed = float(np.linalg.norm(self.data.qvel[4:6]))
        self.hold = self.hold+1 if distance < .025 and speed < .03 else 0
        success = self.hold >= 2
        out_of_bounds = not (.15 < center[0] < .68 and -.28 < center[1] < .28) or self.data.xpos[self.obj,2] > .08
        success = bool(success and not collision and not out_of_bounds)
        self.steps += 1
        self.episode_collision |= collision
        terms = {"progress":40*(old_distance-distance),"success":30*float(success),"collision":-10*float(collision),"time":-.05,"smoothness":-.01*float(np.sum((action-self.previous_action)**2)),"out_of_bounds":-15*float(out_of_bounds)}
        self.previous_action = action.copy()
        terminated = bool(success or collision or out_of_bounds)
        truncated = self.steps >= 100 and not terminated
        info = {"is_success":bool(success and not collision and not out_of_bounds),"collision":bool(self.episode_collision),"distance_m":distance,"reward_terms":terms,"demo_id":self.demo_id,"sim_time":self.data.time}
        return self.obs(),float(sum(terms.values())),terminated,bool(truncated),info
    def base_direction(self,center):
        demo = self.guide.predict(center,self.goal,self.obstacle)
        target = self.goal-center
        target /= max(np.linalg.norm(target),1e-8)
        weight = self.goal_weight if segment_clear(center,self.goal,self.obstacle) else 0.
        direction = (1-weight)*demo+weight*target
        return direction/max(np.linalg.norm(direction),1e-8)
    def contact_offset(self,direction):
        # Ray intersection with the square expanded by the round tool radius.
        rotation = self.data.xmat[self.obj].reshape(3,3)
        local = np.abs((rotation.T@np.r_[direction,0])[:2])
        lo,hi = 0.,.08
        for _ in range(14):
            mid = (lo+hi)/2
            separation = np.linalg.norm(np.maximum(mid*local-.023,0))
            if separation > .012:
                hi = mid
            else:
                lo = mid
        return (lo+hi)/2
    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model,height=480,width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.3,-.03,.02]
        camera.distance,camera.azimuth,camera.elevation = 1.05,90,-65
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render()
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
