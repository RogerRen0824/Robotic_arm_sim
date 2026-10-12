"""Random 0--2 cube tasks; simulator-state observations, not visual recognition.

The legacy experiment remains unchanged. Optional planner assistance is explicit.
"""
import heapq
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from push_env import PushEnv, ROOT

BOUNDS = np.array([[.22,-.23],[.59,.23]])
HALF = .03
INFLATION = .047


def clear_segment(a,b,obstacles):
    for center in obstacles:
        lower,upper = center-HALF-INFLATION,center+HALF+INFLATION
        delta = b-a
        enter,leave = 0.,1.
        for k in range(2):
            if abs(delta[k]) < 1e-10:
                if not lower[k] <= a[k] <= upper[k]:
                    enter,leave = 1.,0.
                    break
            else:
                t = sorted([(lower[k]-a[k])/delta[k],(upper[k]-a[k])/delta[k]])
                enter,leave = max(enter,t[0]),min(leave,t[1])
        if enter <= leave:
            return False
    return True


def route(start,goal,obstacles):
    """Visibility graph baseline; never described as a learned avoidance policy."""
    points = [np.asarray(start),np.asarray(goal)]
    for center in obstacles:
        for offset in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            p = center+np.array(offset)*(HALF+INFLATION+.004)
            if np.all(p>BOUNDS[0]) and np.all(p<BOUNDS[1]) and clear_segment(p,p,obstacles):
                points.append(p)
    distances = [float('inf')]*len(points)
    distances[0] = 0
    previous = {}
    queue = [(0.,0)]
    while queue:
        dist,i = heapq.heappop(queue)
        if dist != distances[i]:
            continue
        if i == 1:
            path = [points[1]]
            while i != 0:
                i = previous[i]
                path.append(points[i])
            return list(reversed(path))
        for j,p in enumerate(points):
            new = dist+float(np.linalg.norm(p-points[i]))
            if j != i and new < distances[j] and clear_segment(points[i],p,obstacles):
                distances[j],previous[j] = new,i
                heapq.heappush(queue,(new,j))
    return None


class RandomPushEnv(PushEnv):
    def __init__(self,counts=(0,1,2),assistance='none'):
        super().__init__(split='train',residual=False,randomize=False)
        assert assistance in ('none','planner','demo')
        self.counts,self.assistance = tuple(counts),assistance
        xml = ET.parse(ROOT/'models/planar_push.xml')
        world = xml.getroot().find('worldbody')
        old = world.find("geom[@name='obstacle']")
        old.set('size','.03 .03 .03')
        ET.SubElement(world,'geom',name='obstacle_2',type='box',pos='.4 0 .03',
                      size='.03 .03 .03',rgba='.6 .25 .12 1')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(),encoding='unicode'))
        self.data,self.ik = mujoco.MjData(self.model),mujoco.MjData(self.model)
        self.tip,self.obj = self.model.site('tip').id,self.model.body('object').id
        self.og,self.pg,self.bg = [self.model.geom(n).id for n in ('object_geom','pusher','obstacle')]
        self.obstacle_ids = [self.bg,self.model.geom('obstacle_2').id]
        self.model.geom_type[self.og] = mujoco.mjtGeom.mjGEOM_BOX
        self.model.geom_size[self.og] = [.023,.023,.025]
        self.model.geom_rbound[self.og] = np.linalg.norm(self.model.geom_size[self.og])
        self.observation_space = spaces.Box(-10,10,shape=(20,),dtype=np.float32)
        self.active = np.empty((0,2))

    def base_direction(self,center):
        if self.assistance == 'none':
            return np.zeros(2)
        path = route(center,self.goal,self.active)
        if path is None:
            return np.zeros(2)
        direction = path[1]-center
        if self.assistance == 'demo' and len(self.active) and not clear_segment(center,self.goal,self.active):
            # Transfer only local direction, scaled into the legacy notebook frame.
            obstacle = min(self.active,key=lambda p:np.linalg.norm(p-center))
            legacy = self.guide.predict(obstacle+(center-obstacle)*2,
                                        obstacle+(self.goal-obstacle)*2,obstacle)
            direction = .85*direction/max(np.linalg.norm(direction),1e-8)+.15*legacy
        return direction/max(np.linalg.norm(direction),1e-8)

    def obs(self):
        center = self.data.xpos[self.obj,:2]
        slots = []
        # Canonical order prevents arbitrary obstacle labels changing the task.
        for k in range(2):
            slots.extend([*(self.active[k]-center),HALF,HALF,1.] if k<len(self.active) else [0.]*5)
        return np.r_[center,self.goal-center,self.data.site_xpos[self.tip,:2]-center,
                     self.data.qvel[4:6],slots,self.base_direction(center)].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        options = options or {}
        if 'scene' in options:
            scene = options['scene']
            start,goal = np.array(scene['start']),np.array(scene['goal'])
            active = np.array(scene['obstacles']).reshape(-1,2)
            assert len(active)<=2 and route(start,goal,active) is not None
        else:
            count = int(self.np_random.choice(self.counts))
            for _ in range(2000):
                start = self.np_random.uniform([.28,-.20],[.53,-.14])
                goal = self.np_random.uniform([.28,.14],[.53,.20])
                active = self.np_random.uniform([.27,-.09],[.54,.09],size=(count,2))
                if count==2 and np.linalg.norm(active[0]-active[1]) < .165:
                    continue
                if clear_segment(start,start,active) and clear_segment(goal,goal,active) and route(start,goal,active) is not None:
                    break
            else:
                raise RuntimeError('Could not sample reachable scene')
        self.active = np.array(sorted(active.tolist()),float).reshape(-1,2)
        self.goal,self.obstacle = goal,np.array([.45,.035])
        self.demo_id = 'random_'+str(len(self.active))
        for k,geom in enumerate(self.obstacle_ids):
            present = k<len(self.active)
            self.model.geom_pos[geom] = [*(self.active[k] if present else [2.,2.]),.03]
            self.model.geom_contype[geom] = int(present)
            self.model.geom_conaffinity[geom] = int(present)
            self.model.geom_rgba[geom,3] = float(present)
        self.model.site_pos[self.model.site('goal').id,:2] = goal
        self.model.geom_friction[self.og,0] = .3
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:4] = [.1,1.2,-1.3,.04]
        self.data.qpos[4:7],self.data.qpos[7:11] = [*start,.025],[1,0,0,0]
        self.data.ctrl[:] = self.data.qpos[:4]
        mujoco.mj_forward(self.model,self.data)
        self.steps,self.hold = 0,0
        self.previous_action = np.zeros(2)
        self.episode_collision = False
        self.residual = self.assistance != 'none'
        self.scene = {'start':start.tolist(),'goal':goal.tolist(),'obstacles':self.active.tolist()}
        return self.obs(),{'scene':self.scene}

    def touching_obstacle(self):
        active_ids = set(self.obstacle_ids[:len(self.active)])
        return any(bool({c.geom1,c.geom2}&active_ids) and bool({c.geom1,c.geom2}&{self.og,self.pg}) for c in self.data.contact)

    def advance(self,target,steps):
        desired = self.solve_ik(target)
        collision = False
        for _ in range(steps):
            self.data.ctrl[:] += np.clip(desired-self.data.ctrl,-.002,.002)
            mujoco.mj_step(self.model,self.data)
            collision |= self.touching_obstacle()
        return bool(collision)

    def reposition(self,target,height):
        collision = False
        self.data.ctrl[:3] = self.data.qpos[:3]
        for _ in range(200):
            self.data.ctrl[3] = height-.12
            mujoco.mj_step(self.model,self.data)
            collision |= self.touching_obstacle()
        for _ in range(12):
            collision |= self.advance(np.r_[target,height],100)
            if np.linalg.norm(self.data.site_xpos[self.tip,:2]-target)<.003:
                break
        desired_xy = self.data.ctrl[:3].copy()
        for _ in range(300):
            self.data.ctrl[:3] = desired_xy
            mujoco.mj_step(self.model,self.data)
            collision |= self.touching_obstacle()
        for _ in range(300):
            self.data.ctrl[3] = -.09
            mujoco.mj_step(self.model,self.data)
            collision |= self.touching_obstacle()
        return bool(collision)

    def step(self,action):
        obs,reward,terminated,truncated,info = super().step(action)
        info.update(obstacle_count=len(self.active),assistance=self.assistance)
        return obs,reward,terminated,truncated,info
