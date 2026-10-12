"""Summarize small development groups without calling them final tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from random_push_env import clear_segment,route

parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()
root = Path(args.run)
report = json.loads((root/'report.json').read_text())
config = json.loads((root/'config.json').read_text())
groups = {'straight':report['baselines']['none'],'planner':report['baselines']['planner'],
          'demo':report['baselines']['demo'],'ppo_'+config['assistance']:report['after']}
summary = {}
for name,records in groups.items():
    summary[name] = {}
    for count in (0,1,2):
        rows = [r for r in records if r['count']==count]
        summary[name][str(count)] = {'episodes':len(rows),'successes':sum(r['success'] for r in rows),
            'collisions':sum(r['collision'] for r in rows),
            'mean_distance_mm':float(np.mean([r['distance_mm'] for r in rows]))}
scenes = json.loads((root/'config.json').read_text())['validation_scenes']
difficulty = []
for scene in scenes:
    start,goal,obstacles = np.array(scene['start']),np.array(scene['goal']),np.array(scene['obstacles']).reshape(-1,2)
    path = route(start,goal,obstacles)
    difficulty.append({'count':len(obstacles),'direct_corridor_clear':clear_segment(start,goal,obstacles),
                       'planner_waypoints':len(path) if path else None})
(root/'summary.json').write_text(json.dumps({'methods':summary,'difficulty':difficulty},indent=2))
print(json.dumps(summary,indent=2))
