"""Plot saved random validation layouts; optional matplotlib dependency."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()
root = Path(args.run)
scenes = json.loads((root/'config.json').read_text())['validation_scenes']
fig,axes = plt.subplots(3,3,figsize=(10,10),sharex=True,sharey=True)
for ax,scene in zip(axes.flat,scenes):
    ax.add_patch(Rectangle((.28,-.20),.25,.06,color='royalblue',alpha=.12))
    ax.add_patch(Rectangle((.28,.14),.25,.06,color='green',alpha=.12))
    for x,y in scene['obstacles']:
        ax.add_patch(Rectangle((x-.03,y-.03),.06,.06,color='saddlebrown'))
        ax.add_patch(Rectangle((x-.077,y-.077),.154,.154,fill=False,linestyle=':',edgecolor='gray'))
    x,y = scene['start']
    ax.add_patch(Rectangle((x-.023,y-.023),.046,.046,color='royalblue'))
    ax.add_patch(Circle(scene['goal'],.025,color='green',alpha=.7))
    ax.plot([scene['start'][0],scene['goal'][0]],[scene['start'][1],scene['goal'][1]],'k--',alpha=.4)
    ax.set(xlim=(.22,.59),ylim=(-.23,.23),aspect='equal',title=f"{len(scene['obstacles'])} obstacles")
    ax.grid(alpha=.2)
for ax in axes[-1]: ax.set_xlabel('Workspace x (m)')
for ax in axes[:,0]: ax.set_ylabel('Workspace y (m)')
fig.suptitle('Random validation scenes: fixed object, independent starts/goals\nBlue/green bands: sampling regions; dotted boxes: planning clearance',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(root/'validation_scenes.png',dpi=150)
