"""Plot measured validation fidelity; runnable with any matplotlib environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root/"outputs/world_model"
report = json.loads((out/"report.json").read_text())
fig,axes = plt.subplots(1,2,figsize=(12,4.5))
names = {"persistence":"No movement","linear_ridge":"Linear ridge","action_blind":"MLP without action","action_conditioned":"Action-conditioned MLP"}
for name,label in names.items():
    axes[0].plot([1,3,5],[report["metrics"][name][str(h)]["position_rmse_mm"] for h in [1,3,5]],marker="o",label=label)
axes[0].set(xlabel="Prediction horizon (macro-actions)",ylabel="Object position RMSE (mm)",title="Held-out validation episodes")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=.2)
examples = json.loads((out/"rollout_examples.json").read_text())
actual,predicted = np.array(examples["actual_xy"]),np.array(examples["predicted_xy"])
axes[1].scatter(actual[:,0]*1000,actual[:,1]*1000,s=12,label="Actual 5-step endpoint",alpha=.6)
axes[1].scatter(predicted[:,0]*1000,predicted[:,1]*1000,s=12,label="Predicted 5-step endpoint",alpha=.6)
axes[1].set(xlabel="Workspace x (mm)",ylabel="Workspace y (mm)",title="Action-conditioned predictions")
axes[1].set_aspect("equal",adjustable="datalim")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=.2)
fig.suptitle("Simulation state world model — final test layouts excluded",fontsize=12)
fig.tight_layout()
fig.savefig(out/"prediction_fidelity.png",dpi=160)
fig.savefig(out/"prediction_fidelity.svg")
