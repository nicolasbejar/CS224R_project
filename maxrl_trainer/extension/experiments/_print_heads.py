import json
import os

base = os.path.join(os.path.dirname(__file__), "..", "figures")
d = json.load(open(os.path.join(base, "toy_maxrl_noise_results.json")))
g = d["flip_grid"]
i01 = g.index(0.1)
i02 = g.index(0.2)
print("flip_grid:", g)
for lbl, runs in d["conditions"].items():
    # runs is a list of per-flip entries OR list of run dicts; detect
    if isinstance(runs, list) and runs and isinstance(runs[0], dict) and "pass1_curve" in runs[0]:
        import numpy as np
        arr = np.array([r["pass1_curve"] for r in runs])
        c = arr.mean(axis=0)
    elif isinstance(runs, dict) and "pass1_curve" in runs:
        c = runs["pass1_curve"]
    else:
        c = runs  # already a curve
    print(f"{lbl:32s} p=0:{c[0]:.3f}  p=0.1:{c[i01]:.3f}  p=0.2:{c[i02]:.3f}")
