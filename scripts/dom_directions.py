"""Difference-of-means residual directions from rollout-level contrasts (CPU; uses the step 2 per-text statistics).

Each rollout contributes one vector: the mean residual stream (after layer 15 / 19) over its productive and loop
tokens (gray tokens were not stored in step 2). A direction = mean over positive rollouts - mean over negative ones.
  loop      loop tokens vs productive tokens within looped rollouts (step 2's direction; uses the loop labels)
  incorrect incorrect vs correct rollouts of the unedited model (no loop labels used)
  cut       rollouts that hit the 32,768-token limit vs rollouts that finished (no grading, no loop labels)
  vsedited  unedited-model rollouts vs edited-model rollouts on the same problems (uses the published fix)
All 200 published MATH-500 first samples per model are used. -> results/directions/<name>_L<layer>.npy (unit norm)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp
from analyze_step2 import load
rows = load()
summ = {(s["model"], s["id"]): s for s in (json.loads(l) for l in open(os.path.join(wsp.ROOT, "results/data/gens_summary.jsonl")))}
truth = np.load(os.path.join(wsp.ROOT, "results/edit_truth.npz"))
os.makedirs(os.path.join(wsp.ROOT, "results/directions"), exist_ok=True)
def vec(r, l, cls=(1, 3)):
    n = sum(r["stats"][f"res{l}_c{c}"][0] for c in cls)
    return sum(r["stats"][f"res{l}_c{c}"][1].double().numpy() for c in cls) / n if n >= 64 else None
def mean(rs, l, cls=(1, 3)):
    v = [vec(r, l, cls) for r in rs]; v = [x for x in v if x is not None]
    return np.mean(v, 0), len(v)
un = [r for r in rows if r["model"] == "unedited"]; ed = [r for r in rows if r["model"] == "edited"]
flag = lambda r, k: summ[(r["model"], r["id"])][k]
groups = {"loop": None,
          "incorrect": ([r for r in un if not flag(r, "first_sample_correct")], [r for r in un if flag(r, "first_sample_correct")]),
          "cut": ([r for r in un if flag(r, "cut")], [r for r in un if not flag(r, "cut")]),
          "vsedited": (un, ed)}
D, info = {}, {}
for l in (15, 19):
    for name, g in groups.items():
        if g is None:
            L = [r for r in un if r["onset"] is not None]
            a, na = mean(L, l, (3,)); b, nb = mean(L, l, (1,))
        else:
            a, na = mean(g[0], l); b, nb = mean(g[1], l)
        d = a - b
        info[f"{name}_L{l}"] = {"n_pos": na, "n_neg": nb, "raw_norm": float(np.linalg.norm(d))}
        D[(name, l)] = d / np.linalg.norm(d)
        np.save(os.path.join(wsp.ROOT, f"results/directions/{name}_L{l}.npy"), D[(name, l)])
    rng = np.random.default_rng(0); v = rng.standard_normal(1024)
    np.save(os.path.join(wsp.ROOT, f"results/directions/random_L{l}.npy"), v / np.linalg.norm(v))
    np.save(os.path.join(wsp.ROOT, f"results/directions/published_L{l}.npy"), truth[f"L{l}_u"] * np.sign(truth[f"L{l}_u"] @ D[("loop", l)]))
    names = list(groups) ; pub = truth[f"L{l}_u"]
    print(f"layer {l}: rollouts per contrast " + ", ".join(f"{n} {info[f'{n}_L{l}']['n_pos']}/{info[f'{n}_L{l}']['n_neg']}" for n in names))
    print("   raw difference norms: " + ", ".join(f"{n} {info[f'{n}_L{l}']['raw_norm']:.3f}" for n in names))
    print("   cosine matrix (rows/cols: " + ", ".join(names) + ", published u):")
    M = np.array([[D[(a, l)] @ D[(b, l)] for b in names] + [abs(D[(a, l)] @ pub)] for a in names])
    for n, row in zip(names, M): print("     " + f"{n:10s}" + " ".join(f"{x:6.3f}" for x in row))
json.dump(info, open(os.path.join(wsp.ROOT, "results/directions/info.json"), "w"), indent=1)
