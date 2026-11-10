import json, pandas as pd, numpy as np, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "results/gen_eval.jsonl"))])
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return f"{(c - h) / d:.2f}-{(c + h) / d:.2f}"
rows = []
for v, g in df.groupby("variant", sort=False):
    n = len(g)
    rows.append({"variant": v, "n": n, "looped": int(g.looped.sum()), "looped_ci": wil(int(g.looped.sum()), n), "finished": int(g.finished.sum()),
                 "correct": int(g.correct.sum()), "mean_repeat": round(g.repeat_frac.mean(), 3), "median_tokens": int(g.n_tokens.median())})
t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(os.path.join(ROOT, "results/gen_eval_table.csv"), index=False)
