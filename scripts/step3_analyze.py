"""Step 3 analysis: blinded labels by unit type.
    python scripts/step3_analyze.py  -> results/step3/labels_unblinded.csv, results/step3/summary.json (and stdout)
"""
import glob, json, os, shutil, sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S3, LAB = os.path.join(ROOT, "results/step3"), "/tmp/john-scratch/step3_labels"
CATS = ["one_job", "one_job_with_related_offshoots", "two_or_three_related_jobs", "unrelated_mixture", "no_coherent_pattern"]
for p in ("r1", "r2"):                       # keep a copy of the raw labels with the results
    os.makedirs(os.path.join(S3, f"labels_{p}"), exist_ok=True)
    for f in glob.glob(os.path.join(LAB, p, "P*.json")):
        shutil.copy(f, os.path.join(S3, f"labels_{p}", os.path.basename(f)))
key = json.load(open(os.path.join(S3, "SECRET_key.json")))
stats = {s["u"]: s for s in (json.loads(l) for l in open(os.path.join(S3, "unit_stats.jsonl")))}
def load(p):
    out = {}
    for f in glob.glob(os.path.join(S3, f"labels_{p}", "P*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:
            print("unreadable", f, e); continue
        out[os.path.basename(f)[:-5]] = d
    return out
r1, r2 = load("r1"), load("r2")
rows = []
for pid, k in key.items():
    d = r1.get(pid)
    if d is None:
        continue
    s = stats[k["u"]]
    rows.append({"pid": pid, **k, "category": d.get("category"), "confidence": d.get("confidence"),
                 "cover_strong": d.get("cover_strong"), "cover_moderate": d.get("cover_moderate"),
                 "repetition_related": bool(d.get("repetition_related")), "explanation": d.get("explanation"),
                 "density30": s["density30"], "top_math_share": s["top_math_share"], "max": s["max"],
                 "category_r2": r2.get(pid, {}).get("category")})
df = pd.DataFrame(rows)
bad = df[~df.category.isin(CATS)]
print(f"labels: {len(df)} of {len(key)} packets; invalid category: {len(bad)}; r2 labels: {int(df.category_r2.notna().sum())}")
df = df[df.category.isin(CATS)].copy()
df["one_job_any"] = df.category.isin(CATS[:2])
df["coherent"] = df.category.isin(CATS[:3])
df.to_csv(os.path.join(S3, "labels_unblinded.csv"), index=False)
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
pd.set_option("display.width", 220)
summ = {}
for scope, sub in (("both layers", df), ("layer 15", df[df.layer == 15]), ("layer 19", df[df.layer == 19])):
    t = []
    for kind in ("edited", "matched", "random", "direction"):
        g = sub[sub.kind == kind]; n = len(g); k = int(g.one_job_any.sum())
        lo, hi = wilson(k, n)
        t.append({"kind": kind, "n": n, "one_job": int((g.category == CATS[0]).sum()), "offshoots": int((g.category == CATS[1]).sum()),
                  "related_jobs": int((g.category == CATS[2]).sum()), "mixture": int((g.category == CATS[3]).sum()),
                  "no_pattern": int((g.category == CATS[4]).sum()), "one_job_any": k, "rate": k / max(n, 1), "ci_lo": lo, "ci_hi": hi,
                  "coherent_rate": g.coherent.mean(), "repetition_related": int(g.repetition_related.sum()),
                  "cover_strong": pd.to_numeric(g.cover_strong, errors="coerce").mean(), "median_density30": g.density30.median()})
    t = pd.DataFrame(t)
    print(f"\n=== {scope} ===\n" + t.round(3).to_string(index=False))
    tests = {}
    for a, b in (("edited", "matched"), ("edited", "random"), ("edited", "direction"), ("matched", "direction"), ("random", "direction")):
        ga, gb = sub[sub.kind == a], sub[sub.kind == b]
        tab = [[int(ga.one_job_any.sum()), int((~ga.one_job_any).sum())], [int(gb.one_job_any.sum()), int((~gb.one_job_any).sum())]]
        tests[f"{a}_vs_{b}"] = float(fisher_exact(tab)[1])
    print("Fisher exact p (one job, with or without offshoots):", {k: round(v, 4) for k, v in tests.items()})
    summ[scope] = {"table": t.to_dict("records"), "fisher_p": tests}
# agreement between passes
both = df[df.category_r2.isin(CATS)]
if len(both):
    a, b = both.one_job_any.values, both.category_r2.isin(CATS[:2]).values
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    kap = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\nagreement r1 vs r2 on {len(both)} units: one-job-any agreement {po:.3f}, kappa {kap:.3f}; exact 5-category agreement {(both.category == both.category_r2).mean():.3f}")
    summ["agreement"] = {"n": int(len(both)), "agree": float(po), "kappa": float(kap), "exact5": float((both.category == both.category_r2).mean())}
# does density predict the label within kind?
print("\none-job rate by density (above / below the kind's median density30):")
for kind in ("edited", "matched", "random", "direction"):
    g = df[df.kind == kind]; m = g.density30.median()
    print(f"  {kind}: denser {g[g.density30 > m].one_job_any.mean():.2f} (n={int((g.density30 > m).sum())}), sparser {g[g.density30 <= m].one_job_any.mean():.2f} (n={int((g.density30 <= m).sum())})")
ed = df[df.kind == "edited"]
print("\nedited neurons labelled repetition-related:")
for _, r in ed[ed.repetition_related].iterrows():
    print(f"  L{r.layer} N{r.idx} [{r.category}] {r.explanation[:200]}")
json.dump(summ, open(os.path.join(S3, "summary.json"), "w"), indent=1)
