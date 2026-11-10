"""Step 3: per-unit statistics and blinded label packets from the corpus pass (CPU only).

Follows the re-blinded protocol of the already-clean replication (ws1_relabel/make_packets_blind.py):
positive pole only; each excerpt shows 20 tokens before and 4 after the firing token, marked [[like this]], and one
number, the activation as a percentage of the unit's corpus maximum (rounded to 5); 10 "strong" excerpts chosen from
the top 25 to match a common percentage profile and 5 "moderate" excerpts from the 30-60% band; at most 2 excerpts
per document and no duplicate excerpt text; unit ids shuffled; nothing in a packet names the unit type.

    python scripts/step3_packets.py  -> results/step3/unit_stats.jsonl, SECRET_key.json,
                                        /tmp/john-scratch/step3_blind/batch_XX/P###.json
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

S3 = os.path.join(wsp.ROOT, "results/step3")
BLIND = "/tmp/john-scratch/step3_blind"
TOPK, NBAND, BEFORE, AFTER, MAX_PER_DOC, BATCH = 25, 8, 20, 4, 2, 32
TOP_TARGET = [100, 95, 91, 87, 84, 81, 78, 75, 72, 69]
MID_TARGET = [34, 40, 46, 52, 58]

tok = AutoTokenizer.from_pretrained(wsp.snapshot("instruct"))
units = json.load(open(os.path.join(S3, "units.json")))
meta = json.load(open(os.path.join(S3, "meta.json")))
vals = np.load(os.path.join(S3, "values.npy"), mmap_mode="r")
tokens, doc, pos = (np.load(os.path.join(S3, f"{k}.npy")) for k in ("tokens", "doc", "pos"))
src = np.array([d[0] for d in meta["docs"]])
doc_start = np.concatenate([[0], np.cumsum([d[2] for d in meta["docs"]])])
valid = pos >= 4
piece = {}


def tk(i):
    t = int(tokens[i])
    if t not in piece:
        piece[t] = tok.decode([t])
    return piece[t]


def excerpt(i):
    d = doc[i]
    a, b = max(doc_start[d], i - BEFORE), min(doc_start[d + 1], i + AFTER + 1)
    return "".join(tk(j) for j in range(a, i)) + "[[" + tk(i) + "]]" + "".join(tk(j) for j in range(i + 1, b))


def choose(order, v, k):
    """Walk candidate token indices in `order`; keep up to k with <= MAX_PER_DOC per document and distinct text."""
    out, per, seen = [], {}, set()
    for i in order:
        d = int(doc[i])
        if per.get(d, 0) >= MAX_PER_DOC:
            continue
        ex = excerpt(i)
        if ex in seen:
            continue
        seen.add(ex)
        per[d] = per.get(d, 0) + 1
        out.append({"i": int(i), "value": float(v[i]), "text": ex, "src": str(src[d])})
        if len(out) == k:
            break
    return out


def pick(cands, target, ext):
    pct = np.array([100 * c["value"] / ext for c in cands])
    ri, ci = linear_sum_assignment(np.abs(pct[:, None] - np.array(target)[None, :]))
    return [cands[ri[j]] for j in np.argsort(ci)]


rng = np.random.default_rng(0)
stats = []
for ui, u in enumerate(units):
    v = np.asarray(vals[:, ui], dtype=np.float32)
    if u["kind"] == "direction":
        v = v - np.float32(meta["dir_offset"][str(u["layer"])][u["idx"]])
    v[~valid] = -np.inf
    ext = float(v.max())
    top = choose(np.argsort(-v)[:5000], v, TOPK)
    band_idx = np.flatnonzero((v >= 0.3 * ext) & (v <= 0.6 * ext))
    band = choose(rng.permutation(band_idx)[:2000], v, NBAND)
    fin = v[valid]
    stats.append({**u, "u": ui, "max": ext, "min": float(fin.min()), "mean": float(fin.mean()), "std": float(fin.std()),
                  "density30": float((fin >= 0.3 * ext).mean()), "density10": float((fin >= 0.1 * ext).mean()),
                  "top_math_share": float(np.mean([c["src"] == "math" for c in top])),
                  "n_band": int(len(band_idx)), "top": top, "band": band})
    if (ui + 1) % 64 == 0:
        print(f"{ui + 1}/{len(units)} units", flush=True)
with open(os.path.join(S3, "unit_stats.jsonl"), "w") as f:
    for s in stats:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# blinded packets
if os.path.isdir(BLIND):
    shutil.rmtree(BLIND)
perm = np.random.default_rng(11).permutation(len(units))
key, short = {}, 0
for new, old in enumerate(perm.tolist()):
    s = stats[old]
    ext = s["max"]
    top = pick(s["top"], TOP_TARGET, ext)
    mid = pick(s["band"], MID_TARGET, ext) if len(s["band"]) >= len(MID_TARGET) else list(s["band"])
    short += len(mid) < len(MID_TARGET)
    r = np.random.default_rng([11, new])
    fmt = lambda c: {"text": c["text"], "activation_pct": int(5 * round(100 * c["value"] / ext / 5))}  # noqa: E731
    top_f = sorted([fmt(c) for c in top], key=lambda d: -d["activation_pct"])
    mid_f = [fmt(c) for c in mid]
    mid_f = [mid_f[i] for i in r.permutation(len(mid_f))]
    pid = f"P{new:03d}"
    b = os.path.join(BLIND, f"batch_{new // BATCH:02d}")
    os.makedirs(b, exist_ok=True)
    json.dump({"id": pid,
               "about": ("Text excerpts showing where one internal unit of a language model is active. The excerpts "
                         "come from web text and from a model's written reasoning on math problems. In each excerpt "
                         "the token the unit responds to is marked [[like this]], and activation_pct is the unit's "
                         "activation on that token as a percentage of the strongest activation it reached in a "
                         "3.9M-token corpus (rounded to 5). 'strong' lists excerpts from among its strongest "
                         "activations; 'moderate' lists randomly chosen excerpts where it reached 30-60% of that "
                         "maximum."),
               "strong": top_f, "moderate": mid_f}, open(os.path.join(b, f"{pid}.json"), "w"), ensure_ascii=False,
              indent=1)
    key[pid] = {"u": old, "kind": s["kind"], "layer": s["layer"], "idx": s["idx"], "batch": new // BATCH}
json.dump(key, open(os.path.join(S3, "SECRET_key.json"), "w"), indent=1)
print(f"packets {len(key)} in {len(units) // BATCH} batches under {BLIND}; units with fewer than 5 moderate "
      f"excerpts: {short}")
import pandas as pd  # noqa: E402
df = pd.DataFrame([{k: v for k, v in s.items() if k not in ("top", "band")} for s in stats])
print(df.groupby(["layer", "kind"])[["max", "std", "density30", "density10", "top_math_share", "n_band"]].median()
      .round(4).to_string())
