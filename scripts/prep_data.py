"""Build the step 2 dataset from the published generations (no model run).

Sources: exp8_math/eval/{unedited,p50c}/parts/math500_s*of10/math500_first_samples.jsonl.gz (first sample of each
MATH-500 problem, 200 problems per model) and the MATH-500 problem texts (HuggingFaceH4/MATH-500).

Per generation: token ids of prompt + generation, whether it hit the 32,768-token limit, and a per-token repeat flag
r[t] = the 16-token window ending at t already occurred, ending within the previous 2,048 tokens. Loop onset = the first
generated token from which every later 256-token window has a repeat rate of at least 0.8.

    python scripts/prep_data.py      -> results/data/gens.pt, results/data/gens_summary.jsonl
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402

NGRAM, WINDOW, WIN_FRAC, MIN_TAIL, LIMIT = 16, 2048, 0.8, 256, 32768
EV = os.path.join(wsp.ROOT, "ext/lm_understand-logs/exp8_math/eval")


def repeat_flags(ids):
    """r[t] = 1 if ids[t-NGRAM+1 : t+1] also ends at some t' with 0 < t - t' <= WINDOW."""
    r = np.zeros(len(ids), dtype=np.uint8)
    last = {}
    for t in range(NGRAM - 1, len(ids)):
        g = tuple(ids[t - NGRAM + 1:t + 1])
        p = last.get(g)
        if p is not None and t - p <= WINDOW:
            r[t] = 1
        last[g] = t
    return r


def onset(r):
    """Strict loop onset: the first t such that every 256-token window starting at or after t has a repeat rate of
    at least WIN_FRAC, with at least MIN_TAIL tokens remaining. None if there is none."""
    n = len(r)
    if n < 2 * MIN_TAIL:
        return None
    c = np.concatenate([[0], np.cumsum(r)])
    w = (c[MIN_TAIL:] - c[:-MIN_TAIL]) / MIN_TAIL          # w[t] = mean r[t:t+256], t = 0 .. n-256
    smin = np.minimum.accumulate(w[::-1])[::-1]
    idx = np.flatnonzero(smin >= WIN_FRAC)
    return int(idx[0]) if len(idx) else None


def local_rate(r, half=128):
    """Centered moving average of r (window 2*half), same length as r."""
    c = np.concatenate([[0], np.cumsum(r)])
    i = np.arange(len(r))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half, len(r))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    os.environ.pop("HF_HUB_OFFLINE", None)
    tok = AutoTokenizer.from_pretrained(wsp.snapshot("instruct"))
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prob = {r["unique_id"]: r for r in ds}
    rows, summ = [], []
    for model, run in (("unedited", "unedited"), ("edited", "p50c")):
        files = sorted(glob.glob(f"{EV}/{run}/parts/math500_s*of10/math500_first_samples.jsonl.gz"))
        correct = {}
        for f in sorted(glob.glob(f"{EV}/{run}/parts/math500_s*of10/math500_per_problem.json")):
            for p in json.load(open(f)):
                correct[p["id"]] = p["correct"]
        for f in files:
            for line in gzip.open(f, "rt"):
                d = json.loads(line)
                p = prob[d["id"]]
                assert p["answer"] == d["answer"], d["id"]
                prompt_ids = tok(wsp.render(tok, p["problem"]), add_special_tokens=False).input_ids
                gen_ids = tok(d["text"], add_special_tokens=False).input_ids
                r = repeat_flags(gen_ids)
                on = onset(r)
                cut = len(gen_ids) >= LIMIT - 8
                s = {"model": model, "id": d["id"], "subject": p["subject"], "level": p["level"],
                     "n_prompt": len(prompt_ids), "n_gen": len(gen_ids), "cut": bool(cut),
                     "think_closed": "</think>" in d["text"], "onset": on,
                     "repeat_frac": float(r.mean()) if len(r) else 0.0,
                     "first_sample_correct": int(correct[d["id"]][0]), "chars": len(d["text"])}
                summ.append(s)
                rows.append({**s, "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
                             "gen_ids": torch.tensor(gen_ids, dtype=torch.int32),
                             "repeat": torch.from_numpy(r)})
    os.makedirs(os.path.join(wsp.ROOT, "results/data"), exist_ok=True)
    torch.save(rows, os.path.join(wsp.ROOT, "results/data/gens.pt"))
    with open(os.path.join(wsp.ROOT, "results/data/gens_summary.jsonl"), "w") as f:
        for s in summ:
            f.write(json.dumps(s) + "\n")
    import pandas as pd
    df = pd.DataFrame(summ)
    df["loop"] = df.onset.notna()
    print(df.groupby("model").agg(n=("id", "size"), cut=("cut", "mean"), loop=("loop", "mean"),
                                  closed=("think_closed", "mean"), correct=("first_sample_correct", "mean"),
                                  n_gen_median=("n_gen", "median"), n_gen_max=("n_gen", "max"),
                                  tokens_total=("n_gen", "sum")).to_string())
    print("\ncut x loop:\n", pd.crosstab([df.model, df.cut], df.loop).to_string())
    lo = df[df.loop]
    print("\nloop onset (generated-token index), looped generations:")
    print(lo.groupby("model").onset.describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_string())
    print("\nrepeat fraction before onset (looped) / overall (not looped):")
    for m in ("unedited", "edited"):
        pre = [float(r["repeat"][:r["onset"]].float().mean()) for r in rows
               if r["model"] == m and r["onset"] not in (None, 0)]
        non = [r["repeat_frac"] for r in rows if r["model"] == m and r["onset"] is None]
        print(f"  {m}: pre-onset mean {np.mean(pre):.3f} (n={len(pre)}); non-looped mean "
              f"{np.mean(non) if non else float('nan'):.3f} (n={len(non)})")


if __name__ == "__main__":
    main()
