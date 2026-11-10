"""Step 3 corpus pass: per-token values of 512 units over general web text and the model's own math reasoning.

Units, per layer (15, 19), all at the input of mlp.down_proj of the UNEDITED model (bf16):
  edited     the 64 neurons changed by the published edit
  random     64 neurons drawn uniformly from the other neurons
  matched    64 other neurons, nearest to the edited ones in (log activation variance, log index), one per edited
             neuron, without replacement and disjoint from `random`
  direction  64 random unit directions r in neuron space; value = (act - mu) . r, mu = corpus mean activation
Corpus: 2.0 M tokens of FineWeb (documents cut at 1,024 tokens, raw text, no chat template) and the 200 published
unedited-model MATH-500 generations as processed in step 2 (prompt in chat format + generation, loop tails cut
2,048 tokens after onset). The first 4 positions of every document are excluded from statistics.

Writes results/step3/: units.json, values.f16 (memmap [tokens, 512]), tokens.npy, doc.npy, pos.npy, meta.json.

    python -u scripts/step3_pass.py
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402
from collect import cap_of  # noqa: E402

LAYERS, N = (15, 19), 64
OUT = os.path.join(wsp.ROOT, "results/step3")


def pick_units():
    truth = np.load(os.path.join(wsp.ROOT, "results/edit_truth.npz"))
    sc = np.load(os.path.join(wsp.ROOT, "results/step2_scores.npz"))
    rng = np.random.default_rng(0)
    units, dirs = [], {}
    for l in LAYERS:
        ed = truth[f"L{l}_cols"]
        var = sc[f"all|L{l}|baseline|activation_variance"]
        others = np.setdiff1d(np.arange(3584), ed)
        rnd = np.sort(rng.choice(others, N, replace=False))
        pool = np.setdiff1d(others, rnd)
        f = np.stack([np.log(var + 1e-12), np.log(np.arange(3584) + 1.0)], 1)
        f = (f - f[pool].mean(0)) / f[pool].std(0)
        avail, matched = set(pool.tolist()), []
        for j in ed:
            cand = np.array(sorted(avail))
            m = int(cand[np.argmin(((f[cand] - f[j]) ** 2).sum(1))])
            matched.append(m)
            avail.remove(m)
        g = rng.standard_normal((N, 3584))
        dirs[l] = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
        for kind, idx in (("edited", ed), ("random", rnd), ("matched", matched)):
            units += [{"kind": kind, "layer": l, "idx": int(i)} for i in idx]
        units += [{"kind": "direction", "layer": l, "idx": i} for i in range(N)]
    return units, dirs


def main():
    os.makedirs(OUT, exist_ok=True)
    units, dirs = pick_units()
    json.dump(units, open(os.path.join(OUT, "units.json"), "w"))
    np.savez(os.path.join(OUT, "directions.npz"), **{f"L{l}": dirs[l] for l in LAYERS})
    # corpus
    docs = []
    for line in open(os.path.join(wsp.ROOT, "results/data/fineweb_sample.jsonl")):
        d = json.loads(line)
        docs.append(("web", d["id"], torch.tensor(d["ids"])))
    rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
    for i, r in enumerate(rows):
        if r["model"] == "unedited":
            docs.append(("math", f"{i}", torch.cat([r["prompt_ids"], r["gen_ids"][:cap_of(r)]]).long()))
    total = sum(len(d[2]) for d in docs)
    print(f"{len(docs)} documents, {total:,} tokens, {len(units)} units", flush=True)
    vals = np.lib.format.open_memmap(os.path.join(OUT, "values.npy"), mode="w+", dtype=np.float16,
                                     shape=(total, len(units)))
    tokens, doc, pos = np.zeros(total, np.int32), np.zeros(total, np.int32), np.zeros(total, np.int32)

    tok, model = wsp.load("instruct", dtype=torch.bfloat16)
    model.model.visual = None
    model.eval().to("cuda")
    lm = model.model.language_model
    acts = {}
    for l in LAYERS:
        wsp.layers(model)[l].mlp.down_proj.register_forward_pre_hook(
            (lambda l: lambda m, a: acts.__setitem__(l, a[0].detach()[0]))(l))
    col = {l: torch.tensor([u["idx"] for u in units if u["layer"] == l and u["kind"] != "direction"],
                           device="cuda") for l in LAYERS}
    R = {l: torch.from_numpy(dirs[l]).to("cuda") for l in LAYERS}
    # column layout in values: for each layer in order: 192 neurons (edited, random, matched) then 64 directions
    off = {l: sum(1 for u in units if u["layer"] < l) for l in LAYERS}
    mean_sum = {l: torch.zeros(3584, device="cuda", dtype=torch.float64) for l in LAYERS}
    n_valid, cur, t0 = 0, 0, time.time()
    for di, (src, did, ids) in enumerate(docs):
        s, pkv = 0, None
        while s < len(ids):
            step = 1024 if s < 6144 else 512 if s < 14336 else 256
            x = ids[s:s + step].unsqueeze(0).to("cuda")
            with torch.no_grad():
                o = lm(input_ids=x, past_key_values=pkv, use_cache=len(ids) > 1024)
            pkv = o.past_key_values if len(ids) > 1024 else None
            del o
            n = x.shape[1]
            for l in LAYERS:
                a = acts[l].float()
                v = torch.cat([a[:, col[l]], a @ R[l].T], 1)
                vals[cur:cur + n, off[l]:off[l] + v.shape[1]] = v.half().cpu().numpy()
                valid = torch.arange(s, s + n, device="cuda") >= 4
                mean_sum[l] += a[valid].double().sum(0)
            n_valid += int((torch.arange(s, s + n) >= 4).sum())
            tokens[cur:cur + n] = ids[s:s + n].numpy()
            doc[cur:cur + n] = di
            pos[cur:cur + n] = np.arange(s, s + n)
            cur += n
            s += n
        del pkv
        if (di + 1) % 400 == 0 or di + 1 == len(docs):
            torch.cuda.empty_cache()
            print(f"{di + 1}/{len(docs)} docs, {cur:,} tokens, {cur / (time.time() - t0):.0f} tok/s, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    vals.flush()
    np.save(os.path.join(OUT, "tokens.npy"), tokens)
    np.save(os.path.join(OUT, "doc.npy"), doc)
    np.save(os.path.join(OUT, "pos.npy"), pos)
    mu = {l: (mean_sum[l] / n_valid).cpu().numpy() for l in LAYERS}
    json.dump({"docs": [(s, d, len(i)) for s, d, i in docs], "n_valid": n_valid,
               "dir_offset": {str(l): (dirs[l].astype(np.float64) @ mu[l]).tolist() for l in LAYERS},
               "col_offset": {str(l): off[l] for l in LAYERS}},
              open(os.path.join(OUT, "meta.json"), "w"))
    print("PASS_DONE", flush=True)


if __name__ == "__main__":
    main()
