"""Step 4, cheap proxy: how much does each candidate weight edit lower the probability of the loop text, and how
much does it disturb productive text?

For each held-out looped generation of the unedited model (published text), teacher-force prompt + generation up to
512 tokens past the loop onset through the model with the edit applied, and record the log-prob of each actual next
token. Report, relative to the unedited model, the mean change in log-prob per token on
  loop tokens        (the 512 tokens from the loop onset on), and
  productive tokens  (class 1 tokens before the onset).
A useful edit makes loop text much less likely and leaves productive text alone. This is the editor's own screening
metric (work_notes: "repetitive branch logp" vs "productive branches").

Edits act on mlp.down_proj of layers 15 and 19 together. Rankings and the direction u_hat (difference of mean
residual after the layer, loop minus productive tokens) are fit on the discovery half of the problems only; the
evaluation texts are from the held-out half.

    python -u scripts/step4_proxy.py        -> results/step4_proxy.jsonl (append-only, resumable)
"""
import glob
import json
import os
import sys

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402
from analyze_step2 import contrast, load as load_stats  # noqa: E402
from collect import classes  # noqa: E402

LAYERS, N_TEXTS, MAX_ONSET, LOOP_SPAN, CHUNK, ALPHA = (15, 19), 24, 8000, 512, 1024, 6.0
SMOKE = os.environ.get("WSP_SMOKE") == "1"
if SMOKE:
    N_TEXTS = 2
OUT = os.path.join(wsp.ROOT, "results/step4_proxy_smoke.jsonl" if SMOKE else "results/step4_proxy.jsonl")


def fit_discovery():
    """u_hat and neuron rankings per layer, from the discovery half only."""
    rows = [r for r in load_stats() if r["split"] == 0]
    un = [r for r in rows if r["model"] == "unedited"]
    L, N = [r for r in un if r["onset"] is not None], [r for r in un if r["onset"] is None]
    fit = {}
    for l in LAYERS:
        rc = contrast(L, (3,), L, (1,), f"res{l}")
        uhat = rc["diff"] / np.linalg.norm(rc["diff"])
        d_act = contrast(L, (3,), L, (1,), f"act{l}")["diff"]
        d_between = contrast(L, (1, 3), N, (1,), f"act{l}")["diff"]
        fit[l] = {"uhat": uhat, "d_act": d_act, "d_between": d_between}
    return fit


def main():
    rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
    truth = np.load(os.path.join(wsp.ROOT, "results/edit_truth.npz"))
    from collect import split_of
    cand = [i for i, r in enumerate(rows) if r["model"] == "unedited" and r["onset"] is not None
            and r["onset"] <= MAX_ONSET and split_of(r["id"]) == 1]
    rng = np.random.default_rng(0)
    texts = sorted(rng.choice(cand, size=min(N_TEXTS, len(cand)), replace=False).tolist())
    print(f"{len(texts)} held-out looped texts (of {len(cand)} with onset <= {MAX_ONSET}); "
          f"tokens per variant: {sum(len(rows[i]['prompt_ids']) + rows[i]['onset'] + LOOP_SPAN for i in texts):,}")

    tok, model = wsp.load("instruct", dtype=torch.bfloat16)
    model.model.visual = None
    model.eval().to("cuda")
    lm, head = model.model.language_model, model.lm_head
    W0 = {l: wsp.layers(model)[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}       # bf16, on GPU
    W0d = {l: W0[l].double().cpu().numpy() for l in LAYERS}
    fit = fit_discovery()
    from safetensors import safe_open
    fe = safe_open(glob.glob(wsp.snapshot("edited") + "/*.safetensors")[0], "pt")
    Wpub = {l: fe.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight") for l in LAYERS}

    def rank(l, name):
        f, w = fit[l], W0d[l]
        s = {"act": np.abs(f["d_act"]), "between": np.abs(f["d_between"]),
             "write": f["d_act"] * (f["uhat"] @ w), "weights": np.abs(f["uhat"] @ w)}[name]
        return np.argsort(-s, kind="stable")

    def directional(l, cols, u, alpha):
        w = W0d[l].copy()
        w[:, cols] -= alpha * np.outer(u, u @ w[:, cols])
        return torch.from_numpy(w).to(torch.bfloat16)

    def zeroed(l, cols):
        w = W0d[l].copy()
        w[:, cols] = 0
        return torch.from_numpy(w).to(torch.bfloat16)

    allc = np.arange(3584)
    pub = {l: truth[f"L{l}_cols"] for l in LAYERS}
    upub = {l: truth[f"L{l}_u"] for l in LAYERS}
    rs = np.random.default_rng(1)
    rand_cols = [{l: rs.choice(3584, 64, replace=False) for l in LAYERS} for _ in range(3)]
    rand_dir = {}
    for l in LAYERS:
        v = rs.standard_normal(1024)
        v -= (v @ fit[l]["uhat"]) * fit[l]["uhat"]
        rand_dir[l] = v / np.linalg.norm(v)

    V = {}   # name -> (description, {layer: weight tensor}, columns per layer)
    V["none"] = ("unedited model", {l: W0[l].cpu() for l in LAYERS}, 0)
    V["published"] = ("the published edit (checkpoint weights)", Wpub, 64)
    V["pubcols_owndir_a6"] = ("published 64 columns, our direction", {l: directional(l, pub[l], fit[l]["uhat"], ALPHA) for l in LAYERS}, 64)
    V["pubcols_randdir_a6"] = ("published 64 columns, random direction orthogonal to ours", {l: directional(l, pub[l], rand_dir[l], ALPHA) for l in LAYERS}, 64)
    for name in ("act", "write", "weights", "between"):
        V[f"{name}_top64_a6"] = (f"top 64 by {name} ranking, our direction", {l: directional(l, rank(l, name)[:64], fit[l]["uhat"], ALPHA) for l in LAYERS}, 64)
    for j, rc in enumerate(rand_cols):
        V[f"random64_{j}_a6"] = ("64 random columns, our direction", {l: directional(l, rc[l], fit[l]["uhat"], ALPHA) for l in LAYERS}, 64)
    for a in (1.0, 2.0, 4.0, 6.0):
        V[f"full_owndir_a{a:g}"] = ("all 3,584 columns, our direction", {l: directional(l, allc, fit[l]["uhat"], a) for l in LAYERS}, 3584)
    V["full_pubdir_a2"] = ("all 3,584 columns, published direction", {l: directional(l, allc, upub[l], 2.0) for l in LAYERS}, 3584)
    for name in ("act", "write"):
        V[f"{name}_top64_zero"] = (f"top 64 by {name} ranking, columns zeroed", {l: zeroed(l, rank(l, name)[:64]) for l in LAYERS}, 64)
    V["pubcols_zero"] = ("published 64 columns zeroed", {l: zeroed(l, pub[l]) for l in LAYERS}, 64)
    for k in (8, 16, 32, 128, 256):
        V[f"write_top{k}_a6"] = (f"top {k} by write ranking, our direction", {l: directional(l, rank(l, 'write')[:k], fit[l]["uhat"], ALPHA) for l in LAYERS}, k)
    for k in (16, 32, 128, 256):
        V[f"act_top{k}_a6"] = (f"top {k} by act ranking, our direction", {l: directional(l, rank(l, 'act')[:k], fit[l]["uhat"], ALPHA) for l in LAYERS}, k)

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line)
            done.add((d["variant"], d["text"]))

    def logps(ids):
        """log-prob of each next token: lp[t] = log p(ids[t+1] | ids[:t+1]), length len(ids) - 1."""
        pkv, s, out = None, 0, []
        while s < len(ids):
            x = ids[s:s + CHUNK].unsqueeze(0).to("cuda")
            with torch.no_grad():
                o = lm(input_ids=x, past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                h = o.last_hidden_state[0]
                tgt = ids[s + 1:s + 1 + x.shape[1]].to("cuda")
                for a in range(0, len(tgt), 256):
                    lg = head(h[a:a + 256][:len(tgt) - a]).float()
                    out.append(torch.log_softmax(lg, -1).gather(1, tgt[a:a + 256].unsqueeze(1))[:, 0].cpu())
            del o
            s += x.shape[1]
        del pkv
        torch.cuda.empty_cache()
        return torch.cat(out)

    base = {}
    order = ["none"] + [v for v in V if v != "none"]
    if SMOKE:
        order = order[:4]
    for vi, name in enumerate(order):
        desc, weights, k = V[name]
        for l in LAYERS:
            wsp.layers(model)[l].mlp.down_proj.weight.data.copy_(weights[l].to("cuda"))
        changed = int(sum((weights[l].to("cuda") != W0[l]).sum() for l in LAYERS))
        for i in texts:
            if name != "none" and (name, i) in done:
                continue
            row = rows[i]
            npr, on = len(row["prompt_ids"]), row["onset"]
            ids = torch.cat([row["prompt_ids"], row["gen_ids"][:on + LOOP_SPAN]]).long()
            lp = logps(ids)                                   # lp[t] predicts ids[t+1]
            cls = np.concatenate([np.zeros(npr, np.int8), classes(row)[:on + LOOP_SPAN]])[1:]   # class of target
            if name == "none":
                base[i] = lp
                continue
            dlt = (lp - base[i]).numpy()
            rec = {"variant": name, "desc": desc, "text": i, "id": row["id"], "k_per_layer": k,
                   "weights_changed": changed,
                   "loop_delta": float(dlt[cls == 3].mean()), "prod_delta": float(dlt[cls == 1].mean()),
                   "loop_base_logp": float(base[i].numpy()[cls == 3].mean()),
                   "prod_base_logp": float(base[i].numpy()[cls == 1].mean()),
                   "n_loop": int((cls == 3).sum()), "n_prod": int((cls == 1).sum())}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"{vi + 1}/{len(order)} {name} done", flush=True)
    print("PROXY_DONE", flush=True)


if __name__ == "__main__":
    main()
