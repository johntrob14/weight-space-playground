"""Step 2 analysis: rank L15 / L19 MLP neurons of the unedited model by activation difference between loop and
non-loop conditions; score the rankings against the 64 edited neurons per layer.

Contrasts (positive class first):
  within     loop tokens vs productive tokens, looped unedited-model generations (token-pooled)
  within_pt  the same, as a paired per-generation difference (t statistic over generations)
  between    looped vs non-looped unedited-model generations, all non-gray generated tokens (the contrast as asked)
  between_p  looped vs non-looped unedited-model generations, productive tokens only (before any loop)
  vs_edited  loop tokens (unedited-model text) vs productive tokens of the edited model's text on the same problems
Scores per neuron: |difference of means|, |Cohen's d| (token-pooled sd), |AUROC - 0.5| (within, vs_edited only),
|paired t|. Rankings are fit on the discovery half of the problems and on all problems.

    python scripts/analyze_step2.py     -> results/step2.json, results/step2_table.csv, results/step2_scores.npz
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS, KS = (15, 19), (16, 32, 64, 128, 256)
rng = np.random.default_rng(0)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results/acts/stats/*.pt"))):
        rows.append(torch.load(f))
    return rows


def pooled(rows, key, cls):
    """Token-pooled n, mean, variance for stat `key` (e.g. act15) over the given (row, class) pairs."""
    n, s, q = 0, 0, 0
    for r in rows:
        for c in cls:
            a = r["stats"][f"{key}_c{c}"]
            n += a[0]
            s = s + a[1].double().numpy()
            q = q + a[2].double().numpy()
    if n == 0:
        return 0, None, None
    m = s / n
    return n, m, np.maximum(q / n - m * m, 0)


def per_text_mean(r, key, c):
    a = r["stats"][f"{key}_c{c}"]
    return a[1].double().numpy() / a[0] if a[0] >= 64 else None


def contrast(pos_rows, pos_cls, neg_rows, neg_cls, key):
    n1, m1, v1 = pooled(pos_rows, key, pos_cls)
    n0, m0, v0 = pooled(neg_rows, key, neg_cls)
    if n1 == 0 or n0 == 0:          # contrast not available yet (partial data)
        z = np.zeros(3584 if key.startswith("act") else 1024)
        return {"n_pos": n1, "n_neg": n0, "diff": z, "d": z, "mean_pos": z, "mean_neg": z}
    diff = m1 - m0
    sd = np.sqrt((n1 * v1 + n0 * v0) / (n1 + n0))
    return {"n_pos": n1, "n_neg": n0, "diff": diff, "d": diff / np.maximum(sd, 1e-8), "mean_pos": m1, "mean_neg": m0}


def paired(rows, key):
    d = []
    for r in rows:
        a, b = per_text_mean(r, key, 3), per_text_mean(r, key, 1)
        if a is not None and b is not None:
            d.append(a - b)
    d = np.stack(d)
    return {"n_texts": len(d), "diff": d.mean(0), "t": d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)) + 1e-12)}


def auroc_from_hist(hp, hn):
    """AUROC per neuron from binned counts (positives hp, negatives hn): [neurons, bins]."""
    hp, hn = hp.astype(np.float64), hn.astype(np.float64)
    cn = np.cumsum(hn, 1) - hn                      # negatives strictly below each bin
    u = (hp * (cn + 0.5 * hn)).sum(1)
    return u / np.maximum(hp.sum(1) * hn.sum(1), 1)


def score_ranking(score, truth_cols, n=3584):
    order = np.argsort(-score, kind="stable")
    is_t = np.zeros(n, bool)
    is_t[truth_cols] = True
    hits = is_t[order]
    out = {f"P@{k}": float(hits[:k].mean()) for k in KS}
    out["recall@64"] = float(hits[:64].sum() / len(truth_cols))
    out["recall@256"] = float(hits[:256].sum() / len(truth_cols))
    prec = np.cumsum(hits) / np.arange(1, n + 1)
    out["AP"] = float((prec * hits).sum() / hits.sum())
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    out["median_rank_of_edited"] = float(np.median(ranks[truth_cols]))
    # AUROC of the score for edited vs other neurons
    out["AUROC"] = float(1 - (ranks[truth_cols].mean() - (len(truth_cols) - 1) / 2) / (n - len(truth_cols)))
    return out


def main():
    rows = load()
    truth = np.load(os.path.join(ROOT, "results/edit_truth.npz"))
    st = torch.load(os.path.join(ROOT, "results/acts/state.pt"))
    hist = st["hist"].numpy()        # [split, model, class(0 productive, 1 loop), layer, neuron, bin]
    un = [r for r in rows if r["model"] == "unedited"]
    ed = [r for r in rows if r["model"] == "edited"]
    looped = [r for r in un if r["onset"] is not None]
    nonloop = [r for r in un if r["onset"] is None]
    ed_by_id = {r["id"]: r for r in ed}
    info = {"n_texts": {"unedited": len(un), "edited": len(ed), "unedited_looped": len(looped),
                        "unedited_nonlooped": len(nonloop)},
            "tokens": {"unedited_loop": int(sum(r["stats"]["act15_c3"][0] for r in un)),
                       "unedited_productive_looped_texts": int(sum(r["stats"]["act15_c1"][0] for r in looped)),
                       "unedited_productive_nonlooped_texts": int(sum(r["stats"]["act15_c1"][0] for r in nonloop)),
                       "edited_productive": int(sum(r["stats"]["act15_c1"][0] for r in ed)),
                       "all_processed": int(sum(r["n_tokens"] for r in rows))}}
    print(json.dumps(info, indent=1))
    table, scores, dirs = [], {}, {}
    for fit in ("discovery", "all"):
        sel = (lambda rs: [r for r in rs if r["split"] == 0]) if fit == "discovery" else (lambda rs: rs)
        L, N, E = sel(looped), sel(nonloop), sel(ed)
        for li, l in enumerate(LAYERS):
            cols, u, proj0 = truth[f"L{l}_cols"], truth[f"L{l}_u"], truth[f"L{l}_proj0"]
            key = f"act{l}"
            C = {"within": contrast(L, (3,), L, (1,), key),
                 "between": contrast(L, (1, 3), N, (1,), key),
                 "between_p": contrast(L, (1,), N, (1,), key),
                 "vs_edited": contrast(L, (3,), [ed_by_id[r["id"]] for r in L if r["id"] in ed_by_id], (1,), key)}
            P = paired(L, key)
            S = {}
            for name, c in C.items():
                S[f"{name}|absdiff"] = np.abs(c["diff"])
                S[f"{name}|absd"] = np.abs(c["d"])
            S["within_pt|abst"] = np.abs(P["t"])
            sp = [0] if fit == "discovery" else [0, 1]
            hp = hist[sp, 0, 1, li].sum(0)
            S["within|auroc"] = np.abs(auroc_from_hist(hp, hist[sp, 0, 0, li].sum(0)) - 0.5)
            S["vs_edited|auroc"] = np.abs(auroc_from_hist(hp, hist[sp, 1, 0, li].sum(0)) - 0.5)
            # our own residual direction (difference of means after layer l, loop - productive), unit norm
            rc = contrast(L, (3,), L, (1,), f"res{l}")
            uhat = rc["diff"] / np.linalg.norm(rc["diff"])
            mc = contrast(L, (3,), L, (1,), f"mlp{l}")
            dirs[(fit, l)] = {"cos_res_dom_vs_u": float(abs(uhat @ u)),
                              "cos_mlpout_dom_vs_u": float(abs(mc["diff"] @ u) / np.linalg.norm(mc["diff"])),
                              "sign": float(np.sign(uhat @ u))}
            # write contrast along a direction: (mean act loop - mean act productive)_j * (dir . W0[:, j])
            # proj0 = u_published . W0; for our direction we need W0: loaded lazily below
            W0 = get_W0(l)
            S["write|own_dir_signed"] = C["within"]["diff"] * (uhat @ W0)
            S["write|own_dir_abs"] = np.abs(S["write|own_dir_signed"])
            S["write|published_u_signed"] = C["within"]["diff"] * proj0 * np.sign(uhat @ u)
            S["write|published_u_abs"] = np.abs(S["write|published_u_signed"])
            # baselines
            n_all, m_all, v_all = pooled(sel(un), key, (1, 3))
            S["baseline|activation_variance"] = v_all
            S["baseline|abs_mean_activation"] = np.abs(m_all)
            S["baseline|low_index_first"] = -np.arange(3584.0)
            S["baseline|weights_only_abs_u_dot_W"] = np.abs(proj0)
            S["baseline|weights_only_own_dir"] = np.abs(uhat @ W0)
            S["baseline|down_col_norm"] = np.linalg.norm(W0, axis=0)
            for name, s in S.items():
                table.append({"fit": fit, "layer": l, "method": name, **score_ranking(s, cols)})
                scores[f"{fit}|L{l}|{name}"] = s
            rnd = [score_ranking(rng.random(3584), cols) for _ in range(200)]
            table.append({"fit": fit, "layer": l, "method": "baseline|random (mean of 200)",
                          **{k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}})
            if fit == "all":
                scores[f"L{l}|mean_loop"] = C["within"]["mean_pos"]
                scores[f"L{l}|mean_productive"] = C["within"]["mean_neg"]
                scores[f"L{l}|uhat"] = uhat
    df = pd.DataFrame(table)
    df.to_csv(os.path.join(ROOT, "results/step2_table.csv"), index=False)
    np.savez(os.path.join(ROOT, "results/step2_scores.npz"), **scores)
    # held-out stability: Spearman between discovery and held-out |d| (within)
    stab = {}
    for l in LAYERS:
        a = contrast([r for r in looped if r["split"] == 0], (3,), [r for r in looped if r["split"] == 0], (1,),
                     f"act{l}")["d"]
        b = contrast([r for r in looped if r["split"] == 1], (3,), [r for r in looped if r["split"] == 1], (1,),
                     f"act{l}")["d"]
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        stab[l] = float(np.corrcoef(ra, rb)[0, 1])
    info["spearman_d_discovery_vs_heldout"] = stab
    info["directions"] = {f"{k[0]}|L{k[1]}": v for k, v in dirs.items()}
    json.dump({"info": info, "table": table}, open(os.path.join(ROOT, "results/step2.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    for fit in ("discovery", "all"):
        for l in LAYERS:
            d = df[(df.fit == fit) & (df.layer == l)].drop(columns=["fit", "layer"]).sort_values("AP", ascending=False)
            print(f"\n=== rankings fit on {fit} problems, layer {l} (64 edited of 3584; chance precision 0.018) ===")
            print(d.round(3).to_string(index=False))
    print("\nstability (Spearman of Cohen's d, discovery vs held-out halves):", stab)
    print("directions:", json.dumps(info["directions"], indent=1))


_W = {}


def get_W0(l):
    if l not in _W:
        from safetensors import safe_open
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import wsp
        f = safe_open(glob.glob(wsp.snapshot("instruct") + "/*.safetensors")[0], "pt")
        _W[l] = f.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight").double().numpy()
    return _W[l]


if __name__ == "__main__":
    main()
