"""Step 1: verify the published weight edit.

Checks, all on CPU from the safetensors files (bit-exact bfloat16 comparisons):
  A. Which tensors differ between Qwen/Qwen3.5-0.8B and the edited HF checkpoint.
  B. sha256 of base / edited tensors against the published manifest and model card.
  C. Rebuilding the edit from edit.npz gives the HF checkpoint's tensors.
  D. Which entries / columns / rows changed; do the columns equal the editor's selected_neurons.
  E. Structure of the delta: rank, direction u, strength alpha in W = W0 - alpha * u (mask * (u @ W0)).
  F. How the 64 columns relate to a weights-only ranking by |u @ W0[:, j]|.

    python scripts/step1_diff.py > results/step1_diff.log   (writes results/step1_diff.json, results/edit_truth.npz)
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

HUB = os.path.expanduser("~/.cache/huggingface/hub")
REV = {
    "instruct": ("models--Qwen--Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "edited": ("models--RikiyaT--Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b"),
    "base": ("models--Qwen--Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIT = os.path.join(ROOT, "ext/lm_understand-logs/exp8_math/runs/p50c")


def st_path(key):
    d, rev = REV[key]
    (p,) = glob.glob(os.path.join(HUB, d, "snapshots", rev, "*.safetensors"))
    return p


def bits(t):
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bf16_to_f64(b):
    return torch.from_numpy(b.view(np.int16).copy()).view(torch.bfloat16).double().numpy()


def main():
    out = {"revisions": {k: v[1] for k, v in REV.items()}}
    fi, fe = safe_open(st_path("instruct"), "pt"), safe_open(st_path("edited"), "pt")
    ki, ke = set(fi.keys()), set(fe.keys())
    print(f"A. tensors: instruct {len(ki)}, edited {len(ke)}, key sets equal: {ki == ke}")
    differing, n_param = [], 0
    for k in sorted(ki & ke):
        a, b = fi.get_tensor(k), fe.get_tensor(k)
        n_param += a.numel()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(
                a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                b.view(torch.int16) if b.dtype == torch.bfloat16 else b):
            differing.append(k)
    print(f"   total entries compared: {n_param:,}")
    print(f"   differing tensors ({len(differing)}): {differing}")
    out["n_entries_total"] = n_param
    out["differing_tensors"] = differing
    out["keys_only_in_one"] = sorted(ki ^ ke)

    man = json.load(open(os.path.join(EDIT, "edit/manifest.json")))
    frozen = json.load(open(os.path.join(EDIT, "work_notes/frozen_weights_manifest.json")))
    z = np.load(os.path.join(EDIT, "edit/edit.npz"), allow_pickle=False)
    truth, out["layers"] = {}, {}
    for i, t in enumerate(man["tensors"]):
        name = t["name"]
        layer = int(name.split(".layers.")[1].split(".")[0])
        b0, b1 = bits(fi.get_tensor(name)), bits(fe.get_tensor(name))
        r = {"name": name, "shape": list(b0.shape)}
        r["base_sha_ok"] = sha(b0) == t["base_sha256"]
        r["edited_sha_ok"] = sha(b1) == t["edited_sha256"]
        rebuilt = b0.copy().reshape(-1)
        rebuilt[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        r["npz_rebuild_equals_hf_edited"] = bool(np.array_equal(rebuilt.reshape(b0.shape), b1))
        mask = b0 != b1
        cols, rows = np.flatnonzero(mask.any(0)), np.flatnonzero(mask.any(1))
        r["changed_entries"] = int(mask.sum())
        r["changed_columns"] = int(len(cols))
        r["changed_rows"] = int(len(rows))
        r["unchanged_rows"] = [int(x) for x in np.flatnonzero(~mask.any(1))]
        r["changed_entries_per_column_min_max"] = [int(mask[:, cols].sum(0).min()), int(mask[:, cols].sum(0).max())]
        sel = frozen[str(layer)]["selected_neurons"]
        r["columns_equal_selected_neurons"] = cols.tolist() == sorted(sel)
        r["columns"] = cols.tolist()

        w0, w1 = bf16_to_f64(b0), bf16_to_f64(b1)
        d = w1 - w0
        r["relative_frobenius_change"] = float(np.linalg.norm(d) / np.linalg.norm(w0))
        u_, s, vt = np.linalg.svd(d[:, cols], full_matrices=False)
        r["delta_singular_values_top4"] = [float(x) for x in s[:4]]
        r["delta_rank1_energy_fraction"] = float(s[0] ** 2 / (s ** 2).sum())
        u = u_[:, 0]
        proj0 = u @ w0                       # how much each neuron writes along u (all 3584 neurons)
        # alpha: least squares of delta = -alpha * outer(u, proj0[cols])
        pred = np.outer(u, proj0[cols])
        alpha = -float((d[:, cols] * pred).sum() / (pred * pred).sum())
        r["alpha_fit"] = alpha
        r["formula_residual_fraction"] = float(np.linalg.norm(d[:, cols] + alpha * pred) / np.linalg.norm(d[:, cols]))
        r["u_norm"] = float(np.linalg.norm(u))
        r["u_abs_max_component"] = float(np.abs(u).max())
        # after the edit, the selected neurons write (1 - alpha) * proj0 along u: sign flips if alpha > 1
        proj1 = u @ w1
        r["proj_ratio_after_over_before_selected_median"] = float(np.median(proj1[cols] / proj0[cols]))
        # weights-only ranking: |u @ W0[:, j]|, and relative to column norm
        rank_abs = np.argsort(-np.abs(proj0))
        r["weights_only_top64_overlap"] = int(len(set(rank_abs[:64].tolist()) & set(cols.tolist())))
        r["selected_rank_by_abs_proj_median_max"] = [
            float(np.median(np.argsort(np.argsort(-np.abs(proj0)))[cols])),
            int(np.argsort(np.argsort(-np.abs(proj0)))[cols].max())]
        r["selected_proj_sign_counts_pos_neg"] = [int((proj0[cols] > 0).sum()), int((proj0[cols] < 0).sum())]
        r["selected_share_of_abs_proj_mass"] = float(np.abs(proj0[cols]).sum() / np.abs(proj0).sum())
        r["selected_share_of_delta_if_all_columns"] = float(
            (proj0[cols] ** 2).sum() / (proj0 ** 2).sum())
        r["selected_index_quantiles_min_med_max"] = [int(cols.min()), float(np.median(cols)), int(cols.max())]
        r["n_selected_below_index_512"] = int((cols < 512).sum())
        out["layers"][str(layer)] = r
        truth[f"L{layer}_cols"] = cols
        truth[f"L{layer}_u"] = u
        truth[f"L{layer}_proj0"] = proj0
        print(f"\n{name}")
        for k, v in r.items():
            if k != "columns":
                print(f"   {k}: {v}")
        print(f"   columns: {cols.tolist()}")
    out["changed_entries_total"] = sum(r["changed_entries"] for r in out["layers"].values())
    out["changed_values_total_incl_rounding_noops"] = sum(
        r["changed_columns"] * r["shape"][0] for r in out["layers"].values())
    print(f"\nchanged entries total (bits differ): {out['changed_entries_total']:,}; "
          f"columns x rows: {out['changed_values_total_incl_rounding_noops']:,}; "
          f"fraction of all entries: {out['changed_values_total_incl_rounding_noops'] / n_param:.6%}")

    # cosine between the two layers' directions
    c = float(truth["L15_u"] @ truth["L19_u"])
    out["cos_u15_u19"] = c
    print(f"cos(u_L15, u_L19) = {c:.4f} (SVD sign is arbitrary)")

    # Base (pretrained) vs instruct, same two tensors, for reference
    if os.path.isdir(os.path.join(HUB, REV["base"][0])):
        fb = safe_open(st_path("base"), "pt")
        kb = set(fb.keys())
        out["base_vs_instruct_keys_equal"] = kb == ki
        nd, tot = 0, 0
        for k in sorted(kb & ki):
            a, b = fb.get_tensor(k), fi.get_tensor(k)
            tot += 1
            if a.shape != b.shape or not torch.equal(a.float(), b.float()):
                nd += 1
        out["base_vs_instruct_differing_tensors"] = [nd, tot]
        print(f"\nBase vs instruct: key sets equal {kb == ki}; {nd}/{tot} shared tensors differ")
        for layer in (15, 19):
            name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            a, b = fb.get_tensor(name).double().numpy(), fi.get_tensor(name).double().numpy()
            rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            out[f"base_vs_instruct_L{layer}_down_proj_rel_change"] = rel
            print(f"   L{layer} down_proj relative difference Base vs instruct: {rel:.4f}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results/step1_diff.json"), "w"), indent=1)
    np.savez(os.path.join(ROOT, "results/edit_truth.npz"), **truth)


if __name__ == "__main__":
    sys.exit(main())
