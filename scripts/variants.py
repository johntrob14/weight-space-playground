"""Named interventions on the unedited model, shared by the proxy and generation evaluations.

Weight edits (mlp.down_proj of layers 15 and 19 together; rankings and u_hat fit on the discovery half only):
  none | published
  {act|write|weights|between}_top{k}_a{alpha}   top-k neurons of a step 2 ranking, W - alpha u_hat (u_hat^T W)
  pubcols_owndir_a{alpha} | full_owndir_a{alpha} | full_pubdir_a{alpha} | random64_{j}_a{alpha}
  {act|write}_top{k}_zero | pubcols_zero
Residual steering (hook on the output of one layer, every position):
  steer_{name}_L{layer}_c{coef}     adds coef * v, v = results/directions/{name}_L{layer}.npy (unit norm)
Rank-1 weight edit along a stored direction, all columns of that layer's down_proj:
  proj_{name}_a{alpha}              uses results/directions/{name}_L15.npy and _L19.npy
"""
import glob
import os
import re

import numpy as np
import torch

import wsp
from analyze_step2 import contrast, load as load_stats

LAYERS = (15, 19)


def fit_discovery():
    rows = [r for r in load_stats() if r["split"] == 0]
    un = [r for r in rows if r["model"] == "unedited"]
    L, N = [r for r in un if r["onset"] is not None], [r for r in un if r["onset"] is None]
    fit = {}
    for l in LAYERS:
        rc = contrast(L, (3,), L, (1,), f"res{l}")
        fit[l] = {"uhat": rc["diff"] / np.linalg.norm(rc["diff"]),
                  "d_act": contrast(L, (3,), L, (1,), f"act{l}")["diff"],
                  "d_between": contrast(L, (1, 3), N, (1,), f"act{l}")["diff"]}
    return fit


class Context:
    def __init__(self, model):
        self.model = model
        self.W0 = {l: wsp.layers(model)[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}
        self.W0d = {l: self.W0[l].double().cpu().numpy() for l in LAYERS}
        self.fit = fit_discovery()
        self.truth = np.load(os.path.join(wsp.ROOT, "results/edit_truth.npz"))
        self.hooks = []
        rs = np.random.default_rng(1)          # same draws as step4_proxy.py
        self.rand_cols = [{l: rs.choice(3584, 64, replace=False) for l in LAYERS} for _ in range(3)]

    def rank(self, l, name):
        f, w = self.fit[l], self.W0d[l]
        s = {"act": np.abs(f["d_act"]), "between": np.abs(f["d_between"]),
             "write": f["d_act"] * (f["uhat"] @ w), "weights": np.abs(f["uhat"] @ w)}[name]
        return np.argsort(-s, kind="stable")

    def directional(self, l, cols, u, alpha):
        w = self.W0d[l].copy()
        w[:, cols] -= alpha * np.outer(u, u @ w[:, cols])
        return torch.from_numpy(w).to(torch.bfloat16)

    def zeroed(self, l, cols):
        w = self.W0d[l].copy()
        w[:, cols] = 0
        return torch.from_numpy(w).to(torch.bfloat16)

    def direction(self, name, l):
        v = np.load(os.path.join(wsp.ROOT, f"results/directions/{name}_L{l}.npy")).astype(np.float64)
        return v / np.linalg.norm(v)

    def weights_for(self, name):
        allc = np.arange(3584)
        pub = {l: self.truth[f"L{l}_cols"] for l in LAYERS}
        uh = {l: self.fit[l]["uhat"] for l in LAYERS}
        if name == "none" or name.startswith("steer_"):
            return {l: self.W0[l] for l in LAYERS}
        if name == "published":
            from safetensors import safe_open
            fe = safe_open(glob.glob(wsp.snapshot("edited") + "/*.safetensors")[0], "pt")
            return {l: fe.get_tensor(f"model.language_model.layers.{l}.mlp.down_proj.weight") for l in LAYERS}
        m = re.fullmatch(r"(act|write|weights|between)_top(\d+)_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, self.rank(l, m[1])[:int(m[2])], uh[l], float(m[3])) for l in LAYERS}
        m = re.fullmatch(r"(act|write)_top(\d+)_zero", name)
        if m:
            return {l: self.zeroed(l, self.rank(l, m[1])[:int(m[2])]) for l in LAYERS}
        if name == "pubcols_zero":
            return {l: self.zeroed(l, pub[l]) for l in LAYERS}
        m = re.fullmatch(r"pubcols_owndir_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, pub[l], uh[l], float(m[1])) for l in LAYERS}
        m = re.fullmatch(r"full_owndir_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, allc, uh[l], float(m[1])) for l in LAYERS}
        m = re.fullmatch(r"full_pubdir_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, allc, self.truth[f"L{l}_u"], float(m[1])) for l in LAYERS}
        m = re.fullmatch(r"random64_(\d)_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, self.rand_cols[int(m[1])][l], uh[l], float(m[2])) for l in LAYERS}
        m = re.fullmatch(r"proj_(.+)_a([\d.]+)", name)
        if m:
            return {l: self.directional(l, allc, self.direction(m[1], l), float(m[2])) for l in LAYERS}
        raise ValueError(name)

    def apply(self, name):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        w = self.weights_for(name)
        for l in LAYERS:
            wsp.layers(self.model)[l].mlp.down_proj.weight.data.copy_(w[l].to("cuda"))
        m = re.fullmatch(r"steer_(.+)_L(\d+)_c(-?[\d.]+)", name)
        if m:
            l, c = int(m[2]), float(m[3])
            v = torch.from_numpy(self.direction(m[1], l)).to("cuda", torch.bfloat16) * c

            def hook(mod, args, out):
                if isinstance(out, tuple):
                    return (out[0] + v,) + tuple(out[1:])
                return out + v
            self.hooks.append(wsp.layers(self.model)[l].register_forward_hook(hook))
        return int(sum((w[l].to("cuda") != self.W0[l]).sum() for l in LAYERS))
