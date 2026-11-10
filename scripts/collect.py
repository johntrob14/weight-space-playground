"""Step 2 data collection: teacher-force the published generations through the UNEDITED model (bf16) and record
MLP-neuron statistics in layers 15 and 19 for productive vs loop tokens.

Token classes (generated tokens only; the prompt is class 0):
  3 loop        t >= strict loop onset (see prep_data.py)
  1 productive  local repeat rate (centered 256-token window) < 0.2, and before the onset if the text loops
  2 gray        everything else (excluded from contrasts)
Tokens processed per text: looped texts up to onset + 2048; unedited non-looped texts in full; edited-model texts up
to 8192 generated tokens.

Outputs in results/acts/:
  stats/<model>_<idx>.pt   per text, per class in (1, 3): n, sum and sum of squares of neuron activations (L15, L19),
                           of the residual stream after layers 15 and 19, and of the MLP outputs of those layers
  state.pt                 pooled histograms per split/model/class/layer/neuron (128 bins on [-8, 8]) for AUROC,
                           running top-20 / bottom-20 activations per neuron, a 1% random token sample of raw
                           activations and residuals, and the list of finished texts (resumable)

    python -u scripts/collect.py [--test] [--limit N]
"""
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402
from prep_data import local_rate  # noqa: E402

LAYERS, CHUNK, TAIL, EDITED_CAP, NB, LO, HI, TOPK, SAMPLE_P = (15, 19), 1024, 2048, 8192, 128, -8.0, 8.0, 20, 0.01
OUT = os.path.join(wsp.ROOT, "results/acts")


def split_of(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2      # 0 discovery, 1 held-out


def classes(row):
    r = row["repeat"].numpy()
    lr = local_rate(r)
    c = np.where(lr < 0.2, 1, 2).astype(np.int8)
    o = row["onset"]
    if o is not None:
        c[o:] = 3
    return c


def cap_of(row):
    n = row["n_gen"]
    if row["model"] == "edited":
        return min(n, EDITED_CAP)
    return n if row["onset"] is None else min(n, row["onset"] + TAIL)


class Taps:
    def __init__(self, model):
        self.x, self.h = {}, []
        for l in LAYERS:
            layer = wsp.layers(model)[l]
            self.h.append(layer.mlp.down_proj.register_forward_pre_hook(self._pre(("act", l))))
            self.h.append(layer.mlp.register_forward_hook(self._out(("mlp", l))))
            self.h.append(layer.register_forward_hook(self._out(("res", l))))

    def _pre(self, key):
        def f(mod, args):
            self.x[key] = args[0].detach()[0]
        return f

    def _out(self, key):
        def f(mod, args, out):
            self.x[key] = (out[0] if isinstance(out, tuple) else out).detach()[0]
        return f


def run_text(lm, taps, ids, on_chunk):
    """Chunked forward with the cache carried; on_chunk(start, end, taps.x) after each chunk."""
    pkv, s = None, 0
    while s < len(ids):
        step = CHUNK if s < 6144 else 512 if s < 14336 else 256      # attention memory grows with the past length
        x = ids[s:s + step].unsqueeze(0).to("cuda")
        with torch.no_grad():
            o = lm(input_ids=x, past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        del o
        on_chunk(s, s + x.shape[1], taps.x)
        s += x.shape[1]
    del pkv
    torch.cuda.empty_cache()


def test(lm, taps, rows):
    row = next(r for r in rows if r["model"] == "unedited" and r["onset"] is not None)
    ids = torch.cat([row["prompt_ids"], row["gen_ids"][:4096 - len(row["prompt_ids"])]]).long()
    with torch.no_grad():
        lm(input_ids=ids.unsqueeze(0).to("cuda"))
    full = {k: v.float().cpu() for k, v in taps.x.items()}
    parts = {k: [] for k in full}
    run_text(lm, taps, ids, lambda s, e, x: [parts[k].append(v.float().cpu()) for k, v in x.items()])
    for k in full:
        ch = torch.cat(parts[k])
        print(f"   chunked vs full, {k}: shape {tuple(ch.shape)}, relative error "
              f"{float((ch - full[k]).norm() / full[k].norm()):.4f}, max abs {float((ch - full[k]).abs().max()):.3f} "
              f"(scale {float(full[k].abs().max()):.1f})")
    del full, parts
    for n in (16384, 32768):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run_text(lm, taps, torch.cat([row["prompt_ids"], row["gen_ids"][:n]]).long()[:n], lambda s, e, x: None)
        torch.cuda.synchronize()
        print(f"   {n}-token text, chunked: {n / (time.time() - t0):.0f} tok/s, peak "
              f"{torch.cuda.max_memory_allocated() // 2 ** 20} MiB", flush=True)


def main():
    rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
    tok, model = wsp.load("instruct", dtype=torch.bfloat16)
    model.model.visual = None
    model.eval().to("cuda")
    lm = model.model.language_model
    taps = Taps(model)
    if "--test" in sys.argv:
        return test(lm, taps, rows)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    os.makedirs(os.path.join(OUT, "stats"), exist_ok=True)
    sp = os.path.join(OUT, "state.pt")
    if os.path.exists(sp):
        st = torch.load(sp)
    else:
        st = {"done": [],
              # hist[split, model, class(0=productive,1=loop), layer, neuron, bin]
              "hist": torch.zeros(2, 2, 2, 2, 3584, NB, dtype=torch.int32),
              "top": torch.full((2, TOPK, 3584), -1e9), "top_at": torch.zeros(2, TOPK, 3584, 2, dtype=torch.int32),
              "bot": torch.full((2, TOPK, 3584), 1e9), "bot_at": torch.zeros(2, TOPK, 3584, 2, dtype=torch.int32),
              "sample": []}
    hist = st["hist"].to("cuda")
    top, top_at, bot, bot_at = (st[k].to("cuda") for k in ("top", "top_at", "bot", "bot_at"))
    gen = torch.Generator().manual_seed(0)
    order = [i for i, r in enumerate(rows) if r["model"] == "unedited"] + \
            [i for i, r in enumerate(rows) if r["model"] == "edited"]
    order = [i for i in order if i not in set(st["done"])]
    if limit:
        order = order[:limit]
    edges = torch.linspace(LO, HI, NB + 1, device="cuda")[1:-1]
    t_start, n_tok = time.time(), 0
    for count, i in enumerate(order):
        row = rows[i]
        m = 0 if row["model"] == "unedited" else 1
        cap = cap_of(row)
        npr = len(row["prompt_ids"])
        ids = torch.cat([row["prompt_ids"], row["gen_ids"][:cap]]).long()
        cls = torch.cat([torch.zeros(npr, dtype=torch.int8), torch.from_numpy(classes(row)[:cap])]).to("cuda")
        split = split_of(row["id"])
        acc = {(k, l, c): [0, torch.zeros(3584 if k == "act" else 1024, device="cuda", dtype=torch.float64),
                           torch.zeros(3584 if k == "act" else 1024, device="cuda", dtype=torch.float64)]
               for k in ("act", "mlp", "res") for l in LAYERS for c in (1, 3)}
        samp = []

        def on_chunk(s, e, x):
            nonlocal top, top_at, bot, bot_at
            cc = cls[s:e]
            pick = (torch.rand(e - s, generator=gen) < SAMPLE_P).to("cuda") & (cc > 0)
            for li, l in enumerate(LAYERS):
                a = x[("act", l)].float()
                for c in (1, 3):
                    msk = cc == c
                    if msk.any():
                        for k in ("act", "mlp", "res"):
                            v = (a if k == "act" else x[(k, l)].float())[msk].double()
                            acc[(k, l, c)][0] += int(msk.sum())
                            acc[(k, l, c)][1] += v.sum(0)
                            acc[(k, l, c)][2] += (v * v).sum(0)
                        b = torch.bucketize(a[msk], edges)                      # tokens x neurons
                        hist[split, m, 0 if c == 1 else 1, li].scatter_add_(
                            1, b.T.contiguous(), torch.ones_like(b.T, dtype=torch.int32))
                # running top / bottom activations per neuron over generated tokens
                g = cc > 0
                if g.any():
                    ag = a[g]
                    pos = torch.arange(s, e, device="cuda")[g]
                    k = min(TOPK, ag.shape[0])
                    for sign, (val, at) in ((1, (top, top_at)), (-1, (bot, bot_at))):
                        v, ix = (sign * ag).topk(k, dim=0)
                        v = sign * v
                        new_at = torch.stack([torch.full_like(ix, i), pos[ix]], -1).int()
                        allv, alla = torch.cat([val[li], v]), torch.cat([at[li], new_at])
                        _, keep = (sign * allv).topk(TOPK, dim=0)
                        val[li] = allv.gather(0, keep)
                        at[li] = alla.gather(0, keep.unsqueeze(-1).expand(-1, -1, 2))
            if pick.any():
                p = torch.arange(s, e, device="cuda")[pick]
                samp.append({"text": i, "pos": p.cpu().int(), "cls": cc[pick].cpu(),
                             **{f"{k}{l}": x[(k, l)][pick].half().cpu() for k in ("act", "res") for l in LAYERS}})

        run_text(lm, taps, ids, on_chunk)
        torch.save({"model": row["model"], "id": row["id"], "idx": i, "split": split, "n_tokens": len(ids),
                    "onset": row["onset"], "cut": row["cut"],
                    "stats": {f"{k}{l}_c{c}": (v[0], v[1].float().cpu(), v[2].float().cpu())
                              for (k, l, c), v in acc.items()}},
                   os.path.join(OUT, "stats", f"{row['model']}_{i}.pt"))
        st["sample"] += samp
        st["done"].append(i)
        n_tok += len(ids)
        if (count + 1) % 20 == 0 or count + 1 == len(order):
            st.update(hist=hist.cpu(), top=top.cpu(), top_at=top_at.cpu(), bot=bot.cpu(), bot_at=bot_at.cpu())
            torch.save(st, sp + ".tmp")
            os.replace(sp + ".tmp", sp)
            el = time.time() - t_start
            print(f"{count + 1}/{len(order)} texts, {n_tok:,} tokens, {n_tok / el:.0f} tok/s, "
                  f"{el / 60:.1f} min elapsed, sample rows {sum(len(s['pos']) for s in st['sample'])}", flush=True)
    print("COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
