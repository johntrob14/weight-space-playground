"""Step 3, quantitative first look (no model run): what do the extreme activations of the edited neurons land on?
Uses the running top-20 / bottom-20 activations per neuron over all processed generated tokens (collect.py).
    python scripts/step3_quant.py  -> results/step3_quant.log (stdout), results/step3_quant.json
"""
import collections, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp
from collect import classes, cap_of
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(wsp.snapshot("instruct"))
rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
st = torch.load(os.path.join(wsp.ROOT, "results/acts/state.pt"))
truth = np.load(os.path.join(wsp.ROOT, "results/edit_truth.npz"))
sc = np.load(os.path.join(wsp.ROOT, "results/step2_scores.npz"))
cache = {}
def info(i):
    if i not in cache:
        r = rows[i]; cap = cap_of(r); npr = len(r["prompt_ids"])
        ids = torch.cat([r["prompt_ids"], r["gen_ids"][:cap]]).tolist()
        cls = np.concatenate([np.zeros(npr, np.int8), classes(r)[:cap]])
        cache[i] = (ids, cls)
    return cache[i]
out = {}
rng = np.random.default_rng(0)
tot = collections.Counter()
for i in range(len(rows)):
    ids, cls = info(i)
    for c in (1, 2, 3): tot[c] += int((cls == c).sum())
base = tot[3] / sum(tot.values())
print(f"generated tokens processed: {sum(tot.values()):,}; loop share {base:.3f}")
for li, l in enumerate((15, 19)):
    cols = truth[f"L{l}_cols"]
    res = {}
    for pole, key in (("top", "top_at"), ("bottom", "bot_at")):
        at = st[key][li].numpy()           # [20, 3584, 2]
        val = st["top" if pole == "top" else "bot"][li].numpy()
        loop_frac, ntext, tokshare, toptok = np.zeros(3584), np.zeros(3584), np.zeros(3584), [None] * 3584
        for j in range(3584):
            cl, tk, tx = [], [], set()
            for k in range(20):
                i, p = int(at[k, j, 0]), int(at[k, j, 1])
                ids, cls = info(i)
                cl.append(cls[p]); tk.append(ids[p]); tx.add(i)
            loop_frac[j] = np.mean(np.array(cl) == 3); ntext[j] = len(tx)
            t, n = collections.Counter(tk).most_common(1)[0]
            tokshare[j] = n / 20; toptok[j] = t
        res[pole] = (loop_frac, ntext, tokshare, toptok, val)
        oth = np.setdiff1d(np.arange(3584), cols)
        print(f"L{l} {pole}-20 activations: edited neurons (n=64) vs all other neurons")
        for nm, a in (("share on loop tokens", loop_frac), ("distinct generations", ntext), ("share that is the single most common token", tokshare)):
            print(f"    {nm}: edited mean {a[cols].mean():.3f} median {np.median(a[cols]):.3f} | others mean {a[oth].mean():.3f} median {np.median(a[oth]):.3f}")
        out[f"L{l}_{pole}"] = {"edited_loop_share_mean": float(loop_frac[cols].mean()), "others_loop_share_mean": float(loop_frac[oth].mean()),
                               "edited_single_token_share_mean": float(tokshare[cols].mean()), "others_single_token_share_mean": float(tokshare[oth].mean()),
                               "edited_distinct_texts_mean": float(ntext[cols].mean()), "others_distinct_texts_mean": float(ntext[oth].mean())}
    # magnitude: which pole is larger for edited neurons, and mean activations
    mt, mb = np.abs(res["top"][4][0]), np.abs(res["bottom"][4][0])
    print(f"L{l}: edited neurons whose largest |activation| is negative: {int((mb[cols] > mt[cols]).sum())}/64 (others {float((mb > mt).mean()):.2f})")
    print(f"L{l}: 8 random edited neurons, top-pole context (token at the activation in [[ ]]):")
    for j in rng.choice(cols, 8, replace=False):
        at = st["top_at"][li].numpy()
        ex = []
        for k in (0, 7, 15):
            i, p = int(at[k, j, 0]), int(at[k, j, 1]); ids, cls = info(i)
            ex.append(repr(tok.decode(ids[max(0, p - 12):p]) + "[[" + tok.decode(ids[p:p + 1]) + "]]" + tok.decode(ids[p + 1:p + 4])) + f" (cls {cls[p]})")
        print(f"   N{j}: loop share top {res['top'][0][j]:.2f} bottom {res['bottom'][0][j]:.2f}; top token {tok.decode([res['top'][3][j]])!r} x{res['top'][2][j]:.2f}; max {st['top'][li][0, j]:.1f} min {st['bot'][li][0, j]:.1f}")
        for e in ex: print("        ", e[:170])
json.dump(out, open(os.path.join(wsp.ROOT, "results/step3_quant.json"), "w"), indent=1)
