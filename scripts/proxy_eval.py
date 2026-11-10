"""Cheap proxy for any named intervention (variants.py): change in log-prob per token on loop text and on productive
text, on the same 24 held-out looped generations as step4_proxy.py.

    python -u scripts/proxy_eval.py <variant> ...   -> results/proxy_eval.jsonl (append-only, resumable)
"""
import json, os, sys
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp, variants
from collect import classes, split_of
N_TEXTS, MAX_ONSET, LOOP_SPAN, CHUNK = 24, 8000, 512, 1024
OUT = os.path.join(wsp.ROOT, "results/proxy_eval.jsonl")
rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
cand = [i for i, r in enumerate(rows) if r["model"] == "unedited" and r["onset"] is not None and r["onset"] <= MAX_ONSET and split_of(r["id"]) == 1]
texts = sorted(np.random.default_rng(0).choice(cand, size=N_TEXTS, replace=False).tolist())
tok, model = wsp.load("instruct", dtype=torch.bfloat16); model.model.visual = None; model.eval().to("cuda")
wsp.enable_grouped_attention(model)
lm, head = model.model.language_model, model.lm_head
ctx = variants.Context(model)
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        d = json.loads(line); done.add((d["variant"], d["text"]))
def logps(ids):
    pkv, s, out = None, 0, []
    while s < len(ids):
        x = ids[s:s + CHUNK].unsqueeze(0).to("cuda")
        with torch.no_grad():
            o = lm(input_ids=x, past_key_values=pkv, use_cache=True); pkv = o.past_key_values; h = o.last_hidden_state[0]
            tgt = ids[s + 1:s + 1 + x.shape[1]].to("cuda")
            for a in range(0, len(tgt), 256):
                lg = head(h[a:a + 256][:len(tgt) - a]).float()
                out.append(torch.log_softmax(lg, -1).gather(1, tgt[a:a + 256].unsqueeze(1))[:, 0].cpu())
        del o; s += x.shape[1]
    del pkv; torch.cuda.empty_cache()
    return torch.cat(out)
def prep(i):
    row = rows[i]; npr, on = len(row["prompt_ids"]), row["onset"]
    ids = torch.cat([row["prompt_ids"], row["gen_ids"][:on + LOOP_SPAN]]).long()
    cls = np.concatenate([np.zeros(npr, np.int8), classes(row)[:on + LOOP_SPAN]])[1:]
    return row, ids, cls
names = [n for n in sys.argv[1:] if any((n, i) not in done for i in texts)]
base = {}
ctx.apply("none")
if names:
    for i in texts:
        base[i] = logps(prep(i)[1])
for vi, name in enumerate(names):
    changed = ctx.apply(name)
    for i in texts:
        if (name, i) in done: continue
        row, ids, cls = prep(i)
        dlt = (logps(ids) - base[i]).numpy()
        with open(OUT, "a") as f:
            f.write(json.dumps({"variant": name, "text": i, "id": row["id"], "weights_changed": changed,
                                "loop_delta": float(dlt[cls == 3].mean()), "prod_delta": float(dlt[cls == 1].mean())}) + "\n")
    print(f"{vi + 1}/{len(names)} {name} done", flush=True)
print("PROXY_DONE", flush=True)
