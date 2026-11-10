"""Fit check, part 2: memory and speed on the 4 GB GPU, with clean state between measurements.

fitcheck.py's sections 4-5 were contaminated: after the OOM at 4096 tokens the hooks and tensors stayed alive, and
the full-vocabulary logits (seq x 248,320) were kept. Here: hidden states only (no lm_head) for teacher forcing,
try/finally cleanup, and both dtypes.

    python scripts/fitcheck2.py [fp16|bf16] > results/fitcheck2_<dtype>.log
"""
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402

MB = 2 ** 20
name = sys.argv[1] if len(sys.argv) > 1 else "fp16"
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
out = {"dtype": name}
torch.manual_seed(0)
tok, model = wsp.load("instruct", dtype=dtype)
model.model.visual = None
model.eval().to("cuda")
lm = model.model.language_model
out["weights_MiB"] = torch.cuda.memory_allocated() // MB
print(f"{name}: weights on GPU {out['weights_MiB']} MiB, total {torch.cuda.mem_get_info()[1] // MB} MiB")


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


text = wsp.render(tok, "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?")
# realistic tokens: repeat a math text to the wanted length
body = tok(text + "We factor the denominator: x^2 + x - 6 = (x+3)(x-2). So the zeros are x = -3 and x = 2. " * 4000,
           return_tensors="pt").input_ids

out["tf"] = {}
print("\nteacher-forced forward, hidden states only, taps on L15/L19 (acts moved to CPU by the caller):")
for bs, n in ((1, 1024), (1, 4096), (1, 8192), (4, 1024)):
    key = f"{bs}x{n}"
    clean()
    tap = wsp.NeuronTap(model, [15, 19])
    try:
        x = body[:, :n].repeat(bs, 1).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            h = lm(input_ids=x).last_hidden_state
        torch.cuda.synchronize()
        dt_ = time.time() - t0
        fin = bool(torch.isfinite(h).all()) and all(bool(torch.isfinite(a).all()) for a in tap.acts.values())
        out["tf"][key] = {"s": dt_, "tok_per_s": bs * n / dt_, "peak_MiB": torch.cuda.max_memory_allocated() // MB,
                          "finite": fin}
        print(f"   {key:>9}: {dt_:6.2f}s, {bs * n / dt_:6.0f} tok/s, peak {out['tf'][key]['peak_MiB']} MiB, "
              f"finite {fin}")
    except torch.OutOfMemoryError:
        out["tf"][key] = {"oom": True}
        print(f"   {key:>9}: OOM")
    finally:
        tap.close()
        tap.acts.clear()
        x = h = None

out["gen"] = {}
print("\nsampled generation with KV cache (temperature 0.6, top_p 0.95, top_k 20), fixed new-token count:")
for bs, new in ((1, 512), (4, 512), (1, 2048)):
    key = f"{bs}x{new}"
    clean()
    try:
        enc = tok([text] * bs, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=new, min_new_tokens=new, do_sample=True, temperature=0.6,
                               top_p=0.95, top_k=20)
        torch.cuda.synchronize()
        dt_ = time.time() - t0
        out["gen"][key] = {"s": dt_, "tok_per_s": bs * new / dt_, "peak_MiB": torch.cuda.max_memory_allocated() // MB}
        print(f"   {key:>8}: {dt_:6.1f}s, {bs * new / dt_:5.0f} tok/s total, peak {out['gen'][key]['peak_MiB']} MiB")
        if key == "1x2048":
            s = tok.decode(g[0, enc.input_ids.shape[1]:])
            print("   tail of the 2048-token sample:", repr(s[-400:]))
    except torch.OutOfMemoryError:
        out["gen"][key] = {"oom": True}
        print(f"   {key:>8}: OOM")
    finally:
        enc = g = None

json.dump(out, open(os.path.join(wsp.ROOT, f"results/fitcheck2_{name}.json"), "w"), indent=1)
print("\ndone")
