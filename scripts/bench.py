"""Speed and numerics benchmark for one configuration (bf16 on the GPU).

    python -u scripts/bench.py <tag> [nofla] [gen=1x512,4x512,...] [tf=1x4096,...]

`nofla` blocks the flash-linear-attention import so transformers uses its reference PyTorch code.
Saves the last hidden state and L15/L19 neuron activations on a fixed text to results/bench/<tag>.pt for comparison.
"""
import gc
import json
import os
import sys
import time

args = sys.argv[1:]
tag = args[0]
if "nofla" in args:
    sys.modules["fla"] = None
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402

MB = 2 ** 20
opt = dict(a.split("=") for a in args[1:] if "=" in a)
gen_grid = [tuple(map(int, x.split("x"))) for x in opt.get("gen", "1x512,4x512,8x512").split(",") if x]
tf_grid = [tuple(map(int, x.split("x"))) for x in opt.get("tf", "1x4096,4x1024").split(",") if x]
torch.manual_seed(0)
tok, model = wsp.load("instruct", dtype=torch.bfloat16)
model.model.visual = None
model.eval().to("cuda")
lm = model.model.language_model
out = {"tag": tag, "tf": {}, "gen": {}}


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


text = wsp.render(tok, "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?")
fixed = tok(text + "Okay, let's see. We need to find the vertical asymptotes of y = 2/(x^2+x-6). "
            "First, factor the denominator: x^2 + x - 6 = (x+3)(x-2). The numerator is 2, which is never zero. "
            "So the denominator is zero at x = -3 and x = 2, and there are two vertical asymptotes.\n</think>\n\n"
            "The denominator factors as $(x+3)(x-2)$, so there are $\\boxed{2}$ vertical asymptotes.",
            return_tensors="pt").input_ids.to("cuda")
tap = wsp.NeuronTap(model, [15, 19])
with torch.no_grad():
    o = model(input_ids=fixed)
    # the same text through the cached, token-by-token path used in generation (last 32 tokens one at a time)
    pre = model(input_ids=fixed[:, :-32], use_cache=True)
    pkv, step_logits = pre.past_key_values, []
    for i in range(fixed.shape[1] - 32, fixed.shape[1]):
        s = model(input_ids=fixed[:, i:i + 1], past_key_values=pkv, use_cache=True)
        pkv = s.past_key_values
        step_logits.append(s.logits[0, -1].float().cpu())
tap.close()
os.makedirs(os.path.join(wsp.ROOT, "results/bench"), exist_ok=True)
torch.save({"logits": o.logits[0].float().cpu(), "step_logits": torch.stack(step_logits)},
           os.path.join(wsp.ROOT, f"results/bench/{tag}.pt"))
full, step = o.logits[0, -32:].float().cpu(), torch.stack(step_logits)
out["cached_vs_full_top1_agree"] = float((full.argmax(-1) == step.argmax(-1)).float().mean())
out["cached_vs_full_logit_max_abs_diff"] = float((full - step).abs().max())
print(f"{tag}: cached token-by-token vs full pass, last 32 positions: top-1 agree "
      f"{out['cached_vs_full_top1_agree']:.3f}, max |logit diff| {out['cached_vs_full_logit_max_abs_diff']:.3f}")
del o, pre, s, pkv

body = tok(text + "We factor the denominator: x^2 + x - 6 = (x+3)(x-2). So the zeros are x = -3 and x = 2. " * 2000,
           return_tensors="pt").input_ids
for bs, n in tf_grid:
    key = f"{bs}x{n}"
    clean()
    try:
        x = body[:, :n].repeat(bs, 1).to("cuda")
        for rep in range(2):   # second call excludes kernel compilation
            t0 = time.time()
            with torch.no_grad():
                h = lm(input_ids=x).last_hidden_state
            torch.cuda.synchronize()
            dt_ = time.time() - t0
        out["tf"][key] = {"s": dt_, "tok_per_s": bs * n / dt_, "peak_MiB": torch.cuda.max_memory_allocated() // MB}
        print(f"   tf  {key:>8}: {dt_:6.2f}s, {bs * n / dt_:6.0f} tok/s, peak {out['tf'][key]['peak_MiB']} MiB")
    except torch.OutOfMemoryError:
        out["tf"][key] = {"oom": True}
        print(f"   tf  {key:>8}: OOM")
    finally:
        x = h = None

for bs, new in gen_grid:
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
        print(f"   gen {key:>8}: {dt_:6.1f}s, {bs * new / dt_:5.0f} tok/s total, peak {out['gen'][key]['peak_MiB']} MiB")
    except torch.OutOfMemoryError:
        out["gen"][key] = {"oom": True}
        print(f"   gen {key:>8}: OOM")
    finally:
        enc = g = None
json.dump(out, open(os.path.join(wsp.ROOT, f"results/bench/{tag}.json"), "w"), indent=1)
print("done")
