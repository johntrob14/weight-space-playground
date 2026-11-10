"""Does the experiment fit on the GTX 1650 Ti (4 GB)? Numerics, memory and speed checks on the unedited model.

    python scripts/fitcheck.py > results/fitcheck.log     (writes results/fitcheck.json)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402

out = {}
MB = 2 ** 20


def gpu():
    return {"alloc_MiB": torch.cuda.memory_allocated() // MB, "peak_MiB": torch.cuda.max_memory_allocated() // MB,
            "free_MiB": torch.cuda.mem_get_info()[0] // MB}


torch.manual_seed(0)
tok, model = wsp.load("instruct", dtype=torch.bfloat16)   # as stored
model.eval()
n_all = sum(p.numel() for p in model.parameters())
n_text = sum(p.numel() for p in model.model.language_model.parameters())
out["params_all"], out["params_text"] = n_all, n_text
print(f"params: all {n_all:,}, language model {n_text:,} (embedding tied: "
      f"{model.lm_head.weight is model.model.language_model.embed_tokens.weight})")
print("top-level modules of model.model:", [n for n, _ in model.model.named_children()])
mlp = wsp.layers(model)[15].mlp
print("layer 15 mlp:", mlp)

# 1. is bfloat16 -> float16 lossless for these weights?
bad = tot = 0
worst = 0.0
for n, p in model.model.language_model.named_parameters():
    a = p.detach().float()
    b = p.detach().to(torch.float16).float()
    d = (a != b)
    bad += int(d.sum())
    tot += a.numel()
    if d.any():
        worst = max(worst, float((a - b).abs().max()))
out["bf16_to_fp16_changed_entries"] = bad
out["bf16_to_fp16_max_abs_err"] = worst
print(f"1. bf16 -> fp16 cast: {bad:,} of {tot:,} language-model entries change value (max abs error {worst:.3e})")

# drop the vision tower
vis = [n for n, _ in model.model.named_children() if n != "language_model"]
for n in vis:
    setattr(model.model, n, None)

problem = ("How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?")
text = wsp.render(tok, problem)
print("\n--- rendered prompt (thinking on) ---\n" + text + "\n--- end ---")
ids = tok(text, return_tensors="pt").input_ids
print("prompt tokens:", ids.shape[1])
# a longer teacher-forced text: prompt + a worked solution, to compare numerics over more positions
ids_long = tok(text + "Okay, let's see. We need to find the vertical asymptotes of y = 2/(x^2+x-6). "
               "First, factor the denominator: x^2 + x - 6 = (x+3)(x-2). The numerator is 2, which is never zero. "
               "So the denominator is zero at x = -3 and x = 2, and there are two vertical asymptotes.\n</think>\n\n"
               "The denominator factors as $(x+3)(x-2)$, so there are $\\boxed{2}$ vertical asymptotes.",
               return_tensors="pt").input_ids


def run(m, x, dev):
    tap = wsp.NeuronTap(m, [15, 19])
    with torch.no_grad():
        lg = m(input_ids=x.to(dev)).logits[0].float().cpu()
    acts = {l: a[0].float().cpu() for l, a in tap.acts.items()}
    tap.close()
    return lg, acts


# 2. fp32 CPU reference
m32 = model.float()
t0 = time.time()
ref_lg, ref_act = run(m32, ids_long, "cpu")
print(f"\n2. fp32 CPU reference forward on {ids_long.shape[1]} tokens: {time.time() - t0:.1f}s; "
      f"neuron act shape L15 {tuple(ref_act[15].shape)}")
out["neuron_act_shape"] = list(ref_act[15].shape)
ref_lp = torch.log_softmax(ref_lg, -1)


def compare(tag, lg, acts):
    lp = torch.log_softmax(lg, -1)
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
    r = {"finite": bool(torch.isfinite(lg).all()),
         "top1_agree": float((lg.argmax(-1) == ref_lg.argmax(-1)).float().mean()),
         "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
         "logit_max_abs_diff": float((lg - ref_lg).abs().max())}
    for l in acts:
        a, b = acts[l], ref_act[l]
        r[f"act_L{l}_finite"] = bool(torch.isfinite(a).all())
        r[f"act_L{l}_rel_err"] = float((a - b).norm() / b.norm())
        r[f"act_L{l}_abs_max"] = float(b.abs().max())
    out[tag] = r
    print(f"   {tag}: {r}")


# 3. GPU in fp16 and bf16, compared with the fp32 CPU reference
for tag, dt in (("gpu_fp16", torch.float16), ("gpu_bf16", torch.bfloat16)):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = m32.to(dt).to("cuda")
    out[tag + "_weights_MiB"] = torch.cuda.memory_allocated() // MB
    t0 = time.time()
    lg, acts = run(m, ids_long, "cuda")
    torch.cuda.synchronize()
    t1 = time.time() - t0
    t0 = time.time()
    for _ in range(3):
        run(m, ids_long, "cuda")
    torch.cuda.synchronize()
    out[tag + "_fwd_s"] = (time.time() - t0) / 3
    print(f"\n3. {tag}: weights on GPU {out[tag + '_weights_MiB']} MiB; forward {out[tag + '_fwd_s']:.3f}s "
          f"(first call {t1:.2f}s)")
    compare(tag, lg, acts)
    if tag == "gpu_bf16":
        m32 = m.to("cpu").float()
    else:
        m32 = m.to("cpu").float()
        # refresh fp32 weights from bf16 originals is not needed: fp16 cast checked lossless above if bad == 0
    del m
    torch.cuda.empty_cache()

# 4. memory and speed in fp16: teacher-forced forwards at increasing length, with neuron taps on
model = m32.to(torch.float16).to("cuda")
del m32
torch.cuda.empty_cache()
out["tf"] = {}
print("\n4. teacher-forced forward (fp16, batch 1, taps on L15/L19):")
for n in (1024, 4096, 8192, 16384, 32768):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(1000, 50000, (1, n), device="cuda")
        tap = wsp.NeuronTap(model, [15, 19])
        t0 = time.time()
        with torch.no_grad():
            lg = model(input_ids=x).logits
        torch.cuda.synchronize()
        dt_ = time.time() - t0
        fin = bool(torch.isfinite(lg).all())
        tap.close()
        del lg, tap
        out["tf"][n] = {"s": dt_, "peak_MiB": torch.cuda.max_memory_allocated() // MB, "finite": fin}
        print(f"   {n:6d} tokens: {dt_:.2f}s, peak {out['tf'][n]['peak_MiB']} MiB, finite {fin}")
    except torch.OutOfMemoryError as e:
        out["tf"][n] = {"oom": True}
        print(f"   {n:6d} tokens: OOM")
        break

# 5. generation speed with KV cache
out["gen"] = {}
print("\n5. sampled generation (fp16, temperature 0.6, top_p 0.95, top_k 20), 256 new tokens:")
for bs in (1, 8, 16):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = tok([text] * bs, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=256, min_new_tokens=256, do_sample=True, temperature=0.6,
                               top_p=0.95, top_k=20)
        torch.cuda.synchronize()
        dt_ = time.time() - t0
        out["gen"][bs] = {"s": dt_, "tok_per_s": bs * 256 / dt_, "peak_MiB": torch.cuda.max_memory_allocated() // MB}
        print(f"   batch {bs:2d}: {dt_:.1f}s, {bs * 256 / dt_:.0f} tok/s, peak {out['gen'][bs]['peak_MiB']} MiB")
        if bs == 1:
            print("   sample:", repr(tok.decode(g[0, enc.input_ids.shape[1]:])[:300]))
    except torch.OutOfMemoryError:
        out["gen"][bs] = {"oom": True}
        print(f"   batch {bs}: OOM")
        break

out["gpu_total_MiB"] = torch.cuda.mem_get_info()[1] // MB
out["torch"], out["device"] = torch.__version__, torch.cuda.get_device_name(0)
json.dump(out, open(os.path.join(wsp.ROOT, "results/fitcheck.json"), "w"), indent=1)
print("\ndone")
