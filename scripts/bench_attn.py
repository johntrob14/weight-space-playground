"""Compare the default attention path with grouped_attention_forward: outputs and generation speed at long context.
    python -u scripts/bench_attn.py [grouped]
"""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import wsp
mode = sys.argv[1] if len(sys.argv) > 1 else "default"
tok, model = wsp.load("instruct", dtype=torch.bfloat16); model.model.visual = None; model.eval().to("cuda")
if mode == "grouped":
    wsp.enable_grouped_attention(model)
print(mode, "attn impl seen by layer 3:", model.model.language_model.layers[3].self_attn.config._attn_implementation)
rows = torch.load(os.path.join(wsp.ROOT, "results/data/gens.pt"))
r = next(r for r in rows if r["model"] == "unedited" and r["onset"] is not None)
ids = torch.cat([r["prompt_ids"], r["gen_ids"]]).long()
torch.manual_seed(0)
with torch.no_grad():
    full = model(input_ids=ids[:1024].unsqueeze(0).to("cuda")).logits[0].float().cpu()
    o = model(input_ids=ids[:992].unsqueeze(0).to("cuda"), use_cache=True); pkv = o.past_key_values; step = []
    for i in range(992, 1024):
        s = model(input_ids=ids[i:i+1].unsqueeze(0).to("cuda"), past_key_values=pkv, use_cache=True); pkv = s.past_key_values
        step.append(s.logits[0, -1].float().cpu())
torch.save({"full": full[-64:], "step": torch.stack(step)}, os.path.join(wsp.ROOT, f"results/bench/attn_{mode}.pt"))
del o, s, pkv
lm = model.model.language_model
for bs, ctx, new in ((16, 8100, 80), (12, 8100, 80)):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            pkv = None
            for a in range(0, ctx, 100):
                x = ids[a:a + 100].unsqueeze(0).repeat(bs, 1).to("cuda")
                o = lm(input_ids=x, past_key_values=pkv, use_cache=True); pkv = o.past_key_values; del o
            nxt = ids[ctx:ctx + 1].unsqueeze(0).repeat(bs, 1).to("cuda")
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(new):
                o = model(input_ids=nxt, past_key_values=pkv, use_cache=True); pkv = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1, keepdim=True); del o
            torch.cuda.synchronize(); dt = time.time() - t0
        print(f"   batch {bs}, context {ctx}: {bs * new / dt:.0f} tok/s ({dt / new * 1000:.0f} ms/step), peak {torch.cuda.max_memory_allocated() // 2**20} MiB", flush=True)
    except torch.OutOfMemoryError:
        print(f"   batch {bs}, context {ctx}: OOM", flush=True)
    pkv = x = nxt = None
