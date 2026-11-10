"""Behavioral check: does an intervention stop looping in real generations?

Protocol (the post's, except the token cap): thinking on, user message = problem + boxed-answer instruction,
temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, 1 sample per problem. Token cap MAX_NEW instead of
32,768 (stated deviation). Problems: MATH train practice problems from the logs repo (never used to fit anything
here). Prompts are batched only with prompts of the same token length, so no padding is needed.

Per sample: tokens generated, finished (EOS), thinking closed, strict loop onset (prep_data.onset), repeat fraction,
last \\boxed{} answer and whether math-verify accepts it.

Interventions are named variants built in variants.py: weight edits of mlp.down_proj and/or additive residual
steering vectors.

    python -u scripts/gen_eval.py <variant> [<variant> ...] [--n 32] [--max_new 8192] [--batch 16]
    -> results/gen_eval.jsonl (append-only; (variant, problem id) already present are skipped)
"""
import json
import os
import re
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp  # noqa: E402
from prep_data import onset, repeat_flags  # noqa: E402

OUT = os.path.join(wsp.ROOT, "results/gen_eval.jsonl")


class PresencePenalty(LogitsProcessor):
    """vLLM/OpenAI presence penalty: subtract `penalty` from the logit of every token already generated."""

    def __init__(self, penalty, prompt_len):
        self.penalty, self.prompt_len = penalty, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
        return scores - self.penalty * seen


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[j + 1:k]
        k += 1
    return None


def grade(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        norm = lambda s: re.sub(r"\s+|\\left|\\right|\\!|\$", "", s).replace("dfrac", "frac")  # noqa: E731
        return norm(pred) == norm(gold)


def load_rows(source):
    if source == "practice":
        path = os.path.join(wsp.ROOT, "ext/lm_understand-logs/exp8_math/practice/math_train.jsonl")
        return [json.loads(l) for l in open(path)]
    # MATH-500 problems whose published rollouts were NOT used to fit any direction or ranking here
    from datasets import load_dataset
    used = {json.loads(l)["id"] for l in open(os.path.join(wsp.ROOT, "results/data/gens_summary.jsonl"))}
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in ds
            if r["unique_id"] not in used]


def problems(tok, n, bs, seed=0, source="practice"):
    """n practice problems in groups of bs with identical prompt token length (so batches need no padding).
    Lengths are taken in order of closeness to the median prompt length, among lengths with at least bs problems."""
    rows = load_rows(source)
    ids = [tok(wsp.render(tok, r["problem"]), add_special_tokens=False).input_ids for r in rows]
    by = {}
    for r, x in zip(rows, ids):
        by.setdefault(len(x), []).append((r, x))
    med = float(np.median([len(x) for x in ids]))
    rng = np.random.default_rng(seed)
    out = []
    for L in sorted((L for L in by if len(by[L]) >= bs), key=lambda L: (abs(L - med), L)):
        g = by[L]
        pick = rng.choice(len(g), bs, replace=False)
        out += [g[i] for i in sorted(pick.tolist())]
        if len(out) >= n:
            break
    return out[:n]


def main():
    args = sys.argv[1:]
    opt = {"--n": 32, "--max_new": 8192, "--batch": 16}
    source = "practice"
    if "--math500_heldout" in args:
        args.remove("--math500_heldout")
        source = "math500_heldout"
    global OUT
    if source != "practice":
        OUT = os.path.join(wsp.ROOT, "results/gen_eval_m500.jsonl")
    names = []
    i = 0
    while i < len(args):
        if args[i] in opt:
            opt[args[i]] = int(args[i + 1])
            i += 2
        else:
            names.append(args[i])
            i += 1
    n, max_new, bs = opt["--n"], opt["--max_new"], opt["--batch"]
    import variants
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            d = json.loads(line)
            if d["max_new"] == max_new:
                done.add((d["variant"], d["id"]))
    tok, model = wsp.load("instruct", dtype=torch.bfloat16)
    model.model.visual = None
    model.eval().to("cuda")
    wsp.enable_grouped_attention(model)      # same outputs, no per-token copy of the key/value cache
    ctx = variants.Context(model)
    enc = problems(tok, n, bs, source=source)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")       # also stop here (added 2026-11-11 for the MATH-500 runs)
    stops = [eos, eot] if source != "practice" else [eos]
    for name in names:
        ctx.apply(name)
        todo = [(p, ids) for p, ids in enc if (name, p["id"]) not in done]
        # batches of equal prompt length (no padding), up to bs
        groups = {}
        for p, ids in todo:
            groups.setdefault(len(ids), []).append((p, ids))
        batches = []
        leftovers = []
        for L, g in sorted(groups.items()):
            for a in range(0, len(g), bs):
                batches.append(g[a:a + bs])
        t0, ntok = time.time(), 0
        torch.manual_seed(0)
        for b in batches:
            x = torch.tensor([ids for _, ids in b], device="cuda")
            P = x.shape[1]
            with torch.no_grad():
                g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                   do_sample=True, temperature=0.6, top_p=0.95, top_k=20, eos_token_id=stops,
                                   pad_token_id=eos,
                                   logits_processor=LogitsProcessorList([PresencePenalty(1.5, P)]))
            for (p, _), row in zip(b, g[:, P:].tolist()):
                cutpos = [row.index(t) for t in stops if t in row]
                fin = bool(cutpos)
                gen = row[:min(cutpos)] if fin else row
                text = tok.decode(gen)
                r = repeat_flags(gen)
                on = onset(r)
                pred = last_boxed(text.split("</think>")[-1]) if "</think>" in text else None
                rec = {"variant": name, "id": p["id"], "max_new": max_new, "n_tokens": len(gen), "finished": fin,
                       "think_closed": "</think>" in text, "onset": on, "looped": on is not None,
                       "repeat_frac": float(r.mean()) if len(r) else 0.0, "pred": pred,
                       "correct": grade(pred, p["answer"]), "answer": p["answer"], "text": text}
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                ntok += len(gen)
            del g
            torch.cuda.empty_cache()
            print(f"{name}: batch of {len(b)} done, {ntok:,} tokens, {ntok / (time.time() - t0):.0f} tok/s, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        print(f"VARIANT_DONE {name}", flush=True)
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
