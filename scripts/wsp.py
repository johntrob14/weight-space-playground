"""Shared helpers: model loading (text-only, fp16 on the 4 GB GPU), prompts, MLP-neuron hooks."""
import glob
import os

import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REV = {
    "instruct": ("models--Qwen--Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "edited": ("models--RikiyaT--Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b"),
    "base": ("models--Qwen--Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# the post's evaluation sampling ("think35"): vLLM, thinking on
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)


def snapshot(key):
    d, rev = REV[key]
    return os.path.join(HUB, d, "snapshots", rev)


def load(key="instruct", dtype=torch.float16, device="cuda"):
    """Full release checkpoint; the vision tower is dropped (text prompts never use it)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    path = snapshot(key)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True)
    return tok, model


def layers(model):
    return model.model.language_model.layers


def render(tok, problem, thinking=True):
    return tok.apply_chat_template([{"role": "user", "content": problem + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


class NeuronTap:
    """Records the input of mlp.down_proj (the MLP neuron activations, act_fn(gate_proj x) * up_proj x)."""

    def __init__(self, model, layer_ids):
        self.acts, self.hooks = {}, []
        for l in layer_ids:
            self.hooks.append(layers(model)[l].mlp.down_proj.register_forward_pre_hook(self._mk(l)))

    def _mk(self, l):
        def hook(mod, args):
            self.acts[l] = args[0].detach()
        return hook

    def close(self):
        for h in self.hooks:
            h.remove()


def grouped_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                              is_causal=None, **kwargs):
    """Grouped-query attention without materialising the repeated key/value heads.

    Same result as the default path (which expands 2 key/value heads to 8 on every call, copying the whole cache
    each generated token). query [B, H, T, D]; key, value [B, G, L, D]; head h uses group h // (H // G).
    """
    B, H, T, D = query.shape
    G, L = key.shape[1], key.shape[2]
    R = H // G
    scaling = D ** -0.5 if scaling is None else scaling
    q = query.reshape(B * G, R * T, D)
    k = key.reshape(B * G, L, D)
    v = value.reshape(B * G, L, D)
    # scores^T = k @ q^T keeps the big operand contiguous; [BG, L, RT] -> [B, G, R, T, L]
    scores = torch.bmm(k, q.transpose(1, 2)).transpose(1, 2).reshape(B, G, R, T, L) * scaling
    if attention_mask is not None:
        m = attention_mask[..., :L]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m.reshape(B, 1, 1, m.shape[-2], L), float("-inf"))
        else:
            scores = scores + m.reshape(B, 1, 1, m.shape[-2], L).to(scores.dtype)
    elif T > 1 and (is_causal if is_causal is not None else getattr(module, "is_causal", True)):
        past = L - T
        allowed = torch.arange(L, device=query.device)[None, :] <= (past + torch.arange(T, device=query.device))[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.bmm(p.reshape(B * G, R * T, L), v).reshape(B, H, T, D)
    return out.transpose(1, 2).contiguous(), None


def enable_grouped_attention(model):
    """Route the full-attention layers through grouped_attention_forward."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register("grouped", grouped_attention_forward)
    for l in layers(model):
        if hasattr(l, "self_attn"):
            l.self_attn.config = _with_impl(l.self_attn.config, "grouped")


def _with_impl(config, name):
    import copy
    c = copy.copy(config)
    c._attn_implementation_internal = name
    return c
