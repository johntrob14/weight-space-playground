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
