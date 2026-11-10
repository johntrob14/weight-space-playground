"""Independent recompute of step 1 with no torch/safetensors: raw safetensors parsing with numpy, byte comparison."""
import glob, hashlib, json, os, struct
import numpy as np
HUB = os.path.expanduser("~/.cache/huggingface/hub")
def openst(d, rev):
    (p,) = glob.glob(f"{HUB}/{d}/snapshots/{rev}/*.safetensors")
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None)
    return np.memmap(p, dtype=np.uint8, mode="r", offset=8 + n), hdr
a, ha = openst("models--Qwen--Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
b, hb = openst("models--RikiyaT--Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b")
assert set(ha) == set(hb)
diff = []
for k in sorted(ha):
    (s0, e0), (s1, e1) = ha[k]["data_offsets"], hb[k]["data_offsets"]
    if ha[k]["dtype"] != hb[k]["dtype"] or ha[k]["shape"] != hb[k]["shape"] or not np.array_equal(a[s0:e0], b[s1:e1]):
        diff.append(k)
print("tensors:", len(ha), "differing:", diff)
card = {"model.language_model.layers.15.mlp.down_proj.weight": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
        "model.language_model.layers.19.mlp.down_proj.weight": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2"}
frozen = json.load(open("ext/lm_understand-logs/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json"))
tot = 0
for k in diff:
    assert ha[k]["dtype"] == "BF16"
    sh = ha[k]["shape"]
    x = np.frombuffer(a[slice(*ha[k]["data_offsets"])], dtype="<u2").reshape(sh)
    y = np.frombuffer(b[slice(*hb[k]["data_offsets"])], dtype="<u2").reshape(sh)
    m = x != y
    cols = np.flatnonzero(m.any(0)); L = k.split(".layers.")[1].split(".")[0]
    tot += len(cols) * sh[0]
    print(k, "shape [out,in] =", sh, "| changed entries", int(m.sum()), "| changed input columns", len(cols),
          "| changed output rows", int(m.any(1).sum()), "| edited sha matches model card:",
          hashlib.sha256(y.tobytes()).hexdigest() == card[k], "| columns == Astra's selected_neurons:",
          cols.tolist() == sorted(frozen[L]["selected_neurons"]))
print("columns x rows total:", tot)
