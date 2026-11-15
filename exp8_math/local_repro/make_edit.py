"""Our own p50c-style edit from out/loop.npz (find_loop.py): for MLP layers 15 and 19, u = the unit loop direction of
the layer's output (hidden state l+1, last 36 tokens; turns t017/t026/t043); rank the 3,584 down_proj columns by
(u.W)_j x the loop contrast of activation j (last 24 tokens; turn t068); remove u from the top 64 at strength 6
(turn t105): W = W0 - 6 u (mask64 * u^T W0). Compared with their edit, then written to weights/our_edit.
    python make_edit.py"""
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # keep ../code free of __pycache__
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import st_numpy as S  # noqa: E402

W = os.path.join(HERE, "weights")
LAYERS, TOP, ALPHA = (15, 19), 64, 6.0

if __name__ == "__main__":
    z = np.load(os.path.join(HERE, "out", "loop.npz"))
    base = S.load(os.path.join(W, "base", "model.safetensors"))
    theirs = S.load(os.path.join(W, "their_edit", "model.safetensors"), [f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in LAYERS])
    for l in LAYERS:
        name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        w0 = S.bf16_to_f32(base[name][1]).astype(np.float64)
        u = z["diff_h"][l + 1] / np.linalg.norm(z["diff_h"][l + 1])
        writes = u @ w0
        top = np.argsort(-(writes * z["dm24"][l]))[:TOP]
        mask = np.zeros_like(writes)
        mask[top] = 1
        w = w0 - ALPHA * np.outer(u, mask * writes)
        base[name] = ("BF16", S.bf16_bits(w.astype(np.float32)))
        d_theirs = S.bf16_to_f32(theirs[name][1]) - w0
        cols = np.flatnonzero(np.abs(d_theirs).max(0) > 0)
        u_theirs = d_theirs[:, np.argmax((d_theirs ** 2).sum(0))]
        cos = abs(u @ u_theirs) / np.linalg.norm(u_theirs)
        rel = np.linalg.norm(S.bf16_to_f32(base[name][1]) - S.bf16_to_f32(theirs[name][1])) / np.linalg.norm(d_theirs)
        print(f"layer {l}: |cos(our u, their u)| {cos:.4f}; top-{TOP} columns shared with theirs "
              f"{len(set(top) & set(cols))}/{len(cols)}; |ours - theirs| / |their edit| {rel:.3f}")
    out = os.path.join(W, "our_edit")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(os.path.join(W, "base")):
        if f != "model.safetensors":
            shutil.copy(os.path.join(W, "base", f), out)
    S.save(os.path.join(out, "model.safetensors"), base, {"format": "pt"})
    print(f"-> {out}")
