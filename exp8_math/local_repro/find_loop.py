"""Forward passes on Astra's 8 matched texts (p50c pass fwd_4, requested in turn t009: 4 pairs of a productive and a
looping continuation of the same thinking) -> hidden states and MLP activations -> the loop direction per layer and
the table Astra printed in turn t010 (how much each layer's block, MLP and attention grow the loop contrast).
    python find_loop.py      writes out/loop.npz"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(HERE, "..", "runs", "p50c", "requests", "fwd_4")
BASE, OUT = os.path.join(HERE, "weights", "base"), os.path.join(HERE, "out")


def tail_means(texts, tok, model):
    """per text: the mean over its last 36 and last 24 tokens of every hidden state (embeddings + 24 layer outputs)
    and of every MLP's activation (the input of down_proj)."""
    acts = {}
    hooks = [m.mlp.down_proj.register_forward_hook(lambda _m, i, _o, l=l: acts.__setitem__(l, i[0]))
             for l, m in enumerate(model.model.layers)]
    rows, lengths = [], []
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        mlp = [acts[l] for l in range(len(acts))]
        if i == 0:
            print(f"text 0 ({ids.shape[1]} tokens): hidden_states {len(hs)} x {tuple(hs[0].shape)}, "
                  f"mlp_act {len(mlp)} x {tuple(mlp[0].shape)}, {hs[0].dtype}")
        rows.append({k: [x[0, -k:].float().mean(0).cpu().numpy() for x in list(hs) + mlp] for k in (36, 24)})
        lengths.append(ids.shape[1])
    for h in hooks:
        h.remove()
    return rows, lengths


if __name__ == "__main__":
    texts = json.load(open(REQ + ".json.claimed"))["texts"]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("mps").eval()
    rows, lengths = tail_means(texts, tok, model)
    theirs = json.load(open(REQ + ".done.json"))["lengths"]
    print("token counts", lengths, "same as fwd_4" if lengths == theirs else f"DIFFER from fwd_4 {theirs}")
    nl = len(model.model.layers)

    def contrast(k, j):   # mean over the 4 pairs of (looping - productive), item j of the tail means
        return np.mean([rows[i + 1][k][j] - rows[i][k][j] for i in range(0, len(rows), 2)], 0)

    diff_h = np.stack([contrast(36, j) for j in range(nl + 1)])           # (25, 1024): the loop direction per layer
    dm36 = np.stack([contrast(36, nl + 1 + l) for l in range(nl)])         # (24, 3584)
    dm24 = np.stack([contrast(24, nl + 1 + l) for l in range(nl)])         # for ranking neurons (turn t068)
    print("\nlayer  contrast norm  block growth     MLP  attention")
    for l in range(nl - 1):
        d = diff_h[l + 1] / np.linalg.norm(diff_h[l + 1])
        w = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu().numpy()
        growth, mlp = (diff_h[l + 1] - diff_h[l]) @ d, (w @ dm36[l]) @ d
        print(f"{l:5d}  {np.linalg.norm(diff_h[l + 1]):13.3f}  {growth:12.4f}  {mlp:6.4f}  {growth - mlp:9.4f}")
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "loop.npz"), diff_h=diff_h, dm24=dm24)
    print("-> out/loop.npz")
