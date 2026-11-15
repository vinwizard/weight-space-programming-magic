"""Score models on the same 100 MATH-500 + 100 GSM8K test problems (random.Random(0)), 1 sample each, with the
evaluation's think35 protocol (thinking on; temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5) but at most
8,192 new tokens; graded with ../code/mathqa.py (the last \\boxed{} answer, math-verify). Generation with mlx-lm.
Answers are appended to out/<model>/answers.jsonl, so a rerun resumes where it stopped.
    python score.py base their_edit our_edit      (folders under weights/; prints the table of every scored model)"""
import json
import os
import random
import sys
import types

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_logits_processors, make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # keep ../code free of __pycache__
sys.path.insert(0, os.path.join(HERE, "..", "code"))
sys.modules.setdefault("mcqa", types.SimpleNamespace(loop_signals=lambda t: {"repetitive_end": None}))   # mathqa imports it
import mathqa as Q  # noqa: E402

N, MAX_TOKENS, BATCH, CHUNK = 100, 8192, 32, 64


def problems():
    m = load_dataset("HuggingFaceH4/MATH-500", split="test")
    g = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"set": "math500", "id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in m]
    gs = [{"set": "gsm8k", "id": f"gsm8k_test_{i}", "problem": r["question"],
           "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(g)]
    rng = random.Random(0)
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), N))] + [gs[i] for i in sorted(rng.sample(range(len(gs)), N))]


def generate(name, todo):
    path = os.path.join(HERE, "out", name, "answers.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    todo = [r for r in todo if r["id"] not in done]
    if not todo:
        return
    model, tok = load(os.path.join(HERE, "weights", name), model_config={"model_type": "qwen3_5"})
    mx.random.seed(len(done))
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}],
                                       add_generation_prompt=True, enable_thinking=True) for r in todo]
    # a fresh generator per CHUNK prompts: mlx-lm's per-sequence presence-penalty buffers pile up Metal buffers
    # until "[metal::malloc] Resource limit (499000) exceeded" (hit after ~140 sequences in one generator)
    for c in range(0, len(todo), CHUNK):
        gen = BatchGenerator(model, max_tokens=MAX_TOKENS, stop_tokens=[[t] for t in tok.eos_token_ids],
                             sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                             logits_processors=make_logits_processors(presence_penalty=1.5,
                                                                      presence_context_size=MAX_TOKENS),
                             completion_batch_size=BATCH)
        rows = dict(zip(gen.insert(prompts[c:c + CHUNK]), todo[c:c + CHUNK]))
        toks = {u: [] for u in rows}
        with open(path, "a") as f:
            while responses := gen.next_generated():
                for res in responses:
                    if res.finish_reason != "stop":
                        toks[res.uid].append(res.token)
                    if res.finish_reason is None:
                        continue
                    r, t = rows[res.uid], toks.pop(res.uid)
                    text = tok.decode(t)
                    pred = Q.extract(text)[0]
                    ok = Q.is_correct(pred, r["answer"])
                    f.write(json.dumps({**r, "text": text, "tokens": len(t), "cut": res.finish_reason == "length",
                                        "pred": pred, "correct": ok}) + "\n")
                    f.flush()
                    done.add(r["id"])
                    print(f"{name} {len(done)}/{2 * N} {r['id']} tokens {len(t)} correct {ok}", flush=True)
        gen.close()
        del gen
        mx.clear_cache()


def table():
    print("\nmodel        set      n   accuracy  cut at 8,192")
    for name in sorted(os.listdir(os.path.join(HERE, "out"))):
        p = os.path.join(HERE, "out", name, "answers.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        for s in ("math500", "gsm8k"):
            x = [r for r in rows if r["set"] == s]
            if x:
                print(f"{name:12s} {s:8s} {len(x):3d}  {100 * sum(r['correct'] for r in x) / len(x):7.1f}%  "
                      f"{100 * sum(r['cut'] for r in x) / len(x):10.1f}%")
        cut = [r for r in rows if r["cut"]]
        loop = sum(Q.loop_signals(r["text"])["verbatim_loop_end"] for r in cut)
        print(f"{name:12s} cut answers ending in a verbatim loop: {loop} of {len(cut)}")


if __name__ == "__main__":
    todo = problems()
    for name in sys.argv[1:]:
        os.makedirs(os.path.join(HERE, "out", name), exist_ok=True)
        generate(name, todo)
    table()
