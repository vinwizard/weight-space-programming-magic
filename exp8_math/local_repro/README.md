# Experiment 8, reproduced on a laptop

This folder redoes the p50c edit of Experiment 8 on one Mac (Apple M5 Pro, 24 GB) with no GPU server, no cloud and no
paid API. It downloads the public Qwen3.5-0.8B, rebuilds the published p50c edit, runs the forward passes that find
the loop direction, makes the same kind of edit from those passes, and scores three models (the original, their p50c
edit, our edit) on the same small subset of MATH-500 and GSM8K. No Hugging Face login is needed: the model and the
datasets are public.

## Rerun it

```sh
cd exp8_math/local_repro
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14 was used
.venv/bin/python download.py       # ~1.6 GB -> weights/hf, then the text-only model -> weights/base
.venv/bin/python replay_edit.py    # their p50c edit -> weights/their_edit (sha256 checked)
.venv/bin/python find_loop.py      # forward passes -> out/loop.npz and the table below (about a minute)
.venv/bin/python make_edit.py      # our edit -> weights/our_edit, compared with theirs
nohup nice -n 10 .venv/bin/python score.py their_edit our_edit base > out/score.log 2>&1 &   # hours; rerun to resume
```

`weights/`, `out/` and `.venv/` are ignored by git (`.gitignore` in this folder).

| File | What it does |
|---|---|
| `download.py` | downloads `Qwen/Qwen3.5-0.8B` at the revision `../code/apply_edit.py` names (`2fc0636…`) and writes `weights/base`: the release's `model.language_model.*` tensors with the configs of `../base_model/Qwen3.5-0.8B-text` (the text-only model the experiment edited) |
| `replay_edit.py` | rebuilds the p50c edit with `../code/apply_edit.py`, which checks the sha256 of every base and rebuilt tensor against the evaluated checkpoint |
| `find_loop.py` | transformers on `mps`: forward passes on Astra's 8 matched texts, the per-layer loop direction, the loop-growth table |
| `make_edit.py` | the top-64 columns of `down_proj` in layers 15 and 19, strength 6 -> `weights/our_edit`; compares with theirs |
| `score.py` | mlx-lm generation on 100 MATH-500 + 100 GSM8K problems, graded with `../code/mathqa.py` |

## The recipe, and where it comes from in the p50c run

All of it is in `../runs/p50c/turns/*/answer.md`:

1. **Texts** (turn t009, pass `fwd_4`): 4 pairs; each pair is one practice problem in the raw thinking template with
   the same correct start, then a productive continuation (even index) or a looping one (odd index). `find_loop.py`
   reads them from `../runs/p50c/requests/fwd_4.json.claimed`.
2. **Loop direction** (turns t010, t017, t025): per text, the mean of each hidden state over its last 36 tokens;
   `diffH[l]` = the mean over the 4 pairs of (looping - productive). The direction removed from MLP `l` is the unit
   vector `u = diffH[l+1] / |diffH[l+1]|` (the hidden state after layer `l`). Turn t026 removed it from layers 15 and
   19 at strength 1; turn t043 recovered `u` from that edit and saved it as the run's `repetition_directions.pt`.
3. **Top 64 neurons** (turn t068): score of `down_proj` column `j` = `(u^T W0)_j` x the mean over pairs of (looping -
   productive) MLP activation `j` (the input of `down_proj`), averaged over the last **24** tokens of the `fwd_4`
   texts; the 64 highest scores.
4. **Edit** (turn t105, kept to the end): `W = W0 - 6 u (mask64 * u^T W0)` on `layers.15.mlp.down_proj` and
   `layers.19.mlp.down_proj`, cast to bfloat16.

## The forward pass (output of `find_loop.py`)

```
text 0 (124 tokens): hidden_states 25 x (1, 124, 1024), mlp_act 24 x (1, 124, 3584), torch.bfloat16
token counts [124, 123, 131, 136, 107, 146, 98, 135] same as fwd_4

layer  contrast norm  block growth     MLP  attention
    0          0.164        0.1586  0.0304     0.1281
    1          0.275        0.1787  0.0373     0.1414
    2          0.347        0.1365  0.0343     0.1022
    3          0.395        0.1399  0.0531     0.0869
    4          0.410        0.0795  0.0343     0.0452
    5          0.445        0.0888  0.0236     0.0652
    6          0.562        0.2046  0.0959     0.1087
    7          0.555        0.1475  0.0513     0.0962
    8          0.578        0.0962  0.0223     0.0740
    9          0.629        0.1100  0.0092     0.1008
   10          0.758        0.2309  0.0371     0.1938
   11          0.817        0.1784  0.0359     0.1425
   12          0.928        0.2024  0.0416     0.1608
   13          1.088        0.2438  0.0934     0.1504
   14          1.273        0.3617  0.1494     0.2123
   15          1.564        0.4716  0.2629     0.2087
   16          1.759        0.2797  0.1230     0.1567
   17          1.876        0.1921  0.0494     0.1427
   18          1.908        0.1753  0.0788     0.0966
   19          2.263        0.5285  0.2489     0.2796
   20          2.325        0.1329  0.1051     0.0278
   21          2.512        0.2571  0.1641     0.0931
   22          2.625        0.3105  0.1111     0.1994
```

"contrast norm" is `|diffH[l+1]|`; "block growth" is how much layer `l` adds to the loop contrast along its own
direction, `(diffH[l+1] - diffH[l]) . u`; "MLP" is the part its MLP writes, `(W_down @ mean activation contrast) . u`;
"attention" is the rest. Layers 15 and 19 grow the loop the most, and their MLPs write the largest share of it. This is
the table Astra printed in turn t010 (`../runs/p50c/turns/t011/prompt.txt`); every value agrees to about 0.002
(layer 15: 0.4716 here, 0.472 there; layer 19: 0.5285 here, 0.5304 there).

`make_edit.py` then gives:

```
layer 15: |cos(our u, their u)| 1.0000; top-64 columns shared with theirs 64/64; |ours - theirs| / |their edit| 0.007
layer 19: |cos(our u, their u)| 1.0000; top-64 columns shared with theirs 64/64; |ours - theirs| / |their edit| 0.009
```

Our edit picks the same 128 neurons as theirs, along the same direction; the two edited tensors differ by under 1 % of
the size of the edit (bfloat16 forward passes on Apple's GPU instead of the experiment's server).

## Results

RESULTS_PLACEHOLDER

## Where this differs from the experiment, or had to guess

- **The forward-pass server is not in the repo.** `hidden_l` is taken as transformers' `hidden_states[l]` (the
  embeddings, then the output of each of the 24 layers) and `mlp_act_l` as the input of `down_proj` (3,584 values).
  The token counts match `fwd_4` exactly and the t010 table matches to 3 decimals, so these readings are right.
- **The sign of `u`.** Turn t068 flipped `u` to point along the loop contrast of another pass (`fwd_24`, last 24
  tokens); here `u` points along (looping - productive) of `fwd_4` by construction. Same 64 columns, so same sign.
- **Scoring is not the paper's evaluation.** mlx-lm on the Mac instead of vLLM; 1 sample per problem instead of 4
  (MATH-500) and 2 (GSM8K); a fixed subset of 100 problems per benchmark (`random.Random(0).sample`, the same for
  every model); at most **8,192** new tokens instead of 32,768. The prompt, chat template (thinking on), sampling
  (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 over the generated tokens) and grader are the
  `think35` protocol's. The 8k cap cuts many more answers than 32k does, so the accuracies are lower than the post's;
  the comparison between the three models is the point.
- mlx-lm 0.32 does not know the text-only config's `model_type` (`qwen3_5_text`); `score.py` loads it as `qwen3_5`,
  whose loader maps the same `model.language_model.*` tensor names.
- `../code/mathqa.py` imports a multiple-choice module (`mcqa`) that is not in the repo; `score.py` stubs it, since
  grading never calls it.
