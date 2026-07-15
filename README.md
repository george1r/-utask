# Qwen3 alignment: SFT, Reward Model, DPO and SimPO

| Stage | Reproducible Colab |
|---|---|
| **All five tasks — one top-to-bottom run** | [![Open all tasks in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/alignment_all_tasks_qwen3_colab.ipynb) |
| Task 1 — SFT | [![Open Task 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/task1_sft_qwen3_colab.ipynb) |
| Task 2 — style DPO on top of SFT | [![Open Task 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/task2_dpo_style_qwen3_colab.ipynb) |

This repository contains a reproducible QLoRA workflow for all five alignment tasks using the exact
`Qwen/Qwen3-4B-Instruct-2507` model. The combined notebook performs SFT, style DPO, Reward Model
training, quality DPO, and reference-free SimPO in one runtime. Tasks 4 and 5 both start from the
same style-DPO checkpoint produced earlier in that run, so their comparison is controlled.

No notebook downloads a pre-trained adapter. The combined notebook trains every model/adapter in
the notebook and computes all five metrics; result values are not stored in advance.

## Repository contents

- `notebooks/alignment_all_tasks_qwen3_colab.ipynb` — the complete five-task submission notebook;
- `notebooks/task1_sft_qwen3_colab.ipynb` — end-to-end QLoRA SFT, generation, style scoring, and report;
- `notebooks/task2_dpo_style_qwen3_colab.ipynb` — the same SFT followed by style-preference DPO, generation, scoring, and report;
- `data/kid_adult.jsonl` — 1,489 training pairs;
- `data/good_bad.jsonl` — 2,226 quality-preference training pairs;
- `data/public_test_style.jsonl` — 50 held-out style-test rows, parsed only after training;
- `data/public_test_quality.jsonl` — 50 held-out quality-test pairs;
- `metrics/style_clf.pkl` — the supplied style classifier, loaded with scikit-learn 1.7.2.

The original ZIP, supplied ready-made `gold_rm`, model weights, checkpoints, adapters, caches, and
runtime outputs are intentionally excluded from Git. In particular, Task 3 trains its own reward
model instead of loading the supplied adapter.

## Shared reproducibility contract

The combined notebook:

1. Install pinned versions of Transformers, TRL, PEFT, Accelerate, bitsandbytes, Datasets,
   scikit-learn, SciPy, pandas, joblib, SentencePiece, and related packages. Colab's existing
   NumPy, PyTorch, and CUDA stack is preserved to avoid an in-process binary ABI mismatch.
2. Require a CUDA GPU and print the full software/GPU environment.
3. Fix Python, NumPy, PyTorch, CUDA, Transformers, trainer, and data seeds to 42.
4. Load the exact model at Hugging Face commit
   `cdbee75f17c01a7cc42f958dc650907174af0554` with NF4 4-bit double quantization and fp16 compute.
5. Train LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
6. Uses public data only after the corresponding training stage and never passes a public row to a
   trainer. Style evaluation uses 50 original prompts with `do_sample=False` and `num_beams=1`.
7. Trains the Reward Model with a scalar sequence-classification head and evaluates strict
   `reward(chosen) > reward(rejected)` accuracy on all 50 public quality pairs.
8. Computes Tasks 4 and 5 without generation: it sums answer-token log-probabilities and divides by
   the exact number of answer tokens before comparing chosen and rejected. Prompt tokens and the
   chat terminator are excluded. The diagnostic table also prints raw sums and token counts.
9. Runs pure SimPO with `loss_type="simpo"`, `cpo_alpha=0.0`, and no reference model or adapter.
10. Prints a final table and five machine-readable `FINAL_TASK*_METRIC` lines from values computed
    by the current run.

## Task 1 — SFT

The SFT dataset contains conversational `prompt`/`completion` examples with `kid` as the assistant
target. Completion-only loss masks the user prompt. A complete top-to-bottom T4 run reproduced:

```text
TASK1_NUM_GENERATIONS=50
TASK1_SEED=42
TASK1_DO_SAMPLE=False
TASK1_MEAN_P_SIMPLE=0.971761
TASK1_INTERVAL=0.9 <= P_simple <= 1.0
TASK1_ANSWER=Г
```

## Task 2 — DPO over the reproduced SFT adapter

Task 2 first repeats Task 1 SFT from the pinned base weights. It then saves that newly trained
adapter inside the current runtime and loads an identical copy as `reference`. Tensor equality is
asserted before DPO. The `default` policy adapter is trainable; the reference adapter supplies
reference log-probabilities without loading a second 4B model. DPO uses all 1,489 explicit
conversational preference pairs, one full epoch, sigmoid loss, and `beta=0.1`.

The final cell prints values computed during that run:

```text
TASK2_NUM_GENERATIONS=50
TASK2_SEED=42
TASK2_DO_SAMPLE=False
TASK2_MEAN_P_SIMPLE=0.995563
TASK2_INTERVAL=0.9 <= P_simple <= 1.0
TASK2_ANSWER=Г
```

The complete Task 2 Colab T4 run produced `P_simple = 0.995563`, which is in the `0.9–1.0`
interval (answer `Г`). This is an increase of `0.023802` over the reproduced Task 1 result.

## Task 3 — Reward Model

The causal policy is unloaded before Task 3. A fresh 4-bit sequence-classification model is loaded
from the pinned Qwen revision with one scalar `score` head. QLoRA weights and the score head are
trained on all 2,226 `good_bad` pairs with pairwise reward loss. The public pairwise accuracy uses
strict comparison; ties count as incorrect.

## Task 4 — quality DPO

The Reward Model is unloaded, then trainable policy and frozen reference adapters are both created
from the style-DPO checkpoint made earlier in the same runtime. Tensor equality is asserted before
training. The held-out implicit-preference metric compares mean answer-token log-probability, not
the length-biased raw sum.

## Task 5 — SimPO

The quality-DPO model is unloaded and a fresh policy is loaded from the same style-DPO checkpoint.
TRL's pure SimPO loss is used with `cpo_alpha=0.0`; the notebook asserts that only the `default`
policy adapter exists. Task 5 is evaluated by the exact same helper and public rows as Task 4, and
the final cell prints their accuracy delta.

## Run in Google Colab

Open the **All five tasks** notebook using the first badge, select a fresh **T4 GPU** runtime, and
run all cells from top to bottom without skipping or editing them. In a clean runtime it clones
`https://github.com/george1r/-utask.git` and needs no file upload or path change. The combined run is
substantially longer than Task 1 because it performs five training stages and explicitly unloads
one model before loading the next to stay within T4 memory.

Do not tune hyperparameters from the public-test result. Preserve the original commit history: do
not amend, rebase, force-push, or alter commit dates.

## Data integrity

The combined notebook asserts these SHA-256 values before using each artifact:

```text
52bacff1c6d5d50ca3dd473f8d754cf1dfcce7e02ecf162cda2c18719a138748  data/kid_adult.jsonl
bf50f3af0127df71d891c5a65eb75220104157f3e27b613aacbae1761c08998b  data/good_bad.jsonl
d0f5fb848245b18e97b97fe5158c602f3f2b49b8ec6588f93a0f0e9f10c58efe  data/public_test_style.jsonl
bc8b21bf04c88e99d420569c61f46309f71d04601159b80ea258760e8d871780  data/public_test_quality.jsonl
b5cf7b53417033de19b9c44a43402bce0e6eeece44b1abac2cf596785b60888d  metrics/style_clf.pkl
```

## Validation status

All three notebooks are valid notebook JSON and all code cells compile. The local classifier
sanity check gives mean `P_simple` values of approximately 0.974751 for reference `kid` answers and
0.018412 for reference `adult` answers. Every quality train/public sequence was checked with the
pinned tokenizer; all fit the notebook's 384-token quality limit (observed maximum: 363). The
complete Task 1 and Task 2 runs are recorded above. Tasks 3–5 still require their deliberately
in-notebook Colab T4 training runs; static validation cannot substitute for measured GPU results.
