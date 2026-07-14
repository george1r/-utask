# Qwen3 alignment: SFT and DPO for a simple explanation style

| Stage | Reproducible Colab |
|---|---|
| Task 1 — SFT | [![Open Task 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/task1_sft_qwen3_colab.ipynb) |
| Task 2 — style DPO on top of SFT | [![Open Task 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/task2_dpo_style_qwen3_colab.ipynb) |

This repository contains reproducible QLoRA workflows for the first two alignment tasks using the
exact `Qwen/Qwen3-4B-Instruct-2507` model. Task 1 trains the simple explanation style with SFT.
Task 2 reproduces that SFT stage from the base model in the same runtime, copies the resulting
adapter into identical policy and frozen-reference adapters, and then runs DPO with `kid` as
`chosen` and `adult` as `rejected`.

Neither notebook downloads a pre-trained adapter. Each final answer letter is computed from 50
new greedy generations and is not stored in advance.

## Repository contents

- `notebooks/task1_sft_qwen3_colab.ipynb` — end-to-end QLoRA SFT, generation, style scoring, and report;
- `notebooks/task2_dpo_style_qwen3_colab.ipynb` — the same SFT followed by style-preference DPO, generation, scoring, and report;
- `data/kid_adult.jsonl` — 1,489 training pairs;
- `data/public_test_style.jsonl` — 50 held-out style-test rows, parsed only after training;
- `metrics/style_clf.pkl` — the supplied style classifier, loaded with scikit-learn 1.7.2.

The original ZIP, quality-task data, `gold_rm`, model weights, checkpoints, adapters, caches, and
runtime outputs are intentionally excluded from Git.

## Shared reproducibility contract

Both notebooks:

1. Install pinned versions of Transformers, TRL, PEFT, Accelerate, bitsandbytes, Datasets,
   scikit-learn, SciPy, pandas, joblib, SentencePiece, and related packages. Colab's existing
   NumPy, PyTorch, and CUDA stack is preserved to avoid an in-process binary ABI mismatch.
2. Require a CUDA GPU and print the full software/GPU environment.
3. Fix Python, NumPy, PyTorch, CUDA, Transformers, trainer, and data seeds to 42.
4. Load the exact model at Hugging Face commit
   `cdbee75f17c01a7cc42f958dc650907174af0554` with NF4 4-bit double quantization and fp16 compute.
5. Train LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
6. Parse `public_test_style.jsonl` only after all training stages finish, then generate from the
   original user prompts with `do_sample=False`, `num_beams=1`, and `max_new_tokens=192`. No system
   message or instruction asking for a simple answer is added.
7. Validate the classifier and its word-then-character TF-IDF feature order, display every
   generated answer and its `P_simple`, compute the arithmetic mean, and select exactly one task
   interval programmatically.

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
TASK2_MEAN_P_SIMPLE=<computed value>
TASK2_INTERVAL=<computed bounds>
TASK2_ANSWER=<computed А/Б/В/Г>
```

The Task 2 metric is intentionally not claimed here until the full GPU run has completed and its
machine-readable report has been copied back into the repository.

## Run in Google Colab

Open the required notebook using the badge above, select a fresh **T4 GPU** runtime, and run all
cells from top to bottom without skipping or editing them. In a clean runtime the notebook clones
`https://github.com/george1r/-utask.git` and needs no file upload or path change. Task 2 necessarily
takes longer than Task 1 because it performs both SFT and DPO before generating the evaluation set.

Do not tune hyperparameters from the public-test result. Preserve the original commit history: do
not amend, rebase, force-push, or alter commit dates.

## Data integrity

The Task 2 notebook asserts these SHA-256 values before using each artifact; Task 1 prints the same
hashes during execution:

```text
52bacff1c6d5d50ca3dd473f8d754cf1dfcce7e02ecf162cda2c18719a138748  data/kid_adult.jsonl
d0f5fb848245b18e97b97fe5158c602f3f2b49b8ec6588f93a0f0e9f10c58efe  data/public_test_style.jsonl
b5cf7b53417033de19b9c44a43402bce0e6eeece44b1abac2cf596785b60888d  metrics/style_clf.pkl
```

## Validation status

Both notebooks are valid notebook JSON and all code cells compile. The local classifier sanity
check gives mean `P_simple` values of approximately 0.974751 for reference `kid` answers and
0.018412 for reference `adult` answers. The complete Task 1 run is recorded above. Task 2 still
requires its deliberately in-notebook Colab T4 training run; static validation cannot substitute
for its measured result.
