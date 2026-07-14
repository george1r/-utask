# Task 1: Qwen3 SFT for a simple explanation style

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/george1r/-utask/blob/main/notebooks/task1_sft_qwen3_colab.ipynb)

This repository contains a reproducible solution for **task 1 only**: supervised fine-tuning of
`Qwen/Qwen3-4B-Instruct-2507` on the simple (`kid`) responses from `kid_adult.jsonl`, followed by
greedy generation on all 50 held-out prompts and measurement with the supplied `style_clf.pkl`.

No answer letter or model metric is stored in advance. The last notebook cell computes both from
the outputs of the adapter trained during the same run.

## Repository contents

- `notebooks/task1_sft_qwen3_colab.ipynb` — end-to-end QLoRA SFT, generation, scoring, and report;
- `data/kid_adult.jsonl` — 1,489 training examples;
- `data/public_test_style.jsonl` — 50 held-out style-test examples, opened only after training;
- `metrics/style_clf.pkl` — trusted task style classifier serialized with scikit-learn 1.7.2.

The original ZIP, quality-task data, `gold_rm`, model weights, checkpoints, adapters, caches, and
runtime outputs are intentionally excluded from Git.

## What the notebook does

1. Installs pinned versions of Transformers, TRL, PEFT, Accelerate, bitsandbytes, Datasets,
   `scikit-learn==1.7.2`, SciPy, pandas, joblib, SentencePiece, and related packages. It does not
   replace Colab's PyTorch or CUDA installation.
2. Requires a CUDA GPU and prints the complete software/GPU environment.
3. Fixes Python, NumPy, PyTorch, CUDA, Transformers, trainer, and data seeds to 42.
4. Validates the 1,489 training rows and constructs only conversational `prompt`/`completion`
   examples: user=`prompt`, assistant=`kid`. `adult` is not retained in `train_dataset`.
5. Loads the exact model at Hugging Face commit
   `cdbee75f17c01a7cc42f958dc650907174af0554` with NF4 4-bit double quantization and fp16 compute.
6. Trains LoRA adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
   `down_proj`, with completion-only loss and the official Qwen chat template.
7. Saves only the newly trained adapter to temporary runtime storage.
8. Opens the 50-row public test after training and generates from each original prompt with
   `do_sample=False`, `num_beams=1`, and `max_new_tokens=192`; no system prompt or simplification
   instruction is used.
9. Validates the classifier structure and checks the word-then-character TF-IDF feature order on
   reference `kid` versus `adult` answers.
10. Displays all 50 generated answers and `P_simple` values, computes their arithmetic mean, and
    selects exactly one interval programmatically.

The final cell prints:

```text
TASK1_NUM_GENERATIONS=50
TASK1_SEED=42
TASK1_DO_SAMPLE=False
TASK1_MEAN_P_SIMPLE=<computed value>
TASK1_INTERVAL=<computed bounds>
TASK1_ANSWER=<computed А/Б/В/Г>
```

## Run in Google Colab

Use a fresh **T4 GPU** runtime and run every cell from top to bottom without skipping cells. Do not
tune hyperparameters from the public-test result.

Open the notebook with the badge above. In a clean runtime it clones the public repository from
`https://github.com/george1r/-utask.git`, verifies that all required data and metric files exist,
and then runs without path edits. The original commit history is preserved; do not amend, rebase,
force-push, or alter commit dates.

## Data integrity

The notebook prints hashes during execution. The committed source files have these SHA-256 values:

```text
52bacff1c6d5d50ca3dd473f8d754cf1dfcce7e02ecf162cda2c18719a138748  data/kid_adult.jsonl
d0f5fb848245b18e97b97fe5158c602f3f2b49b8ec6588f93a0f0e9f10c58efe  data/public_test_style.jsonl
b5cf7b53417033de19b9c44a43402bce0e6eeece44b1abac2cf596785b60888d  metrics/style_clf.pkl
```

## Verification status

The notebook is schema-valid, all code cells compile, and the data/classifier checks have been run
locally with `scikit-learn==1.7.2`. The classifier sanity means are approximately 0.974751 for the
reference `kid` answers and 0.018412 for the reference `adult` answers.

This machine has no NVIDIA CUDA/T4 runtime, so the real QLoRA training and 50 generations have not
been executed here. Therefore no task letter is claimed as reproduced yet. It becomes confirmed only
after the last cell completes in a real Colab T4 run and prints `TASK1_MEAN_P_SIMPLE` and
`TASK1_ANSWER` from the newly trained model.
