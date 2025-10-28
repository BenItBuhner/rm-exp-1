# Athena v2 – Recursive Reasoning Base Model

Athena v2 is a refactor of the Tiny Recursive Model (TRM) pipeline geared toward building a **~250‑330 M parameter iterative base model**. The code mirrors TRM’s project layout so existing tooling, configs, and training know‑how transfer directly, but it adds:

* **Scaled architecture presets** (4–5 L layers, 6–8 recursive cycles, 2048+ width, optional reasoning scratchpads, post‑iteration refinement steps).
* **General sequence corpus builders** that turn JSONL prompts/responses into the fixed `inputs`/`labels` tensors the trainer expects, while attaching per‑example iteration budgets for curricula.
* **Extended dataset loader** that can mix multiple corpora, enforce minimum iteration counts, and expose curriculum metadata to the model.
* **Reusable train loop** with EMA, cosine LR, SignSGD puzzle embeddings, optional EMA swap at eval, Hydra configs, and pluggable evaluators (now including text metrics).
* **Multi-device support** for single/multi-GPU (CUDA), single/multi-TPU (PyTorch XLA), and CPU training with automatic detection.

> This repository intentionally keeps file names/function signatures close to TRM for easier diffing and experimentation. The top‑level directory is scoped to "Athena 1 Recursive Preparing/v2" so future iterations can live alongside it without conflicts.

## 🚀 Hardware Support

Athena v2 supports seamless training on:
- **Single CUDA GPU** (e.g., RTX 4060, A100)
- **Multi-GPU** via `torchrun` (NCCL)
- **Single TPU core** (v2/v3/v4)
- **Multi-TPU** (up to 8 cores, automatic spawn)
- **CPU fallback** for testing

See [TPU_SETUP.md](TPU_SETUP.md) for detailed TPU configuration and usage.

---

## Directory Map

```
v2/
├── README.md
├── requirements.txt
├── pretrain.py
├── config/
│   ├── cfg_pretrain.yaml
│   └── arch/
│       └── athena_trm_large.yaml
├── dataset/
│   ├── build_text_dataset.py
│   ├── tokenizer_utils.py
│   └── common.py
├── sequence_dataset.py
├── models/
│   ├── common.py
│   ├── layers.py
│   ├── sparse_embedding.py
│   ├── ema.py
│   ├── losses.py
│   └── recursive_reasoning/
│       └── athena_trm.py
├── evaluators/
│   └── text.py
└── utils/
    └── functions.py
```

---

## Quick Start

1. **Install deps**

   ```bash
   pip install --upgrade pip wheel setuptools
   pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```

2. **Prepare a base-model corpus**

   Point the builder at arbitrarily large raw text/code dumps (plain text or JSONL). The script tokenizes, chunks into fixed-length blocks, auto-derives iteration budgets, and writes the `inputs/labels` tensors.

   ```bash
   python -m dataset.build_corpus_dataset \
     --tokenizer-json path/to/tokenizer.json \
     --input-files data/raw/c4.jsonl.zst data/raw/github.txt.gz \
     --output-dir data/athena_base \
     --seq-len 2048 \
     --stride 2048 \
     --format jsonl \
     --text-key text
   ```

   (Optional) For later SFT or supervised curricula you can still use `dataset.build_text_dataset`, which expects explicit `{prompt,response}` pairs and writes them in the same format, so both pipelines are compatible with the trainer.

3. **Launch pretraining**

   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   torchrun --nproc-per-node 4 pretrain.py \
     data_paths="[data/athena_base]" \
     arch=athena_trm_large \
     global_batch_size=768 \
     +run_name=athena_v1_large
   ```

   Hydra overrides let you swap datasets, iteration curricula, or architecture knobs without editing code.

4. **Evaluate**

   Add `evaluators=[text@PerplexityEvaluator]` (see `evaluators/text.py`) or plug in custom ones. The trainer saves predictions per eval step and checkpoints under `checkpoints/<project>/<run_name>/`.

---

## Goals & Key Changes

### Scaled Model
* Hidden size configurable up to 2304, 32–40 heads, SwiGLU expansion 4–6×.
* Recursive depth bumped to 6–8 outer cycles with per-example iteration targets (`iteration_targets`) fed through the dataset.
* Optional reasoning scratchpad stream: each iteration can emit a short chain‑of‑thought that is **not** fed to the loss but can be logged or later distilled.
* Test‑time compute hooks: after halting, a configurable number of refinement passes polish the final answer.

### Data & Iteration Curricula
* Builders attach `iteration_targets` (min/max) per example so ACT learns when to spend more compute.
* Loader can enforce domain‑specific sampling weights and ensures the trainer sees `iteration_targets` and `domain_ids` alongside sequences.

### Future Work Hooks
* SFT/RL phases can reuse the same config + training script by swapping `data_paths`, enabling KL loss, or plugging evaluators that query reward models.
* The trainer logs per-step histograms (accuracy, exact match, halting accuracy, mean steps) broken down by split so you can measure the impact of longer budgets.

Refer to inline comments for implementation details. The default configs should run end-to-end once you provide a tokenizer and JSONL corpus.
### Colab smoke test

To try everything inside Google Colab without cloning this repo, upload `notebooks/athena_all_in_one_colab.ipynb` and run all cells. The notebook rebuilds the same project structure on the Colab VM, downloads a small Hugging Face dataset, and can perform a quick CUDA-backed training sanity check.
