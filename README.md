```markdown
# CS461: Deep Learning for NLP — HW3
**Parameter-Efficient Fine-Tuning (PEFT): LoRA + Prefix-Tuning**

This repository contains the full implementation, environment configuration, and SLURM GPU wrappers for Homework 3 of CS461 (Fall 2025).  
The assignment focuses on parameter-efficient fine-tuning of a GPT-style model using:

1. **LoRA** (Low-Rank Adaptation)  
2. **Prefix-Tuning** (Key/Value prefix injection)

Large pretrained files and datasets are **not tracked** in this repository.  
They must be obtained from **Canvas** or copied from your **HW2 repository** before running experiments.

---

## Directory Structure

```

HW3/
├── **pycache**/                   # Python cache files
├── logs/                          # Training & evaluation logs
├── scripts/                       # GPU / SLURM wrapper scripts
│   ├── run_lora_gpu.sh            # LoRA training job
│   └── run_prefix_gpu.sh          # Prefix-Tuning training job
├── .gitignore                     # Excludes large folders (data/, pretrain/)
├── peft.yml                       # Conda environment for HW3
├── peft_adapter.py                # Shared bottleneck adapter module
├── peft_lora.py                   # LoRA implementation and trainer
├── peft_prefix(abandoned).py      # Earlier prefix-tuning attempt (not used)
└── starter.py                     # Original TransformerGPT definition from HW2

```

---

## Required External Folders (Not Included)

The following directories are intentionally excluded via `.gitignore`:

- `data/`  
  - `obqa.train.txt`  
  - `obqa.valid.txt`  
  - `obqa.test.txt`  
  - `wiki.train.txt`  
  - `wiki.valid.txt`  
  - `wiki.test.txt`

- `pretrain/`  
  - `model_weights` (pretrained GPT-style backbone used in HW2)

These must be copied manually from Canvas or from your HW2 directory, e.g.:

```

cp -r /path/to/HW2/data      ./data
cp -r /path/to/HW2/pretrain  ./pretrain

````

---

## Environment Setup

Create and activate the HW3 environment:

```bash
conda env create -f peft.yml
conda activate peft
````

Key packages (already specified in `peft.yml`) include:

* PyTorch (GPU-enabled)
* HuggingFace Transformers
* datasets / tokenizers
* numpy, pandas, tqdm

Ensure that the active machine has CUDA-compatible GPUs.

---

## Running LoRA Fine-Tuning

LoRA is implemented in `peft_lora.py`. It injects low-rank matrices into attention projection layers of the GPT-style model.

Submit a GPU job via SLURM:

```bash
sbatch scripts/run_lora_gpu.sh
```

Logs will be stored under `logs/`.

---

## Running Prefix-Tuning

Prefix-Tuning prepends trainable K/V prefixes to the GPT attention mechanism.

Submit a GPU job via:

```bash
sbatch scripts/run_prefix_gpu.sh
```

The file `peft_prefix(abandoned).py` is an archived earlier attempt and not used.

---

## Starter Model

`starter.py` contains the unmodified HW2 TransformerGPT backbone used by both LoRA and Prefix-Tuning.
All PEFT modules wrap or extend this base architecture.

---

## Logs

All experiment logs are saved under `logs/` and include:

* Training loss over steps/epochs
* Validation accuracy / exact-match
* Hyperparameters for each run

---

## Notes

* This repository is code-only: large inputs and pretrained weights are external.
* Scripts assume the presence of local `data/` and `pretrain/` directories.
* The environment `peft` mirrors the HW2 `transformer` environment but is declared separately for clarity.

---

**Author:** Ruitao Zhang
**Course:** CS461 — Deep Learning for NLP, Fall 2025
**Instructor:** Prof. David Demeter

```
```
