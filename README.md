# CS461: Deep Learning for NLP — HW3  
**Parameter-Efficient Fine-Tuning (PEFT): LoRA + Adapter**

This repository contains the full code implementation, environment configuration, and SLURM GPU wrappers for **Homework 3** of CS461 (Fall 2025).  
The assignment focuses on parameter-efficient fine-tuning of a GPT-style model using:

1. **LoRA** (Low-Rank Adaptation)  
2. **Adapter Tuning** (bottleneck adapter inserted into each Transformer block)

> **Prefix-Tuning was explored but ultimately abandoned.**  
Only the working LoRA + Adapter implementations are part of this HW3 submission.

Large pretrained files and datasets are intentionally excluded from version control (see `.gitignore`).  
They must be copied from **Canvas** or from your **HW2 repository** before running experiments.

---

## Directory Structure

```text
HW3/
├── __pycache__/                     # Python cache files
├── logs/                            # Training & evaluation logs
├── scripts/                         # GPU / SLURM wrapper scripts
│   ├── run_lora_gpu.sh              # SLURM wrapper for LoRA training
│   └── run_adapter_gpu.sh           # SLURM wrapper for Adapter training
├── .gitignore                       # Excludes data/ and pretrain/ large folders
├── peft.yml                         # Conda environment for HW3
├── peft_adapter.py                  # Bottleneck adapter module + training script
├── peft_lora.py                     # LoRA implementation + training script
├── peft_prefix(abandoned).py        # Early prefix-tuning attempt (not used)
└── starter.py                       # Original TransformerGPT implementation from HW2
```
---

## Required External Folders (NOT included in this repo)

These folders are **ignored by design** (.gitignore):

### **1. `data/`** — OBQA + WIKI datasets

Contains:

* `obqa.train.txt`
* `obqa.valid.txt`
* `obqa.test.txt`
* `wiki.train.txt`
* `wiki.valid.txt`
* `wiki.test.txt`

### **2. `pretrain/`** — pretrained transformer weights

Contains:

* `model_weights` (≈560MB)

You must copy them manually:

```bash
cp -r ../HW2/data ./data
cp -r ../HW2/pretrain ./pretrain
```

Or download again from Canvas.

---

## Environment Setup

Create and activate the HW3 environment:

```bash
conda env create -f peft.yml
conda activate peft
```

Key packages included inside the YAML:

* **torch 2.8.0** (CUDA 12)
* numpy / pandas
* transformers
* datasets / evaluate / accelerate
* tokenizers
* tqdm

---

## Running Experiments

All training should be launched via SLURM wrappers in `scripts/`.

### **LoRA Training**

```bash
sbatch scripts/run_lora_gpu.sh
```

This calls:

* `peft_lora.py`
* Injects LoRA into Q/K/V layers
* Performs last-token prediction for HW3 classification task

---

### **Adapter Training**

```bash
sbatch scripts/run_adapter_gpu.sh
```

This calls:

* `peft_adapter.py`
* Adds bottleneck adapter (down → nonlinearity → up) into each Transformer block

---

## Notes

* **Prefix-tuning was abandoned** due to architectural complexity and masking issues.
* All HF GPT-style components come from the original **HW2 `starter.py`**, untouched.
* Large files are **not** tracked — use HW2 files or Canvas downloads.
* This repo contains **only lightweight code**, ready for submission.

---

**Author:** Ruitao Zhang  
**Course:** CS461 — Deep Learning for NLP, Fall 2025  
**Instructor:** Prof. David Demeter  
