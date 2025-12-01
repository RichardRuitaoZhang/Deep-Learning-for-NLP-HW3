CS461: Deep Learning for NLP — HW3
Parameter-Efficient Fine-Tuning (PEFT): LoRA + Prefix-Tuning
This repository contains the full implementation, environment configuration, and SLURM wrapper scripts for Homework 3 of CS461: Deep Learning for Natural Language Processing (Fall 2025).
The assignment implements two PEFT techniques:
LoRA (Low-Rank Adaptation)
Prefix-Tuning (Key/Value prefix injection)
This repository includes all necessary code, except for large pretrained files, which must be obtained from HW2.
Directory Structure
HW3/
├── __pycache__/                # Python cache
├── logs/                       # Training logs (LoRA & Prefix runs)
├── scripts/                    # GPU/SLURM wrapper scripts
│   ├── run_lora_gpu.sh         # SLURM wrapper for LoRA training
│   └── run_prefix_gpu.sh       # SLURM wrapper for Prefix-Tuning
├── .gitignore                  # Ignores data/ and pretrain/ large files
├── peft.yml                    # Cond
