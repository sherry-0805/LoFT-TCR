# LoFT-TCR

This repository provides the source code for **LoFT-TCR: A LoRA-based Fine-tuning Framework for TCR–Antigen Binding Prediction**. It includes code and conda environments for:

- Fine-tuning the ESM-2 protein language model with **LoRA (Low-Rank Adaptation)**
- Performing downstream **TCR–antigen binding prediction** using **graph learning**

---

## Environment Setup

### 1. Fine-tuning ESM-2 with LoRA

```bash
conda env create -f finetune.yml
conda activate finetune
```
### 2. Full LoFT-TCR pipeline
```bash
conda env create -f dgl_env.yml
conda activate dgl_env
```
## Guided Tutorial
To execute the full training and evaluation workflow, use:
```bash
sbatch commands.sh
```
The ```commands.sh``` script includes steps for fine-tuning, embedding extraction, graph construction, and model evaluation.
