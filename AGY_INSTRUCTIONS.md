# 🤖 AGY Master Execution Instructions (NVIDIA Rig)

This document provides complete, machine-actionable instructions for **AGY** (the Antigravity AI Agent) operating on the target **NVIDIA Rig** (AMD Ryzen 9 9900X / NVIDIA RTX PRO 4000 Blackwell 24 GB VRAM / 64 GB DDR5 RAM).

---

## 🎯 Goal & Objective

Execute the end-to-end **ESM-Mamba (MambaCross)** neural network training pipeline across all 4 HIV antibody–antigen neutralization prediction experiments, optimize execution for the local NVIDIA GPU, and record consolidated performance metrics.

---

## 📋 Step-by-Step Execution Plan for AGY

### Step 1: Repository Synchronization & Environment Setup

Run the following commands in order:

```cmd
:: 1. Navigate to repository root
cd path\to\esm-neu

:: 2. Pull latest code and optimizations
git pull origin main

:: 3. Create & activate virtual environment (if not already active)
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate

:: 4. Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Hardware & CUDA Verification

Run a quick PyTorch sanity check to verify CUDA and GPU access:

```cmd
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"
```

*Expected Output*: `CUDA Available: True`, Device: `NVIDIA RTX PRO 4000 Blackwell`, VRAM: `~24.0 GB`.

---

### Step 3: ESM-2 Sequence Embedding Verification (Phase A)

Check if pre-extracted ESM-2 sequence embeddings exist on disk under `Outputs/Pretrained_HIV/`:

```cmd
python -c "import os; ab=len(os.listdir('Outputs/Pretrained_HIV/ab')) if os.path.exists('Outputs/Pretrained_HIV/ab') else 0; ag=len(os.listdir('Outputs/Pretrained_HIV/ag')) if os.path.exists('Outputs/Pretrained_HIV/ag') else 0; print(f'Antibody embeddings: {ab}/665, Antigen embeddings: {ag}/2624')"
```

* **If embeddings exist (665 ab / 2624 ag)**: Proceed immediately to Step 4.
* **If embeddings are missing**: Run feature extraction:
  ```cmd
  python shared\Pretrained.py
  ```

---

### Step 4: Execute Neural Network Experiments (Phase B)

Choose **One** of the following execution modes based on your resource allocation preference:

#### 🚀 Mode 1: Fast Parallel Execution (Recommended for 24 GB VRAM)
Launches all 4 experiments concurrently using process pool workers:
```cmd
python run_all_experiments.py --parallel --epochs 30 --batch_size 32
```

#### 📜 Mode 2: Sequential Execution via One-Click Batch Script
Runs CUDA checks, embedding verification, and all 4 experiments sequentially:
```cmd
run_pipeline.bat
```

#### 🔬 Mode 3: Standalone Single Experiment Execution
Run any experiment independently inside its own folder:
```cmd
:: Experiment 1: Random Split
cd experiment_1_random && python train_nn.py --epochs 30 --batch_size 32

:: Experiment 2: Novel Viruses
cd experiment_2_novel_viruses && python train_nn.py --epochs 30 --batch_size 32

:: Experiment 3: Novel Antibodies
cd experiment_3_novel_antibodies && python train_nn.py --epochs 30 --batch_size 32

:: Experiment 4: Both Novel (Double Holdout)
cd experiment_4_both_novel && python train_nn.py --epochs 30 --batch_size 32
```

---

## ⚡ Built-in Performance & Hardware Optimizations

The codebase includes the following automatic optimizations tuned for the target rig:

1. **In-RAM Pre-caching**: `preload_embeddings()` loads all 665 antibodies & 2,624 antigens into system RAM (<50 MB) upon startup. Zero disk I/O overhead during training loops.
2. **Tensor Core Precision**: `torch.set_float32_matmul_precision('high')` activates Tensor Cores (TF32) on modern NVIDIA GPUs.
3. **cuDNN Autotuning**: `torch.backends.cudnn.benchmark = True` finds optimal CUDA kernels.
4. **Non-Blocking Transfers**: Host-to-device transfers use `.to(device, non_blocking=True)` and `pin_memory=True`.
5. **Mixed Precision**: `bfloat16` AMP (`torch.autocast`) runs natively on Blackwell Tensor Cores.

---

## 📊 Summary of Experiments & Expected Outputs

| # | Experiment Folder | Split Column | Holdout Constraint | Train $n$ | Test $n$ | Excluded $n$ |
|---|-------------------|--------------|-------------------|-----------|----------|--------------|
| 1 | `experiment_1_random` | `random` | None (interpolation) | 59,799 | 14,931 | 0 |
| 2 | `experiment_2_novel_viruses` | `vir_block` | 541 viruses held out | 61,219 | 13,511 | 0 |
| 3 | `experiment_3_novel_antibodies` | `ab_block` | 137 antibodies held out | 57,903 | 16,827 | 0 |
| 4 | `experiment_4_both_novel` | `both_block` | 232 Abs & 749 Vir held out | 34,774 | 7,306 | 32,650 |

### Output Validation Checklist

Upon completion, verify that the following artifacts exist:

- [ ] `experiment_1_random/results/results.json` & `best_model.pt`
- [ ] `experiment_2_novel_viruses/results/results.json` & `best_model.pt`
- [ ] `experiment_3_novel_antibodies/results/results.json` & `best_model.pt`
- [ ] `experiment_4_both_novel/results/results.json` & `best_model.pt`
- [ ] Top-level consolidated CSV: `nn_summary_results.csv`

---

## 🛠️ Troubleshooting Guide for AGY

* **CUDA Out of Memory (OOM)**: Reduce batch size by passing `--batch_size 16`.
* **Missing Module Errors**: Re-run `pip install -r requirements.txt`.
* **Data File Missing**: Verify `Data/HIV/ab_ag_pair.csv`, `antibody.csv`, and `antigen.csv` are intact in the project directory.
