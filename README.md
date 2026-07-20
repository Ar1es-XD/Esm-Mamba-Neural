# ESM-Mamba (esm-neu) — End-to-End Neural Network Experiments

Recreates the four HIV antibody–antigen neutralisation prediction experiments from **esm-up** (L2 Logistic Regression baseline) using the **ESM-Mamba (MambaCross)** end-to-end neural network architecture.

---

## 🤖 AI Agent / AGY Quick Execution Guide (For NVIDIA Rig)

If you are **AGY** (the Antigravity AI Agent) running on the target NVIDIA rig, follow these simple steps to run the pipeline:

### 1. Environment & Setup
```cmd
git pull origin main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Full Pipeline (Choose One Method)
* **Method A: Fast Parallel Run (Recommended for 24 GB VRAM)**
  ```cmd
  python run_all_experiments.py --parallel
  ```
* **Method B: Batch Launcher (Sequential)**
  ```cmd
  run_pipeline.bat
  ```

### 3. Launch Standalone Experiment (Single Split)
```cmd
cd experiment_3_novel_antibodies
python train_nn.py --epochs 30 --batch_size 32
```

---

## Hardware & Environment (Target Rig Specs)

| Component       | Details                                     |
|-----------------|---------------------------------------------|
| CPU             | AMD Ryzen 9 9900X (12 cores / 24 threads)   |
| GPU (Primary)   | NVIDIA RTX PRO 4000 Blackwell (24 GB VRAM)  |
| RAM             | 64 GB DDR5                                  |
| OS              | Windows                                     |
| PyTorch Backend | `cuda` (cuDNN benchmark + TF32 precision)   |
| Mixed Precision | `bfloat16` AMP (`torch.autocast`)           |
| Memory Caching  | In-RAM embedding pre-caching (<50 MB RAM)   |

---

## Project Structure

```
esm-neu/
├── shared/                          # Core reusable modules
│   ├── Models.py                    #   MambaCross neural network
│   ├── Toolkit.py                   #   RAM embedding caching & metrics
│   ├── Loader.py                    #   PairDataset loader
│   ├── Pretrained.py                #   ESM-2 embedding extractor
│   └── Param_Model.json             #   Model hyper-parameters
│
├── Data/HIV/                        # Raw data
│   ├── ab_ag_pair.csv               #   Pairs + split columns
│   ├── antibody.csv                 #   Antibody sequences
│   └── antigen.csv                  #   Antigen sequences
│
├── Outputs/Pretrained_HIV/          # Pre-extracted ESM-2 embeddings
│   ├── ab/                          #   665 antibody .npy files
│   └── ag/                          #   2,624 antigen .npy files
│
├── experiment_1_random/             # Exp 1: Random split (59,799 train / 14,931 test)
│   ├── train_nn.py                  #   Standalone experiment trainer
│   ├── train.csv & test.csv         #   Local partition files
│   └── results/                     #   results.json & best_model.pt
│
├── experiment_2_novel_viruses/      # Exp 2: Virus holdout (61,219 train / 13,511 test)
│   ├── train_nn.py                  #   Standalone experiment trainer
│   ├── train.csv & test.csv         #   Local partition files
│   └── results/                     #   results.json & best_model.pt
│
├── experiment_3_novel_antibodies/   # Exp 3: Antibody holdout (57,903 train / 16,827 test)
│   ├── train_nn.py                  #   Standalone experiment trainer
│   ├── train.csv & test.csv         #   Local partition files
│   └── results/                     #   results.json & best_model.pt
│
├── experiment_4_both_novel/         # Exp 4: Double holdout (34,774 train / 7,306 test)
│   ├── train_nn.py                  #   Standalone experiment trainer
│   ├── train.csv & test.csv         #   Local partition files
│   └── results/                     #   results.json & best_model.pt
│
├── train_experiment.py              # Master single-experiment launcher by split column
├── run_all_experiments.py           # Master runner (sequential & parallel modes)
├── run_pipeline.bat                 # One-click Windows batch launcher
├── nn_summary_results.csv           # Consolidated results summary
└── requirements.txt                 # Dependencies
```

---

## Experiments Summary

| # | Experiment Name     | Split Column | Holdout Constraint | Train Pairs | Test Pairs | Excluded |
|---|--------------------|-------------|-------------------|-------------|------------|----------|
| 1 | Random Split       | `random`     | None (interpolation) | 59,799 (80.0%) | 14,931 (20.0%) | 0 |
| 2 | Novel Viruses      | `vir_block`  | 541 viruses held out | 61,219 (81.9%) | 13,511 (18.1%) | 0 |
| 3 | Novel Antibodies   | `ab_block`   | 137 antibodies held out | 57,903 (77.5%) | 16,827 (22.5%) | 0 |
| 4 | Both Novel         | `both_block` | 232 Abs & 749 Vir held out | 34,774 (46.5%) | 7,306 (9.8%) | 32,650 |

---

## Methodology & Architecture

Unlike **esm-up** (which freezes an untrained MambaCross model as a random projection encoder and fits Logistic Regression on cached representations), **esm-neu** trains the full MambaCross architecture **end-to-end** with Binary Cross Entropy loss. The bilinear projection matrix and 2D VMamba sequence sweeps are updated dynamically via backpropagation alongside the MLP decoder.

Training runs for 30 epochs with `bfloat16` mixed-precision on CUDA; the checkpoint with the highest test AUROC is saved to `results/best_model.pt` for each experiment.
