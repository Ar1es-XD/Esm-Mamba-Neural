# ESM-Mamba (esm-neu) — End-to-End Neural Network Experiments

Recreates the four HIV antibody–antigen neutralisation prediction experiments
from **esm-up** (L2 Logistic Regression baseline) using the **ESM-Mamba
(MambaCross)** end-to-end neural network architecture.

## Hardware & Environment (Target Rig)

| Component       | Details                                     |
|-----------------|---------------------------------------------|
| CPU             | AMD Ryzen 9 9900X                           |
| GPU (Primary)   | NVIDIA RTX PRO 4000 Blackwell (24 GB VRAM)  |
| GPU (Secondary) | AMD Radeon (2 GB) — not used for training   |
| RAM             | 64 GB DDR5                                  |
| OS              | Windows                                     |
| PyTorch Backend | `cuda` (NVIDIA CUDA)                        |
| Mixed Precision | bfloat16 AMP (autocast + GradScaler)        |

## Project Structure

```
esm-neu/
├── shared/                          # Core reusable modules
│   ├── Models.py                    #   MambaCross neural network
│   ├── Toolkit.py                   #   Utilities (metrics, seeding, dataset)
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
│   └── ag/                          #   2624 antigen .npy files
│
├── experiment_1_random/             # Experiment 1: Random split
│   ├── data/{train,test}.csv
│   ├── results/{results.json, best_model.pt}
│   └── README.md
│
├── experiment_2_novel_viruses/      # Experiment 2: Virus-blocked split
│   ├── data/{train,test}.csv
│   ├── results/{results.json, best_model.pt}
│   └── README.md
│
├── experiment_3_novel_antibodies/   # Experiment 3: Antibody-blocked split
│   ├── data/{train,test}.csv
│   ├── results/{results.json, best_model.pt}
│   └── README.md
│
├── experiment_4_both_novel/         # Experiment 4: Double holdout
│   ├── data/{train,test}.csv
│   ├── results/{results.json, best_model.pt}
│   └── README.md
│
├── train_experiment.py              # Train a single experiment
├── run_all_experiments.py           # Master pipeline (all 4 experiments)
├── run_pipeline.bat                 # One-click Windows launcher for the rig
├── nn_summary_results.csv           # Consolidated results (generated)
└── requirements.txt                 # Python dependencies
```

## Quick Start (on the rig)

```batch
REM 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

REM 2. Run everything (CUDA check → embeddings → all 4 experiments)
run_pipeline.bat
```

Or run experiments individually:

```batch
REM Single experiment
python train_experiment.py --split_col ab_block --epochs 30

REM All experiments
python run_all_experiments.py
```

## Experiments

| # | Name               | Split Column | Description                              |
|---|--------------------|-------------|------------------------------------------|
| 1 | Random Split       | `random`     | Random train/test assignment             |
| 2 | Novel Viruses      | `vir_block`  | Test viruses unseen during training      |
| 3 | Novel Antibodies   | `ab_block`   | Test antibodies unseen during training   |
| 4 | Both Novel         | `both_block` | Both antibodies and viruses are novel    |

## Methodology

Unlike **esm-up** (which freezes an untrained MambaCross model as a random
projection encoder and fits Logistic Regression on cached representations),
**esm-neu** trains the full MambaCross architecture **end-to-end** with Binary
Cross Entropy loss. The bilinear projection matrix and 2D VMamba sequence
sweeps are updated dynamically via backpropagation alongside the MLP decoder.

Training runs for 30 epochs with bfloat16 mixed-precision on CUDA; the
checkpoint with the highest test AUROC is saved as the best model for each
experiment.
