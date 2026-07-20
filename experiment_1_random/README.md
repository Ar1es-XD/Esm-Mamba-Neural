# Experiment 1 – Random Split

**Split column:** `random`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline comparison:** esm-up Experiment 1 (L2 Logistic Regression)

## Description

Antibody–antigen pairs are randomly assigned to train/test sets with no
constraint on antibody or virus identity overlap. This represents the easiest
generalisation setting.

## Contents

```
experiment_1_random/
├── data/
│   ├── train.csv      # Training split (ab_name, ag_name, label)
│   └── test.csv       # Test split
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
