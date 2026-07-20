# Experiment 3 – Novel Antibodies

**Split column:** `ab_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline comparison:** esm-up Experiment 3 (L2 Logistic Regression)

## Description

All pairs containing certain antibody identities are held out entirely for
testing, so the model must generalise to previously unseen antibody sequences.

## Contents

```
experiment_3_novel_antibodies/
├── data/
│   ├── train.csv      # Training split (ab_name, ag_name, label)
│   └── test.csv       # Test split
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
