# Experiment 4 – Both Novel (Double Holdout)

**Split column:** `both_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline comparison:** esm-up Experiment 4 (L2 Logistic Regression)

## Description

Both the antibody and virus identities in the test set are unseen during
training. Pairs labelled `'excl'` (single-novel overlaps) are excluded to
prevent feature leakage. This is the hardest generalisation setting.

## Contents

```
experiment_4_both_novel/
├── data/
│   ├── train.csv      # Training split (ab_name, ag_name, label)
│   └── test.csv       # Test split
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
