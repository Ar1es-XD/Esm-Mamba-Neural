# Experiment 2 – Novel Viruses

**Split column:** `vir_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline comparison:** esm-up Experiment 2 (L2 Logistic Regression)

## Description

All pairs containing certain virus (antigen) identities are held out entirely
for testing, so the model must generalise to previously unseen viral sequences.

## Contents

```
experiment_2_novel_viruses/
├── data/
│   ├── train.csv      # Training split (ab_name, ag_name, label)
│   └── test.csv       # Test split
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
