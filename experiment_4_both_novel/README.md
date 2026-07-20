# Experiment 4 – Both Novel (Double Holdout)

**Split Column:** `both_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline Comparison:** esm-up Experiment 4 (L2 Logistic Regression)

---

## 🤖 Standalone AGY Execution Command

To run this experiment independently as AGY:
```cmd
python train_nn.py --epochs 30 --batch_size 32
```

---

## Description & Partitioning

Both the antibody and virus identities in the test set are unseen during training. Pairs where only one component is novel (**32,650 pairs**, labeled `'excl'`) are excluded to eliminate single-component feature leakage. This represents the hardest extrapolation setting.

* **Train set ($n$)**: 34,774 pairs (46.5%) | 59.49% neutralizing
* **Test set ($n$)**: 7,306 pairs (9.8%) | 56.78% neutralizing
* **Excluded pairs ($n$)**: 32,650 pairs (43.7%)
* **Held-out Entities**: 232 antibodies and 749 viruses completely excluded from training.

---

## Contents

```
experiment_4_both_novel/
├── train_nn.py     # Standalone neural network training script
├── train.csv       # Training split (34,774 rows: antibody_id, virus_id, neut)
├── test.csv        # Test split (7,306 rows: antibody_id, virus_id, neut)
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
