# Experiment 1 – Random Split

**Split Column:** `random`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline Comparison:** esm-up Experiment 1 (L2 Logistic Regression)

---

## 🤖 Standalone AGY Execution Command

To run this experiment independently as AGY:
```cmd
python train_nn.py --epochs 30 --batch_size 32
```

---

## Description & Partitioning

Antibody–antigen pairs are randomly assigned to train/test sets with no constraint on entity identity overlap. This represents the easiest generalisation setting (interpolation baseline).

* **Train set ($n$)**: 59,799 pairs (80.0%) | 58.54% neutralizing
* **Test set ($n$)**: 14,931 pairs (20.0%) | 58.88% neutralizing
* **Holdout constraints**: None (antibodies and antigens overlap between train and test in different pair combinations).

---

## Contents

```
experiment_1_random/
├── train_nn.py     # Standalone neural network training script
├── train.csv       # Training split (59,799 rows: antibody_id, virus_id, neut)
├── test.csv        # Test split (14,931 rows: antibody_id, virus_id, neut)
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
