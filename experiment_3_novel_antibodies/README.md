# Experiment 3 – Novel Antibodies (Antibody Holdout)

**Split Column:** `ab_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline Comparison:** esm-up Experiment 3 (L2 Logistic Regression)

---

## 🤖 Standalone AGY Execution Command

To run this experiment independently as AGY:
```cmd
python train_nn.py --epochs 30 --batch_size 32
```

---

## Description & Partitioning

All pairs containing certain antibody identities are held out entirely for testing. The model must generalize zero-shot to previously unseen antibody sequences.

* **Train set (n)**: 57,903 pairs (77.5%) | 58.30% neutralizing
* **Test set (n)**: 16,827 pairs (22.5%) | 59.67% neutralizing
* **Held-out Entities**: 137 unique antibodies completely excluded from training.

---

## Contents

```
experiment_3_novel_antibodies/
├── train_nn.py     # Standalone neural network training script
├── train.csv       # Training split (57,903 rows: antibody_id, virus_id, neut)
├── test.csv        # Test split (16,827 rows: antibody_id, virus_id, neut)
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
