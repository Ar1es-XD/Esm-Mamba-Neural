# Experiment 2 – Novel Viruses (Antigen Holdout)

**Split Column:** `vir_block`  
**Method:** End-to-end ESM-Mamba (MambaCross) neural network  
**Baseline Comparison:** esm-up Experiment 2 (L2 Logistic Regression)

---

## 🤖 Standalone AGY Execution Command

To run this experiment independently as AGY:
```cmd
python train_nn.py --epochs 30 --batch_size 32
```

---

## Description & Partitioning

All pairs containing certain virus (antigen) identities are held out entirely for testing. The model must generalize zero-shot to previously unseen viral sequence variants.

* **Train set (n)**: 61,219 pairs (81.9%) | 58.22% neutralizing
* **Test set (n)**: 13,511 pairs (18.1%) | 60.38% neutralizing
* **Held-out Entities**: 541 unique viruses completely excluded from training.

---

## Contents

```
experiment_2_novel_viruses/
├── train_nn.py     # Standalone neural network training script
├── train.csv       # Training split (61,219 rows: antibody_id, virus_id, neut)
├── test.csv        # Test split (13,511 rows: antibody_id, virus_id, neut)
└── results/
    ├── results.json   # Best-epoch metrics (AUROC, AUPRC, Accuracy, F1)
    └── best_model.pt  # Saved MambaCross model weights
```
