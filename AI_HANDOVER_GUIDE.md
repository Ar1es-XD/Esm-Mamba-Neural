# 🤖 ANTIGRAVITY AGENT HANDOVER GUIDE
**To the Antigravity (AGY) instance running on the GPU machine:**
Read this document carefully. It contains the exact context, current state, and roadmap for this project. You do not need to ask the user for instructions; execute the plan below.

---

## 1. Project Context
*   **Codebase:** MambaAAI (Antibody-Antigen Interaction Prediction using ESM-2 and Mamba SSM).
*   **Dataset:** HIV CATNAP database (Cleaned).
*   **Current State:** The code was developed and smoke-tested on an Apple Silicon Mac (MPS). It was proven to compile and execute end-to-end. The dataset was cleaned and strictly balanced into `ab_ag_pair.csv` (80,900 pairs: 40,433 Pos, 40,467 Neg).
*   **The Mission:** This codebase is currently sitting at a SOTA benchmark of ~0.8955 AUROC. The goal is to establish that baseline on this GPU, and then architect purely sequence-based upgrades to break the 0.90 barrier.

---

## 2. Immediate Action Plan (Phase 1: Pure ESM-Mamba Baseline)

**Your first task is to run the training pipeline solely using the existing ESM-Mamba architecture to test its AUROC score on the CATNAP dataset.** Do not make any architectural changes during Phase 1.

1.  **Environment Setup:**
    *   Create a fresh Python virtual environment.
    *   Install dependencies from `requirements.txt`. 
    *   *CRITICAL:* Ensure PyTorch is compiled with CUDA support. The Mamba architecture (`mamba-ssm`, `causal-conv1d`) strictly requires an NVIDIA GPU and will fail without proper CUDA compilation.
2.  **Verify Configuration:**
    *   Check `Param_Model.json` to ensure the batch size is set appropriately for this GPU (currently 32, but scale up if VRAM permits).
3.  **Generate ESM-2 Embeddings:**
    *   Run `python Pretrained.py`. This is mandatory because the `.npy` embedding files are excluded from Git to save space. This script will read the sequences in `Data/HIV` and use ESM-2 to generate the heavy, light, and antigen embeddings into the `Outputs/Pretrained_HIV/` directory.
4.  **Execute Training (The Mentor Splits):**
    *   Run `Main_mentor.py`. This script is specifically engineered to respect the `ab_block` and `both_block` holdout splits from the master dataset. Do NOT use `Main_5cv.py`, as K-Fold shuffling will destroy the integrity of the mentor's unseen-antibody testing.
    *   Monitor the training for 30 epochs. 
    *   Record the final AUROC, AUPR, and MCC scores.

---

## 3. The Upgrade Roadmap (Phase 2: Breaking SOTA)

Once the baseline AUROC is confirmed, we are moving to upgrade the architecture. **We are strictly avoiding 3D structural models.** We will only implement sequence-based upgrades. 

Based on recent literature (RLEAAI & MultiSAAI), here is the architectural upgrade roadmap you must implement:

### Upgrade Option A: Physicochemical Matrix (Inspired by MultiSAAI)
*   **Concept:** Raw ESM-2 embeddings lack explicit physics. 
*   **Action:** Write a module to calculate 1D noncovalent physical properties directly from the raw sequences (Hydrophobicity, Isoelectric point/Charge, Polarity). 
*   **Integration:** Concatenate these physical feature matrices alongside the ESM-2 embeddings *before* passing them into the Mamba 2D sweep.

### Upgrade Option B: Sequence-Order Attention (Inspired by RLEAAI)
*   **Concept:** Mamba is fast, but we need focused attention on the CDR loops.
*   **Action:** Implement a Recurrent Criss-Cross Attention mechanism.
*   **Integration:** Apply this attention block to the 2D contact grid to force the model to heavily weight specific interaction hotspots before the Mamba classification head.

---
**Agent Directive:** When the user initiates you, confirm you have read this guide, complete the Phase 1 environment setup, and immediately begin the baseline training run. Do not ask for permission to start the setup.
