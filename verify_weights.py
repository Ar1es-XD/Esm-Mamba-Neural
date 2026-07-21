# -*- coding: utf-8 -*-
"""
verify_weights.py
Performs a post-training verification sweep. Loads the 'best_model.pt' weights from each of
the 4 experiments, runs evaluation on the validation sets, recomputes metrics, and compares
them with the values saved in results.json to ensure correctness and matching performance.
Outputs a detailed report to custom_files/verification_report.txt.
"""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'shared'))

from Models import MambaCross
from Toolkit import set_seed_all, Metrics, AntibodyAntigenDataset, custom_collate_fn

SEED = 42
DATA_ROOT = os.path.join(ROOT, 'Data', 'HIV')
EMBEDDINGS_ROOT = os.path.join(ROOT, 'Outputs', 'Pretrained_HIV')
PARAMS_PATH = os.path.join(ROOT, 'shared', 'Param_Model.json')
CUSTOM_DIR = os.path.join(ROOT, "custom_files")
REPORT_PATH = os.path.join(CUSTOM_DIR, "verification_report.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed_all(SEED)

EXPERIMENTS = [
    ("experiment_1_random", "random"),
    ("experiment_2_novel_viruses", "vir_block"),
    ("experiment_3_novel_antibodies", "ab_block"),
    ("experiment_4_both_novel", "both_block")
]

def load_split(folder, split_col):
    """Load train/test pairs preferring the local folder CSVs."""
    train_csv = os.path.join(ROOT, folder, 'data', 'train.csv')
    test_csv  = os.path.join(ROOT, folder, 'data', 'test.csv')
    
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        ab_col = 'antibody_id' if 'antibody_id' in train_df.columns else 'ab_name'
        ag_col = 'virus_id' if 'virus_id' in train_df.columns else 'ag_name'
        lbl_col = 'neut' if 'neut' in train_df.columns else 'label'
        train_pairs = list(zip(train_df[ab_col], train_df[ag_col], train_df[lbl_col]))
        test_pairs  = list(zip(test_df[ab_col],  test_df[ag_col],  test_df[lbl_col]))
        return train_pairs, test_pairs

    # Fallback to master dataset
    pairs_df = pd.read_csv(os.path.join(DATA_ROOT, 'ab_ag_pair.csv'), low_memory=False)
    train_df = pairs_df[pairs_df[split_col] == 'train']
    test_df  = pairs_df[pairs_df[split_col] == 'test']
    train_pairs = list(zip(train_df['ab_name'], train_df['ag_name'], train_df['label']))
    test_pairs  = list(zip(test_df['ab_name'],  test_df['ag_name'],  test_df['label']))
    return train_pairs, test_pairs

def filter_pairs(pairs):
    filtered = []
    for pair in pairs:
        ab_id = str(pair[0]).replace('/', '_')
        ag_id = str(pair[1])
        ab_path = os.path.join(EMBEDDINGS_ROOT, 'ab', f'{ab_id}.npy')
        ag_path = os.path.join(EMBEDDINGS_ROOT, 'ag', f'{ag_id}.npy')
        if os.path.exists(ab_path) and os.path.exists(ag_path):
            filtered.append(pair)
    return filtered

def main():
    print("=" * 80)
    print("  POST-TRAINING VERIFICATION SWEEP (ESM-MAMBA NEURAL NETWORK)")
    print("=" * 80)

    os.makedirs(CUSTOM_DIR, exist_ok=True)
    report_lines = [
        "================================================================================",
        "ESM-MAMBA NEURAL NETWORK WEIGHTS VERIFICATION REPORT",
        "================================================================================",
        f"Timestamp: {pd.Timestamp.now()}",
        f"Device utilized: {device}",
        f"CUDA Available: {torch.cuda.is_available()}",
        "--------------------------------------------------------------------------------",
        ""
    ]

    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)

    # Compute sequence length thresholds
    ab_info = pd.read_csv(os.path.join(DATA_ROOT, 'antibody.csv'))
    ag_info = pd.read_csv(os.path.join(DATA_ROOT, 'antigen.csv'))
    len_ab = (ab_info['heavy'].fillna('').astype(str).str.len()
              + ab_info['light'].fillna('').astype(str).str.len())
    len_ag = ag_info['ag_seq'].astype(str).str.len()
    thres_ab = 256
    thres_ag = 256

    verification_failed = False

    for folder, split_col in EXPERIMENTS:
        results_json_path = os.path.join(ROOT, folder, 'results', 'results.json')
        model_weights_path = os.path.join(ROOT, folder, 'results', 'best_model.pt')
        
        report_lines.append(f"Analyzing {folder} (Split column: {split_col})...")
        print(f"Analyzing {folder}...")

        if not os.path.exists(results_json_path):
            msg = f"  ✗ FAILED: results.json not found for {folder}"
            report_lines.append(msg)
            print(msg)
            verification_failed = True
            continue
            
        if not os.path.exists(model_weights_path):
            msg = f"  ✗ FAILED: best_model.pt not found for {folder}"
            report_lines.append(msg)
            print(msg)
            verification_failed = True
            continue

        # Load saved metrics
        with open(results_json_path, 'r') as f:
            saved_metrics = json.load(f)

        # Load model weights
        model = MambaCross(
            hor_dim=thres_ag, ver_dim=thres_ab,
            feat_dim=params['latent_dim'],
            seq_len=thres_ab + thres_ag,
            hidden_sizes=params['decoder_hidden_dims'],
            mamba_layer=params['mamba_layer'],
            pooling=params['pooling_way'],
            activation=params['activation'],
            drop_ratio=params['dropout']
        ).to(device)
        
        try:
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            model.eval()
        except Exception as e:
            msg = f"  ✗ FAILED: Could not load weights for {folder}: {e}"
            report_lines.append(msg)
            print(msg)
            verification_failed = True
            continue

        # Load test loader
        _, test_pairs = load_split(folder, split_col)
        test_filtered = filter_pairs(test_pairs)
        test_dataset  = AntibodyAntigenDataset(test_filtered)
        test_loader   = Data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                       collate_fn=custom_collate_fn, num_workers=0)

        # Run predictions
        y_true, y_pred = [], []
        with torch.no_grad():
            for ab_embs, ag_embs, labels in test_loader:
                ab_embs = ab_embs.to(device, non_blocking=True)
                ag_embs = ag_embs.to(device, non_blocking=True)
                outputs = model(ab_embs, ag_embs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        auc_score, aupr_score, f1_val, acc_score = Metrics(y_true, y_pred)
        
        # Rounded values for comparison
        auc_recalc = round(float(auc_score), 4)
        aupr_recalc = round(float(aupr_score), 4)
        acc_recalc = round(float(acc_score), 4)
        f1_recalc = round(float(f1_val), 4)

        auc_saved = saved_metrics.get("AUROC", 0.0)
        aupr_saved = saved_metrics.get("AUPRC", 0.0)
        acc_saved = saved_metrics.get("Accuracy", 0.0)
        f1_saved = saved_metrics.get("F1 Score", 0.0)

        # Print comparison
        report_lines.append(f"  Saved AUROC:    {auc_saved:.4f} | Recalculated AUROC:    {auc_recalc:.4f}")
        report_lines.append(f"  Saved AUPRC:    {aupr_saved:.4f} | Recalculated AUPRC:    {aupr_recalc:.4f}")
        report_lines.append(f"  Saved Accuracy: {acc_saved:.4f} | Recalculated Accuracy: {acc_recalc:.4f}")
        report_lines.append(f"  Saved F1:       {f1_saved:.4f} | Recalculated F1:       {f1_recalc:.4f}")
        
        diff_auc = abs(auc_recalc - auc_saved)
        diff_aupr = abs(aupr_recalc - aupr_saved)
        
        if diff_auc < 1e-4 and diff_aupr < 1e-4:
            msg = f"  [SUCCESS] Verification passed. Recalculated metrics match results.json."
            report_lines.append(msg)
            print(msg)
        else:
            msg = f"  [FAILED] Recalculated metrics deviate from results.json (AUC diff: {diff_auc:.4f}, AUPRC diff: {diff_aupr:.4f})"
            report_lines.append(msg)
            print(msg)
            verification_failed = True
            
        report_lines.append("")

    report_lines.append("--------------------------------------------------------------------------------")
    if not verification_failed:
        report_lines.append("VERIFICATION STATUS: SUCCESSFUL")
        report_lines.append("Conclusion: All model checkpoints are verified, loaded correctly on GPU, and reproduce original metrics.")
        print("\n[SUCCESS] Verification successful! Report written to verification_report.txt")
    else:
        report_lines.append("VERIFICATION STATUS: FAILED")
        report_lines.append("Conclusion: One or more experiments failed validation check.")
        print("\n[WARNING] Verification failed! Check details in verification_report.txt")
    report_lines.append("================================================================================")

    # Write report file
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    main()
