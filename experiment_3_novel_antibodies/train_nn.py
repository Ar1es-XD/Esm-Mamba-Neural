# -*- coding: utf-8 -*-
"""
train_nn.py - Experiment 3: Novel Antibodies
End-to-end MambaCross neural network trainer for Experiment 3 (Novel Antibodies).
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np

# Path setup to import shared modules
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPERIMENT_DIR)
SHARED_DIR = os.path.join(PROJECT_ROOT, 'shared')
sys.path.insert(0, SHARED_DIR)

from Models import MambaCross
from Toolkit import set_seed_all, Metrics, make_dir, AntibodyAntigenDataset, custom_collate_fn

# Experiment Config
SEED = 42
SPLIT_COL = 'ab_block'
EXP_NAME = 'experiment_3_novel_antibodies'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data', 'HIV')
EMBEDDINGS_ROOT = os.path.join(PROJECT_ROOT, 'Outputs', 'Pretrained_HIV')
PARAMS_PATH = os.path.join(SHARED_DIR, 'Param_Model.json')

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
set_seed_all(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')


def load_split():
    """Load train/test pairs for Novel Antibodies (ab_block) partitioning (prefer local folder CSVs)."""
    train_csv = os.path.join(EXPERIMENT_DIR, 'data', 'train.csv')
    test_csv  = os.path.join(EXPERIMENT_DIR, 'data', 'test.csv')
    
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        ab_col = 'antibody_id' if 'antibody_id' in train_df.columns else 'ab_name'
        ag_col = 'virus_id' if 'virus_id' in train_df.columns else 'ag_name'
        lbl_col = 'neut' if 'neut' in train_df.columns else 'label'
        train_pairs = list(zip(train_df[ab_col], train_df[ag_col], train_df[lbl_col]))
        test_pairs  = list(zip(test_df[ab_col],  test_df[ag_col],  test_df[lbl_col]))
        print(f"[Exp 3 - Novel Antibodies] Loaded directly from local folder: {len(train_pairs)} Train, {len(test_pairs)} Test pairs.")
        return train_pairs, test_pairs

    pairs_df = pd.read_csv(os.path.join(DATA_ROOT, 'ab_ag_pair.csv'), low_memory=False)
    train_df = pairs_df[pairs_df[SPLIT_COL] == 'train']
    test_df  = pairs_df[pairs_df[SPLIT_COL] == 'test']

    train_pairs = list(zip(train_df['ab_name'], train_df['ag_name'], train_df['label']))
    test_pairs  = list(zip(test_df['ab_name'],  test_df['ag_name'],  test_df['label']))
    print(f"[Exp 3 - Novel Antibodies] Loaded from ab_ag_pair.csv: {len(train_pairs)} Train, {len(test_pairs)} Test pairs.")
    return train_pairs, test_pairs


def filter_pairs(pairs):
    """Keep only pairs whose pre-extracted ESM-2 .npy files exist on disk."""
    filtered = []
    for pair in pairs:
        ab_id = str(pair[0]).replace('/', '_')
        ag_id = str(pair[1])
        ab_path = os.path.join(EMBEDDINGS_ROOT, 'ab', f'{ab_id}.npy')
        ag_path = os.path.join(EMBEDDINGS_ROOT, 'ag', f'{ag_id}.npy')
        if os.path.exists(ab_path) and os.path.exists(ag_path):
            filtered.append(pair)
    return filtered


def export_split_csvs(train_pairs, test_pairs, data_dir):
    """Export the partition CSVs into the experiment data/ folder."""
    make_dir(data_dir)
    for name, pairs in [('train', train_pairs), ('test', test_pairs)]:
        df = pd.DataFrame(pairs, columns=['antibody_id', 'virus_id', 'neut'])
        df.to_csv(os.path.join(data_dir, f'{name}.csv'), index=False)
        df.to_csv(os.path.join(EXPERIMENT_DIR, f'{name}.csv'), index=False)


def main(epochs=30, batch_size=32):
    results_dir = os.path.join(EXPERIMENT_DIR, 'results')
    data_dir    = os.path.join(EXPERIMENT_DIR, 'data')
    make_dir(results_dir)

    train_pairs, test_pairs = load_split()
    train_filtered = filter_pairs(train_pairs)
    test_filtered  = filter_pairs(test_pairs)

    print(f"[Exp 3 - Novel Antibodies] Filtered: {len(train_filtered)} Train, {len(test_filtered)} Test pairs.")
    if len(train_filtered) == 0 or len(test_filtered) == 0:
        print("ERROR: No valid ESM-2 embeddings found. Run shared/Pretrained.py or copy Outputs/Pretrained_HIV first.")
        return None

    export_split_csvs(train_filtered, test_filtered, data_dir)

    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)

    ab_info = pd.read_csv(os.path.join(DATA_ROOT, 'antibody.csv'))
    ag_info = pd.read_csv(os.path.join(DATA_ROOT, 'antigen.csv'))
    len_ab = (ab_info['heavy'].fillna('').astype(str).str.len()
              + ab_info['light'].fillna('').astype(str).str.len())
    len_ag = ag_info['ag_seq'].astype(str).str.len()
    thres_ab = int(np.percentile(len_ab, 100))
    thres_ag = int(np.percentile(len_ag, 100))

    use_cuda = device.type == 'cuda'
    loader_kwargs = dict(
        pin_memory=use_cuda,
        num_workers=0,
    )
    train_dataset = AntibodyAntigenDataset(train_filtered)
    test_dataset  = AntibodyAntigenDataset(test_filtered)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=True,
                                   collate_fn=custom_collate_fn, **loader_kwargs)
    test_loader  = Data.DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=False, drop_last=False,
                                   collate_fn=custom_collate_fn, **loader_kwargs)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.BCELoss()

    amp_enabled = use_cuda
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    scaler = GradScaler(enabled=amp_enabled)

    best_auc = 0.0
    best_metrics = {}

    print(f"Training Exp 3 (Novel Antibodies) on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for ab_embs, ag_embs, labels in train_loader:
            ab_embs = ab_embs.to(device, non_blocking=True)
            ag_embs = ag_embs.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(ab_embs, ag_embs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for ab_embs, ag_embs, labels in test_loader:
                ab_embs = ab_embs.to(device, non_blocking=True)
                ag_embs = ag_embs.to(device, non_blocking=True)
                outputs = model(ab_embs, ag_embs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        auc_score, aupr_score, f1_value, acc_score = Metrics(y_true, y_pred)
        avg_loss = epoch_loss / len(train_loader)
        print(f"  [Exp 3] Epoch {epoch+1:>2}/{epochs} Loss={avg_loss:.4f} AUC={auc_score:.4f} AUPRC={aupr_score:.4f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_metrics = {
                "Experiment": EXP_NAME,
                "Split Column": SPLIT_COL,
                "Train n": len(train_filtered),
                "Test n": len(test_filtered),
                "Test %neutralizing": round(float(np.mean(y_true)) * 100, 2),
                "AUROC": round(float(auc_score), 4),
                "AUPRC": round(float(aupr_score), 4),
                "Accuracy": round(float(acc_score), 4),
                "F1 Score": round(float(f1_value), 4),
                "Best Epoch": epoch + 1
            }
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))

    results_json = os.path.join(results_dir, 'results.json')
    with open(results_json, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    print(f"✓ [Exp 3] Best AUC={best_auc:.4f} (epoch {best_metrics.get('Best Epoch')}). Saved -> {results_json}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESM-Mamba on Experiment 3 (Novel Antibodies)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args.epochs, args.batch_size)
