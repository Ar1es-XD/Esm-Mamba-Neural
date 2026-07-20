# -*- coding: utf-8 -*-
"""
train_experiment.py
End-to-end MambaCross neural network trainer for a single experiment split.
Outputs results and model weights into the experiment's own results/ folder.
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

# Add shared/ to path so we can import Models, Toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from Models import MambaCross
from Toolkit import set_seed_all, Metrics, make_dir, AntibodyAntigenDataset, custom_collate_fn

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'Data', 'HIV')
EMBEDDINGS_ROOT = os.path.join(os.path.dirname(__file__), 'Outputs', 'Pretrained_HIV')
PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'shared', 'Param_Model.json')

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

# ── Split → Experiment folder mapping ────────────────────────────────────────
SPLIT_TO_EXPERIMENT = {
    'random':     'experiment_1_random',
    'vir_block':  'experiment_2_novel_viruses',
    'ab_block':   'experiment_3_novel_antibodies',
    'both_block': 'experiment_4_both_novel',
}


def load_split(split_col):
    """Load train/test pairs from ab_ag_pair.csv using the given split column."""
    pairs_df = pd.read_csv(os.path.join(DATA_ROOT, 'ab_ag_pair.csv'), low_memory=False)

    train_df = pairs_df[pairs_df[split_col] == 'train']
    test_df  = pairs_df[pairs_df[split_col] == 'test']

    train_pairs = list(zip(train_df['ab_name'], train_df['ag_name'], train_df['label']))
    test_pairs  = list(zip(test_df['ab_name'],  test_df['ag_name'],  test_df['label']))

    print(f"Loaded '{split_col}' split: {len(train_pairs)} Train, {len(test_pairs)} Test")
    return train_pairs, test_pairs


def filter_pairs(pairs):
    """Keep only pairs whose pre-extracted ESM-2 .npy files exist on disk."""
    filtered = []
    for pair in pairs:
        ab_id = pair[0].replace('/', '_')
        ag_id = pair[1]
        ab_path = os.path.join(EMBEDDINGS_ROOT, 'ab', f'{ab_id}.npy')
        ag_path = os.path.join(EMBEDDINGS_ROOT, 'ag', f'{ag_id}.npy')
        if os.path.exists(ab_path) and os.path.exists(ag_path):
            filtered.append(pair)
    return filtered


def export_split_csvs(train_pairs, test_pairs, data_dir):
    """Save the train/test splits as CSVs into the experiment's data/ folder."""
    make_dir(data_dir)
    for name, pairs in [('train', train_pairs), ('test', test_pairs)]:
        df = pd.DataFrame(pairs, columns=['ab_name', 'ag_name', 'label'])
        df.to_csv(os.path.join(data_dir, f'{name}.csv'), index=False)


def train_and_eval(split_col, epochs, batch_size):
    """Full training + evaluation loop for one experiment split."""
    experiment_dir = os.path.join(os.path.dirname(__file__), SPLIT_TO_EXPERIMENT[split_col])
    results_dir = os.path.join(experiment_dir, 'results')
    data_dir    = os.path.join(experiment_dir, 'data')
    make_dir(results_dir)

    # Load and filter pairs
    train_pairs, test_pairs = load_split(split_col)
    train_filtered = filter_pairs(train_pairs)
    test_filtered  = filter_pairs(test_pairs)

    print(f"Filtered: {len(train_filtered)} Train (from {len(train_pairs)}), "
          f"{len(test_filtered)} Test (from {len(test_pairs)})")

    if len(train_filtered) == 0 or len(test_filtered) == 0:
        print("ERROR: No valid ESM-2 embeddings found. Run shared/Pretrained.py or "
              "copy Outputs/Pretrained_HIV first.")
        return None

    # Export split CSVs (mirrors esm-up structure)
    export_split_csvs(train_filtered, test_filtered, data_dir)

    # Load model hyper-parameters
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)

    # Compute sequence-length thresholds
    ab_info = pd.read_csv(os.path.join(DATA_ROOT, 'antibody.csv'))
    ag_info = pd.read_csv(os.path.join(DATA_ROOT, 'antigen.csv'))
    len_ab = (ab_info['heavy'].fillna('').astype(str).str.len()
              + ab_info['light'].fillna('').astype(str).str.len())
    len_ag = ag_info['ag_seq'].astype(str).str.len()
    thres_ab = int(np.percentile(len_ab, 100))
    thres_ag = int(np.percentile(len_ag, 100))
    print(f"Sequence-length thresholds: ab={thres_ab}, ag={thres_ag}")

    # DataLoaders (pin_memory for CUDA throughput; 0 workers for RAM-cached tensors)
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

    # Model
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

    # Mixed-precision training (bfloat16 on CUDA Blackwell, disabled on MPS/CPU)
    amp_enabled = use_cuda
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    scaler = GradScaler(enabled=amp_enabled)

    best_auc = 0.0
    best_metrics = {}

    print(f"Training on {device} for {epochs} epochs "
          f"{'(AMP bfloat16)' if amp_enabled else '(float32)'} ...")
    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────
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

        # ── Evaluate ─────────────────────────────────────────────────────
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
        print(f"  Epoch {epoch+1:>2}/{epochs}  Loss={avg_loss:.4f}  "
              f"AUC={auc_score:.4f}  AUPRC={aupr_score:.4f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_metrics = {
                "Experiment": SPLIT_TO_EXPERIMENT[split_col],
                "Split Column": split_col,
                "Train n": len(train_filtered),
                "Test n": len(test_filtered),
                "Test %neutralizing": round(float(np.mean(y_true)) * 100, 2),
                "AUROC": round(float(auc_score), 4),
                "AUPRC": round(float(aupr_score), 4),
                "Accuracy": round(float(acc_score), 4),
                "F1 Score": round(float(f1_value), 4),
                "Best Epoch": epoch + 1
            }
            torch.save(model.state_dict(),
                       os.path.join(results_dir, 'best_model.pt'))

    # Save results JSON
    results_json = os.path.join(results_dir, 'results.json')
    with open(results_json, 'w') as f:
        json.dump(best_metrics, f, indent=4)

    print(f"✓ Best AUC={best_auc:.4f} (epoch {best_metrics.get('Best Epoch')}).  "
          f"Saved → {results_json}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ESM-Mamba (MambaCross) on a single experiment split.")
    parser.add_argument('--split_col', type=str, default='ab_block',
                        choices=['random', 'vir_block', 'ab_block', 'both_block'],
                        help="Split column from ab_ag_pair.csv")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_and_eval(args.split_col, args.epochs, args.batch_size)
