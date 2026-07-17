import json
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
from Models import DeepSSM
from Loader import alphabet_coding
from Toolkit import set_seed_all, Metrics, make_dir, AntibodyAntigenDataset, custom_collate_fn

# Config
seed = 42
data_root = 'Data/HIV'
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
set_seed_all(seed)

def load_split(split_col="ab_block"):
    """
    Loads data using the mentor's hardcoded splits.
    split_col can be: 'random', 'ab_block', 'vir_block', or 'both_block'
    """
    pairs_df = pd.read_csv(f"{data_root}/ab_ag_pair.csv", low_memory=False)
    
    train_df = pairs_df[pairs_df[split_col] == 'train']
    test_df = pairs_df[pairs_df[split_col] == 'test']
    
    train_pairs = list(zip(train_df['ab_name'], train_df['ag_name'], train_df['label']))
    test_pairs = list(zip(test_df['ab_name'], test_df['ag_name'], test_df['label']))
    
    print(f"Loaded {split_col} split: {len(train_pairs)} Train, {len(test_pairs)} Test")
    return train_pairs, test_pairs

def train_and_eval(split_col="ab_block"):
    train_pairs, test_pairs = load_split(split_col)
    
    with open('Param_Model.json', 'r') as f:
        params = json.load(f)
    
    train_dataset = AntibodyAntigenDataset(train_pairs)
    test_dataset = AntibodyAntigenDataset(test_pairs)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=params['bs'], shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = Data.DataLoader(test_dataset, batch_size=params['bs'], shuffle=False, drop_last=False, collate_fn=custom_collate_fn)
    
    model = DeepSSM(
        d_model=params['d_model'], 
        d_state=params['d_state'], 
        d_conv=params['d_conv'], 
        expand=params['expand']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.BCELoss()
    
    print(f"Starting Training on {device} using {split_col} split...")
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0
        for ab_embs, ag_embs, labels in train_loader:
            ab_embs, ag_embs, labels = ab_embs.to(device), ag_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(ab_embs, ag_embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for ab_embs, ag_embs, labels in test_loader:
                ab_embs, ag_embs = ab_embs.to(device), ag_embs.to(device)
                outputs = model(ab_embs, ag_embs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
                
        auc_score, aupr_score, f1_value, acc_score = Metrics(y_true, y_pred)
        print(f"Epoch {epoch+1}/{params['epochs']} - Loss: {epoch_loss/len(train_loader):.4f} - Test AUC: {auc_score:.4f}")

if __name__ == "__main__":
    # You can change this to 'random', 'vir_block', or 'both_block'
    train_and_eval(split_col="ab_block")
