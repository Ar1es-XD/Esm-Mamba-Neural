# -*- coding: utf-8 -*-
"""
fig2_partition_splits.py
Plots train, test, and excluded pair breakdown across all 4 experiments.
Saves PNG (300 DPI) and PDF in visualizations/figures/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'visualizations', 'figures')

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Dataset splits
    data = {
        'Experiment': ['Exp 1 (Random)', 'Exp 2 (Novel Viruses)', 'Exp 3 (Novel Antibodies)', 'Exp 4 (Both Novel)'],
        'Train': [59799, 61219, 57903, 34774],
        'Test': [14931, 13511, 16827, 7306],
        'Excluded': [0, 0, 0, 32650]
    }
    
    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA"})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot stacked bar chart
    bars_train = ax.bar(df['Experiment'], df['Train'], label='Train Partition', color='#3B82F6', edgecolor='#2563EB', width=0.5)
    bars_test  = ax.bar(df['Experiment'], df['Test'], bottom=df['Train'], label='Test (Holdout) Partition', color='#10B981', edgecolor='#059669', width=0.5)
    bars_excl  = ax.bar(df['Experiment'], df['Excluded'], bottom=np.array(df['Train'])+np.array(df['Test']), label='Excluded Pairs', color='#F59E0B', edgecolor='#D97706', width=0.5)
    
    ax.set_ylabel("Interaction Pair Count", fontsize=12)
    ax.set_title("ESM-Mamba-Neural Partition Splits & Excluded Pairs Across Experiments", fontsize=14, weight='bold')
    ax.legend(loc='upper right', frameon=True)
    
    # Annotate heights
    for i in range(len(df)):
        # Train label
        h_train = df['Train'][i]
        ax.annotate(f'{h_train}', (i, h_train / 2), ha='center', va='center', color='white', fontweight='bold')
        
        # Test label
        h_test = df['Test'][i]
        ax.annotate(f'{h_test}', (i, h_train + h_test / 2), ha='center', va='center', color='white', fontweight='bold')
        
        # Excluded label
        h_excl = df['Excluded'][i]
        if h_excl > 0:
            ax.annotate(f'{h_excl}', (i, h_train + h_test + h_excl / 2), ha='center', va='center', color='white', fontweight='bold')
            
        total = h_train + h_test + h_excl
        ax.annotate(f'Total: {total}', (i, total + 1000), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')

    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(FIG_DIR, 'fig2_partition_splits.png')
    pdf_path = os.path.join(FIG_DIR, 'fig2_partition_splits.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 2 saved -> {png_path}")

if __name__ == "__main__":
    main()
