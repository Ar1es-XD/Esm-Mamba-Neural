# -*- coding: utf-8 -*-
"""
fig3_sequence_lengths.py
Plots sequence length distributions for Heavy+Light antibodies and antigens.
Saves PNG (300 DPI) and PDF in visualizations/figures/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'Data', 'HIV')
FIG_DIR = os.path.join(ROOT, 'visualizations', 'figures')

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Load sequence info
    ab_df = pd.read_csv(os.path.join(DATA_ROOT, 'antibody.csv'))
    ag_df = pd.read_csv(os.path.join(DATA_ROOT, 'antigen.csv'))
    
    len_ab = (ab_df['heavy'].fillna('').astype(str).str.len()
              + ab_df['light'].fillna('').astype(str).str.len())
    len_ag = ag_df['ag_seq'].astype(str).str.len()
    
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Antibody Length Distribution
    sns.histplot(len_ab, kde=True, color='#6366F1', ax=ax1, edgecolor='#4338CA', linewidth=1.2)
    ax1.set_title("Antibody Heavy + Light Chain Lengths", fontsize=13, weight='bold')
    ax1.set_xlabel("Sequence Length (amino acids)")
    ax1.set_ylabel("Count")
    # Draw threshold line of 256
    ax1.axvline(256, color='#EF4444', linestyle='--', linewidth=2, label='Adaptive Pooling Target (256)')
    ax1.legend(loc='upper right')
    
    # 2. Antigen Length Distribution
    sns.histplot(len_ag, kde=True, color='#10B981', ax=ax2, edgecolor='#059669', linewidth=1.2)
    ax2.set_title("Antigen Sequence Lengths", fontsize=13, weight='bold')
    ax2.set_xlabel("Sequence Length (amino acids)")
    ax2.set_ylabel("Count")
    # Draw threshold line of 256
    ax2.axvline(256, color='#EF4444', linestyle='--', linewidth=2, label='Adaptive Pooling Target (256)')
    ax2.legend(loc='upper right')
    
    plt.suptitle("ESM-Mamba-Neural Input Sequence Length Distributions", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(FIG_DIR, 'fig3_sequence_lengths.png')
    pdf_path = os.path.join(FIG_DIR, 'fig3_sequence_lengths.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 3 saved -> {png_path}")

if __name__ == "__main__":
    main()
