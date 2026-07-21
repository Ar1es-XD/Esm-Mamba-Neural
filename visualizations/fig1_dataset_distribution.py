# -*- coding: utf-8 -*-
"""
fig1_dataset_distribution.py
Plots target class balance (neutralizing vs non-neutralizing) and entity representation counts.
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
    
    # Load metadata
    pair_df = pd.read_csv(os.path.join(DATA_ROOT, 'ab_ag_pair.csv'))
    ab_df = pd.read_csv(os.path.join(DATA_ROOT, 'antibody.csv'))
    ag_df = pd.read_csv(os.path.join(DATA_ROOT, 'antigen.csv'))
    
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Target Class Balance
    class_counts = pair_df['label'].value_counts(normalize=True) * 100
    labels = ['Neutralizing (Class 1)', 'Non-Neutralizing (Class 0)']
    colors = ['#6366F1', '#EC4899']
    
    ax1.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=140, 
            colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax1.set_title("Interaction Target Class Balance\n(Total Pairs: 74,730)", fontsize=13, weight='bold')
    
    # 2. Entity counts
    entity_names = ['Unique Antibodies', 'Unique Antigens (Viruses)']
    entity_counts = [len(ab_df), len(ag_df)]
    sns.barplot(x=entity_names, y=entity_counts, palette=['#10B981', '#3B82F6'], ax=ax2, edgecolor='#333333', linewidth=1.2)
    ax2.set_title("Unique Biological Entities in Dataset", fontsize=13, weight='bold')
    ax2.set_ylabel("Count")
    
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f'{int(height)}',
                     (p.get_x() + p.get_width() / 2., height + 50),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.suptitle("ESM-Mamba-Neural Dataset Distribution & Entity Representation", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(FIG_DIR, 'fig1_dataset_distribution.png')
    pdf_path = os.path.join(FIG_DIR, 'fig1_dataset_distribution.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 1 saved -> {png_path}")

if __name__ == "__main__":
    main()
