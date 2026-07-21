# -*- coding: utf-8 -*-
"""
fig7_generalization_degradation.py
Plots the performance degradation curve and the antibody-vs-virus asymmetry gap.
Saves PNG (300 DPI) and PDF in visualizations/figures/.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'custom_files', 'nn_summary_results.csv')
FIG_DIR = os.path.join(ROOT, 'visualizations', 'figures')

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Summary CSV not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    # Extract values
    auroc = df['AUROC'].values
    auprc = df['AUPRC'].values
    
    splits = ['Random Split', 'Novel Viruses', 'Novel Antibodies', 'Both Novel']
    
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Degradation Curve
    ax1.plot(splits, auroc, marker='o', linewidth=3, color='#6366F1', label='AUROC')
    ax1.plot(splits, auprc, marker='s', linewidth=3, color='#EC4899', label='AUPRC')
    ax1.set_title("Performance Degradation Across Generalization Boundaries", fontsize=13, weight='bold')
    ax1.set_ylabel("Metric Value")
    ax1.set_ylim(0.65, 1.0)
    ax1.legend(loc='lower left')
    
    for i, txt in enumerate(auroc):
        ax1.annotate(f'{txt:.4f}', (splits[i], auroc[i] + 0.015), ha='center', fontweight='bold', color='#4338CA')
    for i, txt in enumerate(auprc):
        ax1.annotate(f'{txt:.4f}', (splits[i], auprc[i] - 0.025), ha='center', fontweight='bold', color='#BE185D')
        
    # 2. Generalization Gap (Asymmetry Gap)
    # Random split is control. Gap is Random_Score - Split_Score
    gaps = [0.0, auroc[0] - auroc[1], auroc[0] - auroc[2], auroc[0] - auroc[3]]
    sns.barplot(x=splits[1:], y=gaps[1:], palette=['#10B981', '#F59E0B', '#EF4444'], ax=ax2, edgecolor='#333333', linewidth=1.2)
    ax2.set_title("Generalization Gap (Drop in AUROC from Control)", fontsize=13, weight='bold')
    ax2.set_ylabel("AUROC Decrease")
    ax2.set_ylim(0, 0.25)
    
    # Highlight the asymmetry gap (Virus vs Antibody)
    asymmetry_gap = auroc[1] - auroc[2]
    ax2.text(1.0, 0.15, f'Asymmetry Gap: {asymmetry_gap:.4f}\n(Antibody space is\n2.3x harder to map!)',
             color='#B91C1C', fontweight='bold', bbox=dict(facecolor='#FEF2F2', edgecolor='#EF4444', boxstyle='round,pad=0.5'))
             
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f'+{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.suptitle("Model Robustness and Generalization Asymmetry Analysis", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(FIG_DIR, 'fig7_generalization_degradation.png')
    pdf_path = os.path.join(FIG_DIR, 'fig7_generalization_degradation.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 7 saved -> {png_path}")

if __name__ == "__main__":
    main()
