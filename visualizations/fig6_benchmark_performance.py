# -*- coding: utf-8 -*-
"""
fig6_benchmark_performance.py
Plots validation AUROC and AUPRC scores for all four generalization splits.
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
    
    # Align names
    df['Experiment_Name'] = df['Experiment'].replace({
        'experiment_1_random': 'Random Split',
        'experiment_2_novel_viruses': 'Novel Viruses',
        'experiment_3_novel_antibodies': 'Novel Antibodies',
        'experiment_4_both_novel': 'Both Novel (Double Holdout)',
        'Experiment 1 – Random Split': 'Random Split',
        'Experiment 2 – Novel Viruses': 'Novel Viruses',
        'Experiment 3 – Novel Antibodies': 'Novel Antibodies',
        'Experiment 4 – Both Novel (Double Holdout)': 'Both Novel (Double Holdout)'
    })
    
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. AUROC Barplot
    sns.barplot(data=df, x='Experiment_Name', y='AUROC', color='#6366F1', ax=ax1, edgecolor='#4338CA', linewidth=1.2)
    ax1.set_title("Validation AUROC Across Splits", fontsize=13, weight='bold')
    ax1.set_xlabel("")
    ax1.set_ylabel("Area Under ROC Curve")
    ax1.set_ylim(0.6, 1.0)
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#312E81')
                     
    # 2. AUPRC Barplot
    sns.barplot(data=df, x='Experiment_Name', y='AUPRC', color='#4F46E5', ax=ax2, edgecolor='#3730A3', linewidth=1.2)
    ax2.set_title("Validation AUPRC Across Splits", fontsize=13, weight='bold')
    ax2.set_xlabel("")
    ax2.set_ylabel("Area Under PR Curve")
    ax2.set_ylim(0.6, 1.0)
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#312E81')
                     
    plt.suptitle("ESM-Mamba Deep Neural Network Performance Benchmarks", fontsize=15, weight='bold', y=0.98)
    plt.tight_layout()
    
    # Save files
    png_path = os.path.join(FIG_DIR, 'fig6_benchmark_performance.png')
    pdf_path = os.path.join(FIG_DIR, 'fig6_benchmark_performance.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure 6 saved -> {png_path}")

if __name__ == "__main__":
    main()
