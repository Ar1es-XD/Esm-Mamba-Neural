# -*- coding: utf-8 -*-
"""
plot_results.py
Generates high-end biophysical metric visualizations for the ESM-Mamba Deep Neural Network
(MambaCross) pipeline, including a comparative chart against the Logistic Regression baseline.
Saves the output plots directly to the custom_files/ directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(ROOT, "custom_files")
NN_CSV = os.path.join(CUSTOM_DIR, "nn_summary_results.csv")
LR_CSV = os.path.join(ROOT, "..", "esm_lr", "summary_results.csv")

def main():
    print("=" * 80)
    print("  GENERATING PERFORMANCE VISUALIZATIONS (NEURAL NETWORK)")
    print("=" * 80)

    os.makedirs(CUSTOM_DIR, exist_ok=True)

    if not os.path.exists(NN_CSV):
        print(f"Error: Neural Network summary CSV not found at {NN_CSV}")
        return

    # 1. Load Neural Network metrics
    df_nn = pd.read_csv(NN_CSV)
    df_nn['Experiment_Name'] = df_nn['Experiment'].replace({
        'experiment_1_random': 'Random Split',
        'experiment_2_novel_viruses': 'Novel Viruses',
        'experiment_3_novel_antibodies': 'Novel Antibodies',
        'experiment_4_both_novel': 'Both Novel (Double Holdout)',
        'Experiment 1 – Random Split': 'Random Split',
        'Experiment 2 – Novel Viruses': 'Novel Viruses',
        'Experiment 3 – Novel Antibodies': 'Novel Antibodies',
        'Experiment 4 – Both Novel (Double Holdout)': 'Both Novel (Double Holdout)'
    })
    df_nn['Model'] = 'ESM-Mamba Neural Network'
    
    # 2. Check and load Logistic Regression metrics
    has_lr = False
    if os.path.exists(LR_CSV):
        df_lr = pd.read_csv(LR_CSV)
        df_lr['Experiment_Name'] = df_lr['Experiment'].replace({
            'Random': 'Random Split',
            'Novel viruses': 'Novel Viruses',
            'Novel antibodies': 'Novel Antibodies',
            'Both novel': 'Both Novel (Double Holdout)'
        })
        df_lr['Model'] = 'Logistic Regression (L2)'
        has_lr = True

    # Apply professional styling
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })

    # Color Palette: Indigo for NN, Emerald for LR
    colors = {'ESM-Mamba Neural Network': '#6366F1', 'Logistic Regression (L2)': '#10B981'}

    # =========================================================================
    # PLOT 1: Standalone Neural Network Performance
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # AUROC Barplot
    sns.barplot(
        data=df_nn, x='Experiment_Name', y='AUROC',
        color='#6366F1', ax=ax1, edgecolor='#4338CA', linewidth=1.2
    )
    ax1.set_title("ESM-Mamba (MambaCross) AUROC")
    ax1.set_xlabel("")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.6, 1.0)
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#312E81')

    # AUPRC Barplot
    sns.barplot(
        data=df_nn, x='Experiment_Name', y='AUPRC',
        color='#4F46E5', ax=ax2, edgecolor='#3730A3', linewidth=1.2
    )
    ax2.set_title("ESM-Mamba (MambaCross) AUPRC")
    ax2.set_xlabel("")
    ax2.set_ylabel("AUPRC")
    ax2.set_ylim(0.6, 1.0)
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#312E81')

    plt.suptitle("ESM-Mamba Deep Neural Network Generalization Baselines", y=0.98, weight='bold')
    plt.tight_layout()
    
    # Save standalone plot
    nn_plot_path = os.path.join(CUSTOM_DIR, "neural_network_performance.png")
    plt.savefig(nn_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Created standalone NN plot -> {nn_plot_path}")

    print("Plotting complete!")

if __name__ == "__main__":
    main()
