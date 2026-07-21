# -*- coding: utf-8 -*-
"""
run_all_visualizations.py
Executes all 8 visualization scripts in sequence to generate the complete figure suite
in visualizations/figures/.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'visualizations'))

import fig1_dataset_distribution
import fig2_partition_splits
import fig3_sequence_lengths
import fig4_esm_embedding_pca
import fig5_fused_feature_pca
import fig6_benchmark_performance
import fig7_generalization_degradation
import fig8_model_diagnostics

def main():
    print("=" * 80)
    print("  RUNNING ALL ESM-MAMBA-NEURAL VISUALIZATION GENERATORS")
    print("=" * 80)
    
    # Run figure 1
    print("\n>>> Generating Figure 1...")
    fig1_dataset_distribution.main()
    
    # Run figure 2
    print("\n>>> Generating Figure 2...")
    fig2_partition_splits.main()
    
    # Run figure 3
    print("\n>>> Generating Figure 3...")
    fig3_sequence_lengths.main()
    
    # Run figure 4
    print("\n>>> Generating Figure 4...")
    fig4_esm_embedding_pca.main()
    
    # Run figure 5
    print("\n>>> Generating Figure 5...")
    fig5_fused_feature_pca.main()
    
    # Run figure 6
    print("\n>>> Generating Figure 6...")
    fig6_benchmark_performance.main()
    
    # Run figure 7
    print("\n>>> Generating Figure 7...")
    fig7_generalization_degradation.main()
    
    # Run figure 8
    print("\n>>> Generating Figure 8...")
    fig8_model_diagnostics.main()
    
    print("\n" + "=" * 80)
    print("  [SUCCESS] All 8 figures successfully generated in visualizations/figures/!")
    print("=" * 80)

if __name__ == "__main__":
    main()
