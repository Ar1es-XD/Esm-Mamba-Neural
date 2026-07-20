# -*- coding: utf-8 -*-
"""
run_all_experiments.py
Master pipeline: sequentially trains ESM-Mamba on all 4 experiment splits
and produces a consolidated summary CSV.
"""
import os
import sys
import json
import subprocess
import pandas as pd

EXPERIMENTS = [
    ("Experiment 1 – Random Split",              "random",     "experiment_1_random"),
    ("Experiment 2 – Novel Viruses",              "vir_block",  "experiment_2_novel_viruses"),
    ("Experiment 3 – Novel Antibodies",           "ab_block",   "experiment_3_novel_antibodies"),
    ("Experiment 4 – Both Novel (Double Holdout)","both_block", "experiment_4_both_novel"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("  Running All ESM-Mamba Neural Network Experiments In Sequence")
    print("=" * 70, "\n")

    summary_rows = []

    for name, split_col, exp_dir in EXPERIMENTS:
        print(f"\n{'─'*70}")
        print(f">>> {name}  (col: {split_col})")
        print(f"{'─'*70}")

        try:
            subprocess.run(
                [sys.executable, os.path.join(ROOT, "train_experiment.py"),
                 "--split_col", split_col,
                 "--epochs", "30"],
                check=True, cwd=ROOT
            )
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        results_path = os.path.join(ROOT, exp_dir, "results", "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    m = json.load(f)
                summary_rows.append({
                    "Experiment": name,
                    "Train n":   m.get("Train n", "N/A"),
                    "Test n":    m.get("Test n", "N/A"),
                    "Test %neut": f"{m.get('Test %neutralizing', 0):.2f}%",
                    "Best Epoch": m.get("Best Epoch", "N/A"),
                    "AUROC":      f"{m.get('AUROC', 0):.4f}",
                    "AUPRC":      f"{m.get('AUPRC', 0):.4f}",
                    "Accuracy":   f"{m.get('Accuracy', 0):.4f}",
                    "F1":         f"{m.get('F1 Score', 0):.4f}",
                })
            except Exception as e:
                print(f"  ✗ Could not read results: {e}")
        else:
            print(f"  ⚠ results.json not found for {split_col}")
            summary_rows.append({
                "Experiment": name,
                "Train n": "N/A", "Test n": "N/A", "Test %neut": "N/A",
                "Best Epoch": "N/A", "AUROC": "N/A", "AUPRC": "N/A",
                "Accuracy": "N/A", "F1": "N/A",
            })

    # ── Summary ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(ROOT, "nn_summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n{'=' * 70}")
    print("                   NEURAL NETWORK RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"{'=' * 70}")
    print(f"\n✓ Summary saved → {summary_csv}\n")


if __name__ == '__main__':
    main()
