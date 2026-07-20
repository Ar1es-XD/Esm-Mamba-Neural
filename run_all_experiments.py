# -*- coding: utf-8 -*-
"""
run_all_experiments.py
Master pipeline: runs all 4 self-contained ESM-Mamba neural network experiments
(either sequentially or concurrently in parallel) by invoking each folder's train_nn.py script,
and aggregates metrics into nn_summary_results.csv.
"""
import os
import sys
import json
import argparse
import subprocess
import pandas as pd

EXPERIMENTS = [
    ("experiment_1_random",             "Experiment 1 – Random Split"),
    ("experiment_2_novel_viruses",      "Experiment 2 – Novel Viruses"),
    ("experiment_3_novel_antibodies",   "Experiment 3 – Novel Antibodies"),
    ("experiment_4_both_novel",         "Experiment 4 – Both Novel (Double Holdout)"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_experiment(folder, name, epochs, batch_size):
    exp_dir = os.path.join(ROOT, folder)
    print(f"\n{'─'*70}\n>>> Launching {name} ({folder})...\n{'─'*70}")
    
    cmd = [sys.executable, "train_nn.py", "--epochs", str(epochs), "--batch_size", str(batch_size)]
    res = subprocess.run(cmd, cwd=exp_dir)
    return folder, name, res.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all ESM-Neu neural network experiments")
    parser.add_argument('--parallel', action='store_true', help="Run experiments concurrently at the same time in parallel processes")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Running All ESM-Mamba Neural Network Experiments ({'Parallel' if args.parallel else 'Sequential'})")
    print("=" * 70, "\n")

    if args.parallel:
        from concurrent.futures import ProcessPoolExecutor
        print("Launching all 4 experiments concurrently at the same time...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_experiment, folder, name, args.epochs, args.batch_size)
                for folder, name in EXPERIMENTS
            ]
            for f in futures:
                f.result()
    else:
        for folder, name in EXPERIMENTS:
            run_experiment(folder, name, args.epochs, args.batch_size)

    # Aggregate results into consolidated summary table
    summary_rows = []
    for folder, name in EXPERIMENTS:
        results_path = os.path.join(ROOT, folder, "results", "results.json")
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
                print(f"  ✗ Could not read results for {name}: {e}")
        else:
            print(f"  ⚠ results.json not found for {folder}")
            summary_rows.append({
                "Experiment": name,
                "Train n": "N/A", "Test n": "N/A", "Test %neut": "N/A",
                "Best Epoch": "N/A", "AUROC": "N/A", "AUPRC": "N/A",
                "Accuracy": "N/A", "F1": "N/A",
            })

    # Save nn_summary_results.csv at top level
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
