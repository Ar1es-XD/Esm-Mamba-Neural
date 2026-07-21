# -*- coding: utf-8 -*-
"""
organize_results.py
Background daemon script that waits for training to complete, creates the 'custom_files'
directory, consolidates all results files, populates the metrics table in the documentation,
and logs the completion.
"""
import os
import time
import shutil
import json
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(ROOT, "nn_summary_results.csv")
CUSTOM_DIR = os.path.join(ROOT, "custom_files")

def format_table(df):
    """Format the dataframe into a clean text-based table."""
    return df.to_string(index=False)

def log_change(message):
    changelog_path = os.path.join(ROOT, "changelog.txt")
    if os.path.exists(changelog_path):
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert before the closing separator
        divider = "================================================================================"
        parts = content.rsplit(divider, 1)
        if len(parts) == 2:
            new_log = f"* {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n"
            updated = parts[0] + new_log + divider + parts[1]
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write(updated)

def main():
    print("Daemon started. Waiting for nn_summary_results.csv to be generated...")
    
    # Wait for the master training script to complete and output summary results
    # We poll every 30 seconds
    while not os.path.exists(SUMMARY_CSV):
        time.sleep(30)
    
    # Wait an extra 5 seconds to ensure files are fully written and unlocked
    time.sleep(5)
    
    print("Summary results found! Starting consolidation...")
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    
    # 1. Copy documentation files
    docs = ["pipeline_documentation.txt", "scientific_review.txt", "methodology_explanation.txt", "changelog.txt"]
    for doc in docs:
        src = os.path.join(ROOT, doc)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CUSTOM_DIR, doc))
            
    # 2. Copy consolidated CSV
    shutil.copy(SUMMARY_CSV, os.path.join(CUSTOM_DIR, "nn_summary_results.csv"))
    
    # 3. Copy individual results JSONs
    experiments = [
        ("experiment_1_random", "results_experiment_1_random.json"),
        ("experiment_2_novel_viruses", "results_experiment_2_novel_viruses.json"),
        ("experiment_3_novel_antibodies", "results_experiment_3_novel_antibodies.json"),
        ("experiment_4_both_novel", "results_experiment_4_both_novel.json")
    ]
    
    for folder, dest_name in experiments:
        src_json = os.path.join(ROOT, folder, "results", "results.json")
        if os.path.exists(src_json):
            shutil.copy(src_json, os.path.join(CUSTOM_DIR, dest_name))
            print(f"Copied results for {folder}")
        else:
            print(f"Warning: results.json not found for {folder}")

    # 4. Populate the metrics table placeholder in pipeline_documentation.txt
    try:
        df = pd.read_csv(SUMMARY_CSV)
        table_str = format_table(df)
        
        # Update both root and custom_files documentation
        for doc_dir in [ROOT, CUSTOM_DIR]:
            doc_path = os.path.join(doc_dir, "pipeline_documentation.txt")
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    doc_content = f.read()
                
                updated_content = doc_content.replace("[METRICS_TABLE_PLACEHOLDER]", table_str)
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
        print("Successfully updated metrics table in pipeline_documentation.txt")
    except Exception as e:
        print(f"Error populating metrics table: {e}")
        
    # 5. Run verification sweep
    try:
        print("Running verification sweep...")
        import subprocess
        res = subprocess.run([os.path.join(ROOT, ".venv", "Scripts", "python.exe"), "verify_weights.py"], cwd=ROOT)
        if res.returncode == 0:
            print("Verification sweep completed successfully!")
            log_change("Verification sweep executed and verification_report.txt generated.")
        else:
            print(f"Warning: verification sweep failed with return code {res.returncode}")
            log_change(f"Verification sweep completed with errors (exit code {res.returncode}).")
    except Exception as e:
        print(f"Error running verification sweep: {e}")

    # 6. Log change to changelog
    log_change("Consolidated results folder 'custom_files' created, copied individual results, and updated metrics table.")
    print("Daemon finished successfully.")

if __name__ == "__main__":
    main()
