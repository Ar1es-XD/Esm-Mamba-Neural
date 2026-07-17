import pandas as pd
import os

# Paths
mentor_file = "../modified_data/catnap_master_sequences (1).csv"
data_root = "Data/HIV"
os.makedirs(data_root, exist_ok=True)

print(f"Loading mentor's master file: {mentor_file}")
df = pd.read_csv(mentor_file, low_memory=False)

# 1. Create antibody.csv
print("Extracting unique antibodies...")
antibodies = df[['Antibody', 'heavy_aa', 'light_aa']].drop_duplicates()
antibodies = antibodies.rename(columns={'Antibody': 'ab_name', 'heavy_aa': 'heavy', 'light_aa': 'light'})
antibodies.to_csv(f"{data_root}/antibody.csv", index=False)
print(f"Saved {len(antibodies)} unique antibodies.")

# 2. Create antigen.csv
print("Extracting unique antigens (viruses)...")
antigens = df[['Virus', 'env_aa']].drop_duplicates()
antigens = antigens.rename(columns={'Virus': 'ag_name', 'env_aa': 'ag_seq'})
antigens.to_csv(f"{data_root}/antigen.csv", index=False)
print(f"Saved {len(antigens)} unique viruses.")

# 3. Create ab_ag_pair.csv
print("Extracting pairs and mentor splits...")
pairs = df[['Antibody', 'Virus', 'neut', 'random', 'ab_block', 'vir_block', 'both_block']].copy()
pairs = pairs.rename(columns={'Antibody': 'ab_name', 'Virus': 'ag_name', 'neut': 'label'})
pairs.to_csv(f"{data_root}/ab_ag_pair.csv", index=False)
print(f"Saved {len(pairs)} interactions to ab_ag_pair.csv with hard-coded splits.")

print("\nSuccess! The mentor's flattened dataset has been converted into the relational format MambaAAI expects.")
print("The ab_ag_pair.csv now contains the 'ab_block', 'vir_block', and 'both_block' columns.")
