# Merge all names with their ground truth labels (train/val/other)
# Combines:
# - values_found_name_train_with_ll.csv (train names matched in df_src)
# - values_found_name_val_with_ll.csv (val names matched in df_src)
# - remaining_values_with_ll.csv (other names not in df_src)

import os
import argparse
import pandas as pd

from config_loader import load_config
from config_helper import format_path, get_output_dir

# Parse arguments
parser = argparse.ArgumentParser(description='Merge all names with ground truth labels')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Get paths from config
OUTPUT_DIR = get_output_dir(config)

# Input files
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'values_found_name_train_with_ll.csv')
VAL_FILE = os.path.join(OUTPUT_DIR, 'values_found_name_val_with_ll.csv')
REMAINING_FILE = os.path.join(OUTPUT_DIR, 'remaining_values_with_ll.csv')

# Output file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'all_names_merged.csv')


def merge_all_names():
    """
    Merge all names from train, val, and remaining (other) sources.
    Adds a 'groundtruth' column: 'train', 'val', or 'other'
    """
    
    dfs = []
    
    # Load train names
    if os.path.exists(TRAIN_FILE):
        df_train = pd.read_csv(TRAIN_FILE)
        df_train['groundtruth'] = 'train'
        # Rename columns for consistency
        if 'value' in df_train.columns:
            df_train = df_train.rename(columns={'value': 'name'})
        print(f"Loaded {len(df_train)} train names from {TRAIN_FILE}")
        dfs.append(df_train)
    else:
        print(f"Warning: Train file not found: {TRAIN_FILE}")
    
    # Load val names
    if os.path.exists(VAL_FILE):
        df_val = pd.read_csv(VAL_FILE)
        df_val['groundtruth'] = 'val'
        # Rename columns for consistency
        if 'value' in df_val.columns:
            df_val = df_val.rename(columns={'value': 'name'})
        print(f"Loaded {len(df_val)} val names from {VAL_FILE}")
        dfs.append(df_val)
    else:
        print(f"Warning: Val file not found: {VAL_FILE}")
    
    # Load remaining (other) names
    if os.path.exists(REMAINING_FILE):
        df_remaining = pd.read_csv(REMAINING_FILE)
        df_remaining['groundtruth'] = 'other'
        # Rename columns for consistency
        if 'extracted_name' in df_remaining.columns:
            df_remaining = df_remaining.rename(columns={'extracted_name': 'name'})
        print(f"Loaded {len(df_remaining)} other names from {REMAINING_FILE}")
        dfs.append(df_remaining)
    else:
        print(f"Warning: Remaining file not found: {REMAINING_FILE}")
    
    if not dfs:
        print("No files found to merge!")
        return None
    
    # Merge all dataframes
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Normalize names: title case (First Last) and remove dots
    if 'name' in df_all.columns:
        def clean_name(x):
            if isinstance(x, str):
                # Remove dots
                x = x.replace('.', '')
                # Title case
                x = x.title()
            return x
        df_all['name'] = df_all['name'].apply(clean_name)
        print(f"Normalized {df_all['name'].notna().sum()} names (title case, removed dots)")
    
    # Ensure consistent columns
    # Keep: name, value_found, idx, ll, n_tokens, groundtruth, and optionally others
    essential_cols = ['name', 'value_found', 'idx', 'll', 'groundtruth']
    optional_cols = ['n_tokens', 'list_tokens', 'list_log_probs', 'pii_type', 'split', 'prompt']
    
    # Select available columns
    available_cols = [c for c in essential_cols + optional_cols if c in df_all.columns]
    df_all = df_all[available_cols]
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total names: {len(df_all)}")
    print(f"\nGroundtruth distribution:")
    print(df_all['groundtruth'].value_counts())
    
    print(f"\nll statistics by groundtruth:")
    print(df_all.groupby('groundtruth')['ll'].describe())
    
    # Check for duplicates
    if 'name' in df_all.columns:
        duplicates = df_all[df_all.duplicated(subset=['name'], keep=False)]
        if len(duplicates) > 0:
            print(f"\nWarning: Found {len(duplicates)} duplicate names")
            print("Duplicate examples:")
            print(duplicates.head(10)[['name', 'groundtruth', 'll']])
    
    # Save merged file
    df_all.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved merged names to: {OUTPUT_FILE}")
    
    return df_all


if __name__ == "__main__":
    df_all = merge_all_names()
    
    if df_all is not None:
        print("\n=== Sample of merged data ===")
        print(df_all.head(20).to_string())
