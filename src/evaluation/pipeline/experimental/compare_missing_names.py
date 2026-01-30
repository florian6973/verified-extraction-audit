#!/usr/bin/env python3
"""
Compare missing names between two values_found_name_train CSV files.
Finds names that are in the copy file but missing from the original file.
"""

import pandas as pd
import sys
import os

def compare_missing_names(copy_file, original_file):
    """
    Compare two CSV files and find names missing from the original.
    
    Args:
        copy_file: Path to the copy CSV file
        original_file: Path to the original CSV file
    
    Returns:
        DataFrame with missing names
    """
    # Load both CSV files
    print(f"Loading copy file: {copy_file}")
    if not os.path.exists(copy_file):
        print(f"Error: Copy file not found: {copy_file}")
        return None
    
    df_copy = pd.read_csv(copy_file)
    print(f"  Loaded {len(df_copy)} rows")
    print(f"  Columns: {df_copy.columns.tolist()}")
    
    print(f"\nLoading original file: {original_file}")
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return None
    
    df_original = pd.read_csv(original_file)
    print(f"  Loaded {len(df_original)} rows")
    print(f"  Columns: {df_original.columns.tolist()}")
    
    # Extract unique names from 'value' column
    if 'value' not in df_copy.columns:
        print("Error: 'value' column not found in copy file")
        return None
    
    if 'value' not in df_original.columns:
        print("Error: 'value' column not found in original file")
        return None
    
    names_copy = set(df_copy['value'].dropna().unique())
    names_original = set(df_original['value'].dropna().unique())
    
    print(f"\nUnique names in copy file: {len(names_copy)}")
    print(f"Unique names in original file: {len(names_original)}")
    
    # Find names in copy but not in original
    missing_names = names_copy - names_original
    print(f"\nMissing names (in copy but not in original): {len(missing_names)}")
    
    # Also find names in original but not in copy (for completeness)
    extra_names = names_original - names_copy
    print(f"Extra names (in original but not in copy): {len(extra_names)}")
    
    # Create DataFrame with missing names and their details from copy file
    if missing_names:
        df_missing = df_copy[df_copy['value'].isin(missing_names)].copy()
        # Remove duplicates to show unique missing names
        df_missing_unique = df_missing.drop_duplicates(subset=['value'])
        
        print(f"\nMissing names details:")
        print(f"  Total rows with missing names: {len(df_missing)}")
        print(f"  Unique missing names: {len(df_missing_unique)}")
        
        return df_missing_unique
    else:
        print("\nNo missing names found!")
        return pd.DataFrame()
    
    # Also return extra names for reference
    if extra_names:
        df_extra = df_original[df_original['value'].isin(extra_names)].copy()
        df_extra_unique = df_extra.drop_duplicates(subset=['value'])
        return df_missing_unique, df_extra_unique
    else:
        return df_missing_unique, pd.DataFrame()


def main():
    # File paths
    copy_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_10000000/values_found_name_train copy.csv"
    original_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_10000000/values_found_name_train.csv"
    
    # Allow command line arguments to override
    if len(sys.argv) >= 3:
        copy_file = sys.argv[1]
        original_file = sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python compare_missing_names.py [copy_file] [original_file]")
        print("Using default paths...")
    
    print("="*80)
    print("Comparing Missing Names")
    print("="*80)
    
    result = compare_missing_names(copy_file, original_file)
    
    if result is not None and len(result) > 0:
        # Save results
        output_dir = os.path.dirname(original_file)
        output_file = os.path.join(output_dir, "missing_names_comparison.csv")
        result.to_csv(output_file, index=False)
        print(f"\nSaved missing names to: {output_file}")
        
        # Print first few missing names
        print(f"\nFirst 20 missing names:")
        print(result[['value', 'idx', 'value_found']].head(20).to_string(index=False))
        
        if len(result) > 20:
            print(f"\n... and {len(result) - 20} more")
    elif result is not None:
        print("\nNo missing names to save.")


if __name__ == "__main__":
    main()
