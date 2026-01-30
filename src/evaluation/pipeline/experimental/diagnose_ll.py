# Diagnose ll values: compare ll from pipeline join vs recomputed ll
# Check if the values are consistent

import os
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/pii_leakage/experimental-recall')

# Files from check_names.py (ll joined from df_src)
TRAIN_WITH_LL_FILE = os.path.join(OUTPUT_DIR, 'values_found_name_train_with_ll.csv')
VAL_WITH_LL_FILE = os.path.join(OUTPUT_DIR, 'values_found_name_val_with_ll.csv')

# File from compute_ll_names.py (recomputed ll)
COMPUTED_LL_FILE = os.path.join(OUTPUT_DIR, 'all_names_ll_computed.csv')

# Original source file (ll_all_output from pipeline)
SRC_LL_FILE = os.path.join(BASE_DIR, 'outputs/pii_leakage/pipeline/ll_all_output_False_1B_10_batch.csv')


def load_and_compare():
    print("="*60)
    print("Diagnosing LL Values")
    print("="*60)
    
    # Load train with ll (from join)
    print(f"\nLoading train with ll from join: {TRAIN_WITH_LL_FILE}")
    df_train_joined = pd.read_csv(TRAIN_WITH_LL_FILE)
    print(f"  Rows: {len(df_train_joined)}")
    print(f"  Columns: {df_train_joined.columns.tolist()}")
    print(f"  ll non-null: {df_train_joined['ll'].notna().sum()}")
    
    # Load val with ll (from join)
    print(f"\nLoading val with ll from join: {VAL_WITH_LL_FILE}")
    df_val_joined = pd.read_csv(VAL_WITH_LL_FILE)
    print(f"  Rows: {len(df_val_joined)}")
    print(f"  ll non-null: {df_val_joined['ll'].notna().sum()}")
    
    # Combine joined data
    df_joined = pd.concat([df_train_joined, df_val_joined])
    df_joined['ll_joined'] = df_joined['ll']
    print(f"\nCombined joined data: {len(df_joined)} rows")
    
    # Load recomputed ll
    print(f"\nLoading recomputed ll: {COMPUTED_LL_FILE}")
    df_computed = pd.read_csv(COMPUTED_LL_FILE)
    print(f"  Rows: {len(df_computed)}")
    print(f"  Columns: {df_computed.columns.tolist()}")
    
    # Check what ll columns exist in computed
    ll_cols = [c for c in df_computed.columns if 'll_' in c.lower()]
    print(f"  LL columns: {ll_cols}")
    
    # Load original source
    print(f"\nLoading original source: {SRC_LL_FILE}")
    df_src = pd.read_csv(SRC_LL_FILE)
    print(f"  Rows: {len(df_src)}")
    print(f"  Columns: {df_src.columns.tolist()}")
    
    # Filter source for Name: prompt and name-patient type
    df_src_name = df_src[(df_src['prompt'] == 'Name: ') & (df_src['pii_type'] == 'name-patient')]
    df_src_name = df_src_name[df_src_name['pii_rate'] == 1.0]
    df_src_name = df_src_name[df_src_name['n_epochs'] == 3]
    print(f"\nFiltered source (Name: , name-patient, pii_rate=1.0, n_epochs=3): {len(df_src_name)} rows")
    
    # Merge joined and computed on name
    print("\n" + "="*60)
    print("Comparing LL values")
    print("="*60)
    
    # Normalize names for comparison
    df_joined['name_lower'] = df_joined['value'].str.lower().str.strip()
    df_computed['name_lower'] = df_computed['name'].str.lower().str.strip()
    
    # Merge
    df_compare = df_joined[['value', 'name_lower', 'll_joined']].merge(
        df_computed[['name', 'name_lower'] + ll_cols],
        on='name_lower',
        how='inner'
    )
    print(f"\nMatched entries: {len(df_compare)}")
    
    if len(df_compare) == 0:
        print("No matching entries found!")
        print("\nSample joined names:")
        print(df_joined['name_lower'].head(10).tolist())
        print("\nSample computed names:")
        print(df_computed['name_lower'].head(10).tolist())
        return
    
    # Find the finetuned Name: column
    ft_name_col = None
    for col in ll_cols:
        if 'finetuned' in col.lower() and 'name' in col.lower():
            ft_name_col = col
            break
    
    if ft_name_col is None:
        print(f"Could not find finetuned Name: column in {ll_cols}")
        return
    
    print(f"\nComparing ll_joined vs {ft_name_col}")
    
    # Compare values
    df_compare['ll_recomputed'] = df_compare[ft_name_col]
    df_compare['diff'] = df_compare['ll_joined'] - df_compare['ll_recomputed']
    df_compare['abs_diff'] = df_compare['diff'].abs()
    
    print(f"\nDifference statistics (ll_joined - ll_recomputed):")
    print(f"  Mean diff:     {df_compare['diff'].mean():.6f}")
    print(f"  Std diff:      {df_compare['diff'].std():.6f}")
    print(f"  Min diff:      {df_compare['diff'].min():.6f}")
    print(f"  Max diff:      {df_compare['diff'].max():.6f}")
    print(f"  Mean abs diff: {df_compare['abs_diff'].mean():.6f}")
    
    # Check for large differences
    large_diff_threshold = 0.1
    large_diffs = df_compare[df_compare['abs_diff'] > large_diff_threshold]
    print(f"\nEntries with abs diff > {large_diff_threshold}: {len(large_diffs)}")
    
    if len(large_diffs) > 0:
        print("\nSample large differences:")
        print(large_diffs[['value', 'll_joined', 'll_recomputed', 'diff']].head(20).to_string())
    
    # Check correlation
    corr = df_compare['ll_joined'].corr(df_compare['ll_recomputed'])
    print(f"\nCorrelation: {corr:.6f}")
    
    # Sample comparison
    print("\n" + "="*60)
    print("Sample Comparison (first 20)")
    print("="*60)
    print(df_compare[['value', 'll_joined', 'll_recomputed', 'diff']].head(20).to_string())
    
    # Also compare with original source
    print("\n" + "="*60)
    print("Comparing with original source file")
    print("="*60)
    
    df_src_name['name_lower'] = df_src_name['value'].str.lower().str.strip()
    df_compare_src = df_compare.merge(
        df_src_name[['name_lower', 'll']].rename(columns={'ll': 'll_src'}),
        on='name_lower',
        how='inner'
    )
    print(f"Matched with source: {len(df_compare_src)}")
    
    if len(df_compare_src) > 0:
        df_compare_src['diff_joined_src'] = df_compare_src['ll_joined'] - df_compare_src['ll_src']
        df_compare_src['diff_recomputed_src'] = df_compare_src['ll_recomputed'] - df_compare_src['ll_src']
        
        print(f"\nll_joined vs ll_src (should be same):")
        print(f"  Mean diff: {df_compare_src['diff_joined_src'].mean():.6f}")
        print(f"  Max diff:  {df_compare_src['diff_joined_src'].abs().max():.6f}")
        
        print(f"\nll_recomputed vs ll_src:")
        print(f"  Mean diff: {df_compare_src['diff_recomputed_src'].mean():.6f}")
        print(f"  Max diff:  {df_compare_src['diff_recomputed_src'].abs().max():.6f}")
        
        print("\nSample 3-way comparison:")
        print(df_compare_src[['value', 'll_joined', 'll_recomputed', 'll_src', 'diff_joined_src', 'diff_recomputed_src']].head(20).to_string())
    
    # Save comparison
    output_file = os.path.join(OUTPUT_DIR, 'diagnose_ll_comparison.csv')
    df_compare.to_csv(output_file, index=False)
    print(f"\nSaved comparison to: {output_file}")
    
    return df_compare


if __name__ == "__main__":
    df_compare = load_and_compare()
