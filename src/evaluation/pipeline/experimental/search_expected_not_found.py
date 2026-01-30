#!/usr/bin/env python3
"""
Search for expected_not_found names in the generation parquet file.
For each name, performs string-wise search in the 'value' column.
"""

import pandas as pd
import os
import sys
from tqdm import tqdm

def search_names_in_parquet(expected_not_found_file, parquet_file, ll_all_output_file=None, value_col='value'):
    """
    Search for each name from expected_not_found.csv in the parquet file.
    
    Args:
        expected_not_found_file: Path to CSV with expected but not found names
        parquet_file: Path to parquet file with generation data
        ll_all_output_file: Path to CSV with ll_all_output data (for token lookup)
        value_col: Column name to search in (default: 'value')
    
    Returns:
        DataFrame with search results
    """
    print("="*80)
    print("Searching Expected Not Found Names in Generation File")
    print("="*80)
    
    # Load expected_not_found names
    print(f"\nLoading expected_not_found names from: {expected_not_found_file}")
    if not os.path.exists(expected_not_found_file):
        print(f"Error: File not found: {expected_not_found_file}")
        return None
    
    df_expected = pd.read_csv(expected_not_found_file)
    print(f"  Loaded {len(df_expected)} rows")
    print(f"  Columns: {df_expected.columns.tolist()}")
    
    if 'value' not in df_expected.columns:
        print("Error: 'value' column not found in expected_not_found file")
        return None
    
    expected_names = df_expected['value'].dropna().unique().tolist()
    print(f"  Unique names to search: {len(expected_names)}")
    
    # Load ll_all_output file for token lookup
    value_to_tokens_source = {}
    if ll_all_output_file and os.path.exists(ll_all_output_file):
        print(f"\nLoading ll_all_output file for token lookup: {ll_all_output_file}")
        try:
            df_ll_all = pd.read_csv(ll_all_output_file)
            print(f"  Loaded {len(df_ll_all)} rows")
            print(f"  Columns: {df_ll_all.columns.tolist()}")
            
            if 'value' in df_ll_all.columns:
                # Create mapping from value to tokens
                for idx, row in df_ll_all.iterrows():
                    val = str(row['value'])
                    if val not in value_to_tokens_source:
                        value_to_tokens_source[val] = {
                            'tokens': row.get('tokens', None),
                            'list_tokens': row.get('list_tokens', None)
                        }
                print(f"  Created token mapping for {len(value_to_tokens_source)} unique values")
            else:
                print(f"  Warning: 'value' column not found in ll_all_output file")
        except Exception as e:
            print(f"  Warning: Failed to load ll_all_output file: {e}")
    elif ll_all_output_file:
        print(f"\nWarning: ll_all_output file not found: {ll_all_output_file}")
        print("  Continuing without source token lookup...")
    
    # Load parquet file
    print(f"\nLoading parquet file: {parquet_file}")
    if not os.path.exists(parquet_file):
        print(f"Error: File not found: {parquet_file}")
        return None
    
    print("  Reading parquet file (this may take a while)...")
    try:
        df_parquet = pd.read_parquet(parquet_file)
        print(f"  Loaded {len(df_parquet)} rows")
        print(f"  Columns: {df_parquet.columns.tolist()}")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None
    
    if value_col not in df_parquet.columns:
        print(f"Error: '{value_col}' column not found in parquet file")
        return None
    
    # Convert value column to string for searching
    print(f"\nConverting '{value_col}' column to string for searching...")
    df_parquet[value_col] = df_parquet[value_col].astype(str)
    
    # Check for token columns in parquet
    has_tokens = 'tokens' in df_parquet.columns
    has_list_tokens = 'list_tokens' in df_parquet.columns
    print(f"  Token columns in parquet: tokens={has_tokens}, list_tokens={has_list_tokens}")
    
    # Check for token columns in expected_not_found
    has_expected_tokens = 'tokens' in df_expected.columns
    has_expected_list_tokens = 'list_tokens' in df_expected.columns
    print(f"  Token columns in expected_not_found: tokens={has_expected_tokens}, list_tokens={has_expected_list_tokens}")
    
    # Also check if we have tokens from ll_all_output
    has_source_tokens = len(value_to_tokens_source) > 0
    if has_source_tokens:
        # Check if any entry has tokens
        has_source_tokens_col = any(v.get('tokens') is not None for v in value_to_tokens_source.values())
        has_source_list_tokens_col = any(v.get('list_tokens') is not None for v in value_to_tokens_source.values())
        print(f"  Token columns from ll_all_output: tokens={has_source_tokens_col}, list_tokens={has_source_list_tokens_col}")
    
    # Create a mapping from value to token info for faster lookup
    print("  Creating value-to-tokens mapping...")
    value_to_tokens = {}
    if has_tokens or has_list_tokens:
        for idx, row in tqdm(df_parquet.iterrows(), desc="Creating value-to-tokens mapping", total=len(df_parquet)):
            val = str(row[value_col])
            if val not in value_to_tokens:
                value_to_tokens[val] = {
                    'tokens': row.get('tokens', None),
                    'list_tokens': row.get('list_tokens', None)
                }
    
    # Get all unique values from parquet for faster lookup
    print("  Creating set of all values for fast lookup...")
    all_values_set = set(df_parquet[value_col].unique())
    print(f"  Unique values in parquet: {len(all_values_set)}")
    
    # Search for each name
    print(f"\nSearching for {len(expected_names)} names...")
    results = []
    
    for name in tqdm(expected_names, desc="Searching names"):
        name_str = str(name)
        found = name_str in all_values_set
        
        # Also check for substring matches (name appears anywhere in a value)
        substring_matches = []
        matched_value = None
        if not found:
            # Check if name is a substring of any value
            for val in all_values_set:
                if name_str in val:
                    substring_matches.append(val)
                    matched_value = val
                    found = True  # Mark as found if substring match exists
                    break  # Just need to know if it exists, not all matches
        else:
            matched_value = name_str
        
        # Get additional info from expected_not_found if available
        name_info = df_expected[df_expected['value'] == name].iloc[0] if len(df_expected[df_expected['value'] == name]) > 0 else None
        
        result = {
            'value': name_str,
            'found': found,
            'match_type': 'exact' if found and not substring_matches else ('substring' if substring_matches else 'not_found'),
            'example_match': substring_matches[0] if substring_matches else (name_str if found else None)
        }
        
        # Add probability columns if available
        if name_info is not None:
            if 'pi' in name_info:
                result['pi'] = name_info['pi']
            if 'p_extracted' in name_info:
                result['p_extracted'] = name_info['p_extracted']
            # Add expected tokens from expected_not_found file if available
            if has_expected_tokens and pd.notna(name_info.get('tokens')):
                result['expected_tokens'] = name_info['tokens']
            if has_expected_list_tokens and pd.notna(name_info.get('list_tokens')):
                result['expected_list_tokens'] = name_info['list_tokens']
        
        # Look up tokens from ll_all_output file if not already found
        if name_str in value_to_tokens_source:
            source_token_info = value_to_tokens_source[name_str]
            if source_token_info['tokens'] is not None and 'expected_tokens' not in result:
                result['expected_tokens'] = source_token_info['tokens']
            if source_token_info['list_tokens'] is not None and 'expected_list_tokens' not in result:
                result['expected_list_tokens'] = source_token_info['list_tokens']
        
        # Add tokens from parquet if found
        if found and matched_value and matched_value in value_to_tokens:
            token_info = value_to_tokens[matched_value]
            if token_info['tokens'] is not None:
                result['found_tokens'] = token_info['tokens']
            if token_info['list_tokens'] is not None:
                result['found_list_tokens'] = token_info['list_tokens']
        
        results.append(result)
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("Search Results Summary")
    print("="*80)
    
    found_count = df_results['found'].sum()
    not_found_count = len(df_results) - found_count
    exact_matches = (df_results['match_type'] == 'exact').sum()
    substring_matches = (df_results['match_type'] == 'substring').sum()
    
    print(f"\nTotal names searched: {len(df_results)}")
    print(f"  Found: {found_count} ({found_count/len(df_results)*100:.2f}%)")
    print(f"    Exact matches: {exact_matches}")
    print(f"    Substring matches: {substring_matches}")
    print(f"  Not found: {not_found_count} ({not_found_count/len(df_results)*100:.2f}%)")
    
    # Report token information availability
    print(f"\nToken information:")
    if 'expected_list_tokens' in df_results.columns:
        has_expected_tokens = df_results['expected_list_tokens'].notna().sum()
        print(f"  Names with expected_list_tokens: {has_expected_tokens}/{len(df_results)}")
    if 'expected_tokens' in df_results.columns:
        has_expected_tokens_col = df_results['expected_tokens'].notna().sum()
        print(f"  Names with expected_tokens: {has_expected_tokens_col}/{len(df_results)}")
    if 'found_list_tokens' in df_results.columns:
        has_found_tokens = df_results['found_list_tokens'].notna().sum()
        print(f"  Found names with found_list_tokens: {has_found_tokens}/{found_count}")
    if 'found_tokens' in df_results.columns:
        has_found_tokens_col = df_results['found_tokens'].notna().sum()
        print(f"  Found names with found_tokens: {has_found_tokens_col}/{found_count}")
    
    # Show statistics for found vs not found
    if 'p_extracted' in df_results.columns:
        found_with_prob = df_results[df_results['found'] == True]['p_extracted'].dropna()
        not_found_with_prob = df_results[df_results['found'] == False]['p_extracted'].dropna()
        
        if len(found_with_prob) > 0:
            print(f"\nFound names (with probability data):")
            print(f"  Count: {len(found_with_prob)}")
            print(f"  Mean P(E_i; N): {found_with_prob.mean():.6f}")
            print(f"  Min P(E_i; N): {found_with_prob.min():.6f}")
            print(f"  Max P(E_i; N): {found_with_prob.max():.6f}")
        
        if len(not_found_with_prob) > 0:
            print(f"\nNot found names (with probability data):")
            print(f"  Count: {len(not_found_with_prob)}")
            print(f"  Mean P(E_i; N): {not_found_with_prob.mean():.6f}")
            print(f"  Min P(E_i; N): {not_found_with_prob.min():.6f}")
            print(f"  Max P(E_i; N): {not_found_with_prob.max():.6f}")
    
    # Show examples
    print(f"\nFirst 20 found names:")
    found_df = df_results[df_results['found'] == True].head(20)
    display_cols = ['value', 'match_type']
    if 'p_extracted' in found_df.columns:
        display_cols.append('p_extracted')
    if 'expected_list_tokens' in found_df.columns:
        display_cols.append('expected_list_tokens')
    if 'found_list_tokens' in found_df.columns:
        display_cols.append('found_list_tokens')
    display_cols.append('example_match')
    
    available_cols = [col for col in display_cols if col in found_df.columns]
    print(found_df[available_cols].to_string(index=False))
    
    if not_found_count > 0:
        print(f"\nFirst 20 not found names:")
        not_found_df = df_results[df_results['found'] == False].head(20)
        display_cols = ['value']
        if 'p_extracted' in not_found_df.columns:
            display_cols.append('p_extracted')
        if 'expected_list_tokens' in not_found_df.columns:
            display_cols.append('expected_list_tokens')
        
        available_cols = [col for col in display_cols if col in not_found_df.columns]
        print(not_found_df[available_cols].to_string(index=False))
    
    # Save results
    output_dir = os.path.dirname(expected_not_found_file)
    output_file = os.path.join(output_dir, "expected_not_found_search_results.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved search results to: {output_file}")
    
    return df_results


def main():
    # Default file paths
    expected_not_found_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_10000000/expected_not_found.csv"
    parquet_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-all-test/generation_False_all_10_1B_0.1_3_10000000.parquet"
    ll_all_output_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_10_batch.csv"
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        expected_not_found_file = sys.argv[1]
        parquet_file = sys.argv[2]
    if len(sys.argv) >= 4:
        ll_all_output_file = sys.argv[3]
    
    if len(sys.argv) < 3:
        print("Usage: python search_expected_not_found.py [expected_not_found_file] [parquet_file] [ll_all_output_file]")
        print("Using default paths...")
    
    # Search
    results = search_names_in_parquet(
        expected_not_found_file=expected_not_found_file,
        parquet_file=parquet_file,
        ll_all_output_file=ll_all_output_file
    )
    
    if results is None:
        print("\nSearch failed")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Search completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
