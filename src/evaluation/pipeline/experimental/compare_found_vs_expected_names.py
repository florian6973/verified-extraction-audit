#!/usr/bin/env python3
"""
Compare found names with expected names based on theoretical probabilities.
For a given budget, calculates which names are expected to be extracted
and compares with actually found names.
"""

import pandas as pd
import numpy as np
import os
import sys

def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    """P(name i appears at least once) after N i.i.d. draws."""
    return 1.0 - np.power((1.0 - pi), N)


def get_expected_names(df_scores, budget, pi_col="p_ft_Name: ", split_col="split_x", 
                       threshold_method="expected_count", threshold_value=None):
    """
    Get expected names to be extracted based on theoretical probabilities.
    
    Args:
        df_scores: DataFrame with scores and probabilities
        budget: Budget N (number of attempts)
        pi_col: Column name for per-draw extraction probability
        split_col: Column name for split (to filter train names)
        threshold_method: How to select expected names
            - "expected_count": Select top K names where K = expected unique count (sum of P(E_i; N))
            - "probability": Select names where P(E_i; N) >= threshold_value
            - "top_k": Select top threshold_value names by P(E_i; N)
            - "percentile": Select names above threshold_value percentile
        threshold_value: Threshold value for selection (ignored for "expected_count")
    
    Returns:
        DataFrame with expected names and their probabilities
    """
    # Filter to train names only
    if split_col in df_scores.columns:
        df_train = df_scores[df_scores[split_col].astype(str).str.lower() == "train"].copy()
    else:
        print(f"Warning: '{split_col}' column not found, using all names")
        df_train = df_scores.copy()
    
    if len(df_train) == 0:
        print("Error: No train names found")
        return pd.DataFrame()
    
    # Get pi values
    if pi_col not in df_train.columns:
        print(f"Error: '{pi_col}' column not found")
        print(f"Available columns: {df_train.columns.tolist()}")
        return pd.DataFrame()
    
    # Convert to numeric
    df_train[pi_col] = pd.to_numeric(df_train[pi_col], errors="coerce")
    df_train = df_train.dropna(subset=[pi_col])
    
    if len(df_train) == 0:
        print("Error: No valid pi values found")
        return pd.DataFrame()
    
    # Calculate P(E_i; N)
    pi_values = df_train[pi_col].values.astype(float)
    p_extracted = prob_extracted_at_least_once(pi_values, float(budget))
    
    df_train = df_train.copy()
    df_train['p_extracted'] = p_extracted
    df_train['pi'] = pi_values
    
    # Sort by extraction probability (descending)
    df_train = df_train.sort_values('p_extracted', ascending=False)
    
    # Select expected names based on method
    if threshold_method == "expected_count":
        # Expected number of unique names = sum of P(E_i; N)
        expected_count = np.sum(p_extracted)
        k = int(np.ceil(expected_count))
        df_expected = df_train.head(k).copy()
        print(f"Expected unique names count: {expected_count:.2f}")
        print(f"Selected top {k} names by P(E_i; {budget}) (rounded up from expected count)")
    elif threshold_method == "probability":
        if threshold_value is None:
            threshold_value = 0.5
        df_expected = df_train[df_train['p_extracted'] >= threshold_value].copy()
        print(f"Selected {len(df_expected)} names with P(E_i; {budget}) >= {threshold_value}")
    elif threshold_method == "top_k":
        if threshold_value is None:
            raise ValueError("threshold_value required for top_k method")
        k = int(threshold_value)
        df_expected = df_train.head(k).copy()
        print(f"Selected top {k} names by P(E_i; {budget})")
    elif threshold_method == "percentile":
        if threshold_value is None:
            threshold_value = 50
        percentile_val = np.percentile(p_extracted, threshold_value)
        df_expected = df_train[df_train['p_extracted'] >= percentile_val].copy()
        print(f"Selected {len(df_expected)} names above {threshold_value}th percentile (P >= {percentile_val:.6f})")
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    return df_expected


def compare_found_vs_expected(found_file, scores_file, budget=10**7, 
                              pi_col="p_ft_Name: ", split_col="split_x",
                              threshold_method="probability", threshold_value=0.5):
    """
    Compare found names with expected names based on theoretical probabilities.
    
    Args:
        found_file: Path to CSV with found names (values_found_name_train.csv)
        scores_file: Path to CSV with theoretical scores and probabilities
        budget: Budget N (default: 10^7)
        pi_col: Column name for per-draw extraction probability
        split_col: Column name for split
        threshold_method: Method to select expected names
        threshold_value: Threshold value for selection
    
    Returns:
        Dictionary with comparison results
    """
    print("="*80)
    print("Comparing Found Names vs Expected Names (Theoretical)")
    print("="*80)
    
    # Load found names
    print(f"\nLoading found names from: {found_file}")
    if not os.path.exists(found_file):
        print(f"Error: File not found: {found_file}")
        return None
    
    df_found = pd.read_csv(found_file)
    print(f"  Loaded {len(df_found)} rows")
    print(f"  Columns: {df_found.columns.tolist()}")
    
    if 'value' not in df_found.columns:
        print("Error: 'value' column not found in found names file")
        return None
    
    found_names = set(df_found['value'].dropna().unique())
    print(f"  Unique found names: {len(found_names)}")
    
    # Load scores file
    print(f"\nLoading scores file from: {scores_file}")
    if not os.path.exists(scores_file):
        print(f"Error: File not found: {scores_file}")
        return None
    
    df_scores = pd.read_csv(scores_file)
    print(f"  Loaded {len(df_scores)} rows")
    print(f"  Columns: {df_scores.columns.tolist()}")
    
    # Get expected names
    print(f"\nCalculating expected names for budget N = {budget}")
    df_expected = get_expected_names(
        df_scores, 
        budget=budget,
        pi_col=pi_col,
        split_col=split_col,
        threshold_method=threshold_method,
        threshold_value=threshold_value
    )
    
    if len(df_expected) == 0:
        print("Error: No expected names found")
        return None
    
    # Get expected name values
    if 'value' not in df_expected.columns:
        print("Error: 'value' column not found in expected names")
        return None
    
    expected_names = set(df_expected['value'].dropna().unique())
    print(f"  Unique expected names: {len(expected_names)}")
    
    # Compare
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    
    # Names found but not expected (false positives from theory perspective)
    found_not_expected = found_names - expected_names
    print(f"\nFound but NOT expected: {len(found_not_expected)}")
    
    # Names expected but not found (missed names)
    expected_not_found = expected_names - found_names
    print(f"Expected but NOT found: {len(expected_not_found)}")
    
    # Names in both
    both = found_names & expected_names
    print(f"Found AND expected: {len(both)}")
    
    # Calculate statistics
    total_expected = len(expected_names)
    total_found = len(found_names)
    recall = len(both) / total_expected if total_expected > 0 else 0
    precision = len(both) / total_found if total_found > 0 else 0
    
    print(f"\nStatistics:")
    print(f"  Total expected: {total_expected}")
    print(f"  Total found: {total_found}")
    print(f"  Recall (found/expected): {recall:.4f} ({len(both)}/{total_expected})")
    print(f"  Precision (expected/found): {precision:.4f} ({len(both)}/{total_found})")
    
    # Prepare all train names with pi and p_extracted for lookup
    print(f"\nPreparing probability data for all train names...")
    if split_col in df_scores.columns:
        df_all_train = df_scores[df_scores[split_col].astype(str).str.lower() == "train"].copy()
    else:
        df_all_train = df_scores.copy()
    
    if pi_col in df_all_train.columns and 'value' in df_all_train.columns:
        df_all_train[pi_col] = pd.to_numeric(df_all_train[pi_col], errors="coerce")
        df_all_train = df_all_train.dropna(subset=[pi_col, 'value'])
        pi_values_all = df_all_train[pi_col].values.astype(float)
        p_extracted_all = prob_extracted_at_least_once(pi_values_all, float(budget))
        df_all_train = df_all_train.copy()
        df_all_train['pi'] = pi_values_all
        df_all_train['p_extracted'] = p_extracted_all
        # Keep unique values (one row per name)
        df_all_train_lookup = df_all_train[['value', 'pi', 'p_extracted']].drop_duplicates(subset=['value'])
    else:
        df_all_train_lookup = pd.DataFrame(columns=['value', 'pi', 'p_extracted'])
    
    # Create detailed comparison DataFrames
    df_found_not_expected = pd.DataFrame({
        'value': list(found_not_expected)
    })
    
    # Add pi and p_extracted for found_not_expected names
    if len(df_found_not_expected) > 0 and len(df_all_train_lookup) > 0:
        df_found_not_expected = df_found_not_expected.merge(
            df_all_train_lookup,
            on='value',
            how='left'
        )
        # Sort by p_extracted (descending), with NaN values at the end
        df_found_not_expected = df_found_not_expected.sort_values(
            'p_extracted', 
            ascending=False, 
            na_position='last'
        )
        # Report how many have probability data
        has_prob = df_found_not_expected['p_extracted'].notna().sum()
        print(f"  Found {has_prob}/{len(df_found_not_expected)} found_not_expected names in scores file")
    
    df_expected_not_found = df_expected[df_expected['value'].isin(expected_not_found)].copy()
    if len(df_expected_not_found) > 0:
        df_expected_not_found = df_expected_not_found[['value', 'pi', 'p_extracted']].drop_duplicates(subset=['value'])
        df_expected_not_found = df_expected_not_found.sort_values('p_extracted', ascending=False)
    
    df_both = df_expected[df_expected['value'].isin(both)].copy()
    if len(df_both) > 0:
        df_both = df_both[['value', 'pi', 'p_extracted']].drop_duplicates(subset=['value'])
        df_both = df_both.sort_values('p_extracted', ascending=False)
    
    # Save results
    output_dir = os.path.dirname(found_file)
    results = {
        'found_not_expected': df_found_not_expected,
        'expected_not_found': df_expected_not_found,
        'both': df_both,
        'stats': {
            'total_expected': total_expected,
            'total_found': total_found,
            'found_and_expected': len(both),
            'found_not_expected_count': len(found_not_expected),
            'expected_not_found_count': len(expected_not_found),
            'recall': recall,
            'precision': precision,
        }
    }
    
    # Save to files
    if len(df_found_not_expected) > 0:
        output_file = os.path.join(output_dir, "found_not_expected.csv")
        df_found_not_expected.to_csv(output_file, index=False)
        print(f"\nSaved found but not expected names to: {output_file}")
        
        # Print summary of found but not expected
        if 'p_extracted' in df_found_not_expected.columns:
            df_with_prob = df_found_not_expected[df_found_not_expected['p_extracted'].notna()]
            if len(df_with_prob) > 0:
                print(f"\nSummary of found but not expected (with probability data):")
                print(f"  Count: {len(df_with_prob)}")
                print(f"  Mean pi: {df_with_prob['pi'].mean():.6f}")
                print(f"  Mean P(E_i; {budget}): {df_with_prob['p_extracted'].mean():.6f}")
                print(f"  Min P(E_i; {budget}): {df_with_prob['p_extracted'].min():.6f}")
                print(f"  Max P(E_i; {budget}): {df_with_prob['p_extracted'].max():.6f}")
                print(f"\nTop 20 found but not expected names (by P(E_i; N)):")
                print(df_with_prob.head(20)[['value', 'pi', 'p_extracted']].to_string(index=False))
    
    if len(df_expected_not_found) > 0:
        output_file = os.path.join(output_dir, "expected_not_found.csv")
        df_expected_not_found.to_csv(output_file, index=False)
        print(f"Saved expected but not found names to: {output_file}")
        print(f"\nTop 20 expected but not found names (by P(E_i; N)):")
        print(df_expected_not_found.head(20)[['value', 'pi', 'p_extracted']].to_string(index=False))
    
    if len(df_both) > 0:
        output_file = os.path.join(output_dir, "found_and_expected.csv")
        df_both.to_csv(output_file, index=False)
        print(f"Saved found and expected names to: {output_file}")
    
    # Print summary of expected but not found
    if len(df_expected_not_found) > 0:
        print(f"\nSummary of expected but not found:")
        print(f"  Mean pi: {df_expected_not_found['pi'].mean():.6f}")
        print(f"  Mean P(E_i; {budget}): {df_expected_not_found['p_extracted'].mean():.6f}")
        print(f"  Min P(E_i; {budget}): {df_expected_not_found['p_extracted'].min():.6f}")
        print(f"  Max P(E_i; {budget}): {df_expected_not_found['p_extracted'].max():.6f}")
    
    return results


def main():
    # Default file paths
    found_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_10000000/values_found_name_train.csv"
    scores_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_10000000/scores_1B_10_pii_rate_0.1_n_epochs_3_p.csv"
    budget = 10**7
    pi_col = "p_ft_Name: "
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        found_file = sys.argv[1]
        scores_file = sys.argv[2]
    if len(sys.argv) >= 4:
        budget = float(sys.argv[3])
    if len(sys.argv) >= 5:
        pi_col = sys.argv[4]
    
    if len(sys.argv) < 3:
        print("Usage: python compare_found_vs_expected_names.py [found_file] [scores_file] [budget] [pi_col]")
        print("Using default paths...")
    
    # Compare
    results = compare_found_vs_expected(
        found_file=found_file,
        scores_file=scores_file,
        budget=budget,
        pi_col=pi_col,
        threshold_method="expected_count",  # Use expected count method by default
        threshold_value=None
    )
    
    if results is None:
        print("\nComparison failed")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Comparison completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
