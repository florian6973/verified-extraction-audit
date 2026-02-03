from src._repo import REPO_ROOT
"""
Create summary plots for theoretical curves.
Plots total recall, extracted FPR, and extracted TPR as a function of budget.
All models are plotted in the same subplots with proper legends.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, PercentFormatter
from matplotlib.collections import LineCollection
import pickle

# Try to import scipy for binomial CI computation
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Base directory containing all experimental recall outputs
# BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output"
# BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output-test"
BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output-all-large"


# Path to threshold_extracted_fpr0.05_results.csv
THRESHOLD_CSV = os.path.join(BASE_DIR, "threshold_extracted_fpr0.05_results.csv")

def parse_directory_name(dir_name):
    """
    Parse directory name to extract parameters.
    Format: {dataset_size}_{model_size}_{pii_rate}_{n_epochs}_{k}
    Example: 10_1B_0.05_3_10000
    """
    parts = dir_name.split('_')
    if len(parts) >= 4:
        try:
            dataset_size = int(parts[0])
            model_size = parts[1]
            pii_rate = float(parts[2])
            n_epochs = int(parts[3])
            k = int(parts[4]) if len(parts) > 4 else None
            return {
                'dataset_size': dataset_size,
                'model_size': model_size,
                'pii_rate': pii_rate,
                'n_epochs': n_epochs,
                'k': k
            }
        except (ValueError, IndexError):
            return None
    return None

def load_threshold_data(csv_path):
    """
    Load threshold_extracted_fpr0.05_results.csv and return a dictionary.
    
    Returns:
        Dictionary: {directory_name: {'avg_threshold': float, 'params': dict}}
    """
    threshold_map = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: threshold_extracted_fpr0.05_results.csv not found at {csv_path}")
        return threshold_map
    
    try:
        df = pd.read_csv(csv_path)
        if 'directory' in df.columns and 'avg_threshold' in df.columns:
            for _, row in df.iterrows():
                directory = row['directory']
                avg_threshold = row['avg_threshold']
                params = parse_directory_name(directory)
                
                threshold_map[directory] = {
                    'avg_threshold': avg_threshold,
                    'params': params
                }
            print(f"Loaded {len(threshold_map)} entries from threshold_extracted_fpr0.05_results.csv")
        else:
            print(f"Warning: threshold_extracted_fpr0.05_results.csv missing required columns")
    except Exception as e:
        print(f"Error reading threshold_extracted_fpr0.05_results.csv: {e}")
    
    return threshold_map

def collect_theoretical_data(base_dir, threshold_map, threshold_df=None, use_filtered_exp=False):
    """
    Collect theoretical curve data for all directories.
    
    Args:
        base_dir: Base directory containing all experimental outputs
        threshold_map: Dictionary mapping directory names to threshold info
        threshold_df: Optional DataFrame with threshold CSV data (for file paths)
    
    Returns:
        List of dictionaries with theoretical data
    """
    all_data = []
    
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    for directory, info in threshold_map.items():
        if '3' not in directory:
            continue
        # if not (('3' in directory or '2' in directory) and ('0.01' in directory or '1.0' in directory)):
            # continue

        avg_threshold = info['avg_threshold']
        params = info['params']
        
        if params is None:
            print(f"Warning: Could not parse directory name: {directory}")
            continue
        
        # Debug: Check if budget (k) is extracted correctly
        if params.get('k') is None:
            print(f"Warning: Directory {directory} does not have budget (k) parameter in 5th position")
            print(f"  Expected format: {{dataset_size}}_{{model_size}}_{{pii_rate}}_{{n_epochs}}_{{k}}")
            print(f"  Example: 10_1B_0.1_3_500000")
        
        # Look for theoretical curves file
        plots_dir = base_path / directory / "plots_theory"
        theory_file = plots_dir / f"theoretical_curves_tau_{avg_threshold}.csv"
        
        if not theory_file.exists():
            print(f"Warning: Theoretical curves file not found: {theory_file}")
            continue
        
        print(f"Processing {directory} (tau={avg_threshold:.6f})...")
        
        try:
            df_theory = pd.read_csv(theory_file)
            
            # Filter out infinite budgets (asymptote rows)
            df_finite = df_theory[np.isfinite(df_theory['budget'])].copy()
            
            if len(df_finite) == 0:
                print(f"  Warning: No finite budgets found in {theory_file}")
                continue
            
            # Extract the columns we need (with verification only)
            if 'recall_with_verification' not in df_finite.columns:
                print(f"  Warning: Missing 'recall_with_verification' column in {theory_file}")
                continue
            if 'fpr_extracted_with_verification' not in df_finite.columns:
                print(f"  Warning: Missing 'fpr_extracted_with_verification' column in {theory_file}")
                continue
            if 'tpr_extracted_with_verification' not in df_finite.columns:
                print(f"  Warning: Missing 'tpr_extracted_with_verification' column in {theory_file}")
                continue
            if 'precision_with_verification' not in df_finite.columns:
                print(f"  Warning: Missing 'precision_with_verification' column in {theory_file}")
                # Continue anyway, precision is optional
            
            # Also get recall without verification for extraction plot
            recall_without_verification = None
            if 'recall_without_verification' in df_finite.columns:
                recall_without_verification = df_finite['recall_without_verification'].values
            else:
                print(f"  Warning: Missing 'recall_without_verification' column in {theory_file}")
                recall_without_verification = np.full(len(df_finite), np.nan)
            
            # Load scores file to get total counts and compute cumulative FP
            total_train_names = None
            total_val_names = None
            cumulative_fp_data = None
            
            # Load experimental (single-budget) results for this directory at tau=avg_threshold.
            # The budget (k) is extracted from the directory name.
            exp_tp = None
            exp_fp = None
            exp_total_recall = None
            exp_tpr = None
            exp_fpr = None
            exp_precision = None
            exp_extraction_recall = None  # Extraction recall without verification
            exp_budget = params.get('k')  # Budget from directory name
            
            # Error bars (CI bounds) for experimental points
            exp_total_recall_err = None
            exp_tpr_err = None
            exp_fpr_err = None
            exp_precision_err = None
            exp_tp_err = None
            exp_fp_err = None
            exp_extraction_recall_err = None
            
            try:
                # Choose which experimental metrics file to use
                if use_filtered_exp:
                    exp_metrics_filename = "all_names_ll_computed_with_scores_metrics_by_threshold_without_others.csv"
                    exp_metrics_bootstrap_filename = "all_names_ll_computed_with_scores_metrics_by_threshold_with_bootstrap_without_others.csv"
                else:
                    exp_metrics_filename = "all_names_ll_computed_with_scores_metrics_by_threshold.csv"
                    exp_metrics_bootstrap_filename = "all_names_ll_computed_with_scores_metrics_by_threshold_with_bootstrap.csv"
                
                exp_metrics_path = base_path / directory / exp_metrics_filename
                exp_metrics_bootstrap_path = base_path / directory / exp_metrics_bootstrap_filename
                
                # Try to load bootstrap file first (has CI columns), fall back to regular file
                df_exp_metrics = None
                has_bootstrap = False
                
                if exp_metrics_bootstrap_path.exists():
                    try:
                        df_exp_metrics = pd.read_csv(exp_metrics_bootstrap_path)
                        has_bootstrap = True
                        print(f"  Loaded bootstrap metrics file: {exp_metrics_bootstrap_path.name}")
                        # Check if CI columns are present
                        ci_cols = [col for col in df_exp_metrics.columns if '_ci_lower' in col or '_ci_upper' in col]
                        if ci_cols:
                            print(f"  Found {len(ci_cols)} CI columns in bootstrap file")
                        else:
                            print(f"  Warning: Bootstrap file exists but no CI columns found")
                    except Exception as e:
                        print(f"  Warning: Could not load bootstrap metrics file: {e}")
                else:
                    print(f"  Bootstrap file not found: {exp_metrics_bootstrap_path.name}")
                
                if df_exp_metrics is None and exp_metrics_path.exists():
                    df_exp_metrics = pd.read_csv(exp_metrics_path)
                    has_bootstrap = False
                    print(f"  Loaded regular metrics file (no bootstrap): {exp_metrics_path.name}")
                
                if df_exp_metrics is not None and 'threshold' in df_exp_metrics.columns and 'TP' in df_exp_metrics.columns and 'FP' in df_exp_metrics.columns:
                    thr_vals = pd.to_numeric(df_exp_metrics['threshold'], errors='coerce').values
                    # Pick exact match if possible, otherwise nearest threshold
                    if np.any(np.isfinite(thr_vals)):
                        diffs = np.abs(thr_vals - float(avg_threshold))
                        idx = int(np.nanargmin(diffs))
                        exp_tp = float(df_exp_metrics.iloc[idx]['TP'])
                        exp_fp = float(df_exp_metrics.iloc[idx]['FP'])
                        # Also capture rate metrics if present (for the first theoretical plot)
                        if 'total_recall' in df_exp_metrics.columns:
                            val = df_exp_metrics.iloc[idx]['total_recall']
                            if pd.notna(val):
                                exp_total_recall = float(val)
                            else:
                                print(f"  Warning: total_recall is NaN at threshold {avg_threshold:.6f} for {directory}")
                        if 'TPR' in df_exp_metrics.columns:
                            val = df_exp_metrics.iloc[idx]['TPR']
                            if pd.notna(val):
                                exp_tpr = float(val)
                        if 'FPR' in df_exp_metrics.columns:
                            val = df_exp_metrics.iloc[idx]['FPR']
                            if pd.notna(val):
                                exp_fpr = float(val)
                        if 'precision' in df_exp_metrics.columns:
                            val = df_exp_metrics.iloc[idx]['precision']
                            if pd.notna(val):
                                exp_precision = float(val)
                        
                        # Load error bars from bootstrap CI columns if available
                        if has_bootstrap:
                            # Calculate error bars as distance from mean to CI bounds
                            # For errorbar, we need [lower_error, upper_error] format
                            if 'total_recall_ci_lower' in df_exp_metrics.columns and 'total_recall_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['total_recall_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['total_recall_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_total_recall_err = [exp_total_recall - ci_lower, ci_upper - exp_total_recall]
                            
                            if 'TPR_ci_lower' in df_exp_metrics.columns and 'TPR_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['TPR_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['TPR_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_tpr_err = [exp_tpr - ci_lower, ci_upper - exp_tpr]
                            
                            if 'FPR_ci_lower' in df_exp_metrics.columns and 'FPR_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['FPR_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['FPR_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_fpr_err = [exp_fpr - ci_lower, ci_upper - exp_fpr]
                            
                            if 'precision_ci_lower' in df_exp_metrics.columns and 'precision_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['precision_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['precision_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_precision_err = [exp_precision - ci_lower, ci_upper - exp_precision]
                            
                            if 'TP_ci_lower' in df_exp_metrics.columns and 'TP_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['TP_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['TP_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_tp_err = [exp_tp - ci_lower, ci_upper - exp_tp]
                            
                            if 'FP_ci_lower' in df_exp_metrics.columns and 'FP_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['FP_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['FP_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                                    exp_fp_err = [exp_fp - ci_lower, ci_upper - exp_fp]
                            
                            # Check for extraction recall CI columns (if available)
                            if 'extraction_recall_ci_lower' in df_exp_metrics.columns and 'extraction_recall_ci_upper' in df_exp_metrics.columns:
                                ci_lower = df_exp_metrics.iloc[idx]['extraction_recall_ci_lower']
                                ci_upper = df_exp_metrics.iloc[idx]['extraction_recall_ci_upper']
                                if not pd.isna(ci_lower) and not pd.isna(ci_upper) and exp_extraction_recall is not None:
                                    exp_extraction_recall_err = [exp_extraction_recall - ci_lower, ci_upper - exp_extraction_recall]
                        
                        # Warn if nearest is far (helps debugging mismatched tau formatting)
                        if diffs[idx] > 1e-6:
                            print(f"  Note: experimental threshold nearest to tau (|Δ|={diffs[idx]:.2e}) in {exp_metrics_path.name}")
                else:
                    print(f"  Warning: experimental metrics missing required columns in {exp_metrics_path}")
                    if df_exp_metrics is not None:
                        print(f"    Available columns: {df_exp_metrics.columns.tolist()}")
                if df_exp_metrics is None:
                    print(f"  Warning: experimental metrics file not found: {exp_metrics_path}")
                    print(f"    Looking for: {exp_metrics_filename} or {exp_metrics_bootstrap_filename}")
            except Exception as e:
                print(f"  Warning: could not load experimental metrics for {directory}: {e}")
                import traceback
                traceback.print_exc()
            
            # Try to get file path from threshold_df if available
            scores_file_path = None
            if threshold_df is not None:
                matching_rows = threshold_df[threshold_df['directory'] == directory]
                if len(matching_rows) > 0 and 'file_path' in matching_rows.columns:
                    scores_file_path = matching_rows.iloc[0]['file_path']
            
            # If not found in threshold_df, try to find it in the directory
            if scores_file_path is None or not os.path.exists(scores_file_path):
                scores_dir = base_path / directory
                # Try to find scores file with pattern scores_*_p.csv
                scores_files = list(scores_dir.glob("scores_*_p.csv"))
                if len(scores_files) > 0:
                    scores_file_path = str(scores_files[0])
            
            if scores_file_path and os.path.exists(scores_file_path):
                try:
                    df_scores = pd.read_csv(scores_file_path)
                    if 'split_x' in df_scores.columns:
                        total_train_names = len(df_scores[df_scores['split_x'] == 'train'])
                        total_val_names = len(df_scores[df_scores['split_x'] == 'val'])
                        print(f"  Found counts: train={total_train_names}, val={total_val_names}")
                        
                        # Compute cumulative false positives
                        # Get non-members (val)
                        nonmembers = df_scores[df_scores['split_x'] == 'val'].copy()
                        
                        # Get pi values and q values
                        pi_col = "p_ft_Name: "
                        score_col = "score_oof_member_proba"
                        
                        if pi_col in nonmembers.columns and score_col in nonmembers.columns:
                            pi_nm = pd.to_numeric(nonmembers[pi_col], errors='coerce').fillna(0).values
                            scores_nm = pd.to_numeric(nonmembers[score_col], errors='coerce').fillna(0).values
                            q_nm = (scores_nm >= avg_threshold).astype(float)
                            
                            # Compute cumulative FP for each budget
                            budgets = df_finite['budget'].values
                            cumulative_fp = []
                            
                            # Also get members to compute expected TP for verification
                            members = df_scores[df_scores['split_x'] == 'train'].copy()
                            pi_mem = pd.to_numeric(members[pi_col], errors='coerce').fillna(0).values if pi_col in members.columns else None
                            scores_mem = pd.to_numeric(members[score_col], errors='coerce').fillna(0).values if score_col in members.columns else None
                            q_mem = (scores_mem >= avg_threshold).astype(float) if scores_mem is not None else None
                            
                            # Compute extraction recall without verification for experimental data
                            # Use values_found_name_train_with_ll.csv which contains actual extracted train names
                            if total_train_names is not None and total_train_names > 0:
                                exp_extraction_file = base_path / directory / "values_found_name_train_with_ll.csv"
                                if exp_extraction_file.exists():
                                    try:
                                        df_extracted_train = pd.read_csv(exp_extraction_file)
                                        if 'value' in df_extracted_train.columns:
                                            unique_extracted_train = df_extracted_train['value'].nunique()
                                            exp_extraction_recall = unique_extracted_train / total_train_names
                                            print(f"  Experimental extraction recall (without verification): {unique_extracted_train}/{total_train_names} = {exp_extraction_recall:.6f}")
                                            
                                            # Compute binomial CI as fallback if CI columns don't exist in metrics file
                                            # This provides uncertainty bars even if bootstrap CI columns aren't available
                                            if exp_extraction_recall_err is None and unique_extracted_train > 0:
                                                # Use Clopper-Pearson (exact) interval if scipy available, otherwise normal approximation
                                                if HAS_SCIPY:
                                                    # Clopper-Pearson exact interval (95% CI)
                                                    # Lower: beta.ppf(0.025, k, n-k+1), Upper: beta.ppf(0.975, k+1, n-k)
                                                    k = unique_extracted_train
                                                    n = total_train_names
                                                    ci_lower = stats.beta.ppf(0.025, k, n - k + 1)
                                                    ci_upper = stats.beta.ppf(0.975, k + 1, n - k)
                                                    exp_extraction_recall_err = [exp_extraction_recall - ci_lower, ci_upper - exp_extraction_recall]
                                                else:
                                                    # Normal approximation (95% CI)
                                                    p = exp_extraction_recall
                                                    n = total_train_names
                                                    se = np.sqrt(p * (1 - p) / n)
                                                    z = 1.96  # 95% CI
                                                    ci_lower = max(0, p - z * se)
                                                    ci_upper = min(1, p + z * se)
                                                    exp_extraction_recall_err = [exp_extraction_recall - ci_lower, ci_upper - exp_extraction_recall]
                                        else:
                                            print(f"  Warning: 'value' column not found in {exp_extraction_file}")
                                    except Exception as e:
                                        print(f"  Warning: Could not load extraction recall from {exp_extraction_file}: {e}")
                                else:
                                    print(f"  Warning: values_found_name_train_with_ll.csv not found: {exp_extraction_file}")
                            
                            for N in budgets:
                                # P(E_i;N) = 1 - (1 - pi_i)^N
                                pE_nm = 1 - np.power(1 - pi_nm, N)
                                # Cumulative FP = sum P(E_i;N) * q_i
                                cum_fp = np.sum(pE_nm * q_nm)
                                cumulative_fp.append(cum_fp)
                            
                            cumulative_fp_data = np.array(cumulative_fp)
                            
                            # Debug: Check a sample budget point and verify against theoretical values
                            sample_idx = min(10, len(budgets) - 1) if len(budgets) > 0 else 0
                            if len(budgets) > 0 and len(df_finite) > sample_idx:
                                sample_budget = budgets[sample_idx]
                                sample_fp = cumulative_fp[sample_idx]
                                # Also compute expected extracted non-members
                                pE_nm_sample = 1 - np.power(1 - pi_nm, sample_budget)
                                expected_extracted_nm = np.sum(pE_nm_sample)
                                fpr_check = sample_fp / expected_extracted_nm if expected_extracted_nm > 0 else 0
                                
                                # Get theoretical FPR for comparison
                                theoretical_fpr = df_finite.iloc[sample_idx]['fpr_extracted_with_verification']
                                
                                # Compute expected TP for comparison
                                expected_tp = None
                                if pi_mem is not None and q_mem is not None:
                                    pE_mem_sample = 1 - np.power(1 - pi_mem, sample_budget)
                                    expected_tp = np.sum(pE_mem_sample * q_mem)
                                    theoretical_recall = df_finite.iloc[sample_idx]['recall_with_verification']
                                    expected_tp_from_recall = theoretical_recall * len(members)
                                
                                print(f"  Sample at budget={sample_budget:.0f}:")
                                print(f"    cum_FP={sample_fp:.2f}, expected_extracted_nm={expected_extracted_nm:.2f}")
                                print(f"    FPR_computed={fpr_check:.4f}, FPR_theoretical={theoretical_fpr:.4f}")
                                if expected_tp is not None:
                                    print(f"    expected_TP={expected_tp:.2f}, expected_TP_from_recall={expected_tp_from_recall:.2f}")
                            
                            print(f"  Computed cumulative FP for {len(budgets)} budget points")
                        else:
                            print(f"  Warning: Missing required columns for cumulative FP calculation")
                except Exception as e:
                    print(f"  Warning: Could not load/process scores file {scores_file_path}: {e}")
            else:
                print(f"  Warning: Could not find scores file for {directory}")
            
            # Get precision if available
            precision_data = None
            if 'precision_with_verification' in df_finite.columns:
                precision_data = df_finite['precision_with_verification'].values
            else:
                # Fill with NaN if not available
                precision_data = np.full(len(df_finite), np.nan)
            
            all_data.append({
                'directory': directory,
                'dataset_size': params['dataset_size'],
                'model_size': params['model_size'],
                'pii_rate': params['pii_rate'],
                'n_epochs': params['n_epochs'],
                'tau': avg_threshold,
                'budget': df_finite['budget'].values,
                'recall': df_finite['recall_with_verification'].values,
                'recall_without_verification': recall_without_verification,  # For extraction plot
                'fpr_extracted': df_finite['fpr_extracted_with_verification'].values,
                'tpr_extracted': df_finite['tpr_extracted_with_verification'].values,
                'precision': precision_data,
                'total_train_names': total_train_names,
                'total_val_names': total_val_names,
                'cumulative_fp': cumulative_fp_data,
                'exp_budget': exp_budget,  # Budget from directory name
                'exp_tp': exp_tp,
                'exp_fp': exp_fp,
                'exp_total_recall': exp_total_recall,
                'exp_tpr': exp_tpr,
                'exp_fpr': exp_fpr,
                'exp_precision': exp_precision,
                'exp_extraction_recall': exp_extraction_recall,  # Extraction recall without verification
                'exp_total_recall_err': exp_total_recall_err,
                'exp_tpr_err': exp_tpr_err,
                'exp_fpr_err': exp_fpr_err,
                'exp_precision_err': exp_precision_err,
                'exp_tp_err': exp_tp_err,
                'exp_fp_err': exp_fp_err,
                'exp_extraction_recall_err': exp_extraction_recall_err,
            })
            
            print(f"  Loaded {len(df_finite)} budget points")
            
        except Exception as e:
            print(f"Error processing {directory}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No theoretical data collected. Check that theoretical curve files exist.")
    
    return all_data

def create_summary_plots(theory_data, output_path=None, use_filtered_exp=False, show_cost=False, short=False, annotate=False, hide_tau_in_legend=False):
    """
    Create summary plots organized by (dataset_size, model_size).
    Each row represents one (dataset_size, model_size) combination with 4 subplots:
    Total Recall, Extracted FPR, Extracted TPR, and Precision.
    
    Args:
        theory_data: List of theoretical data dictionaries
        output_path: Optional path to save the plot
        use_filtered_exp: Whether to use filtered experimental metrics
        show_cost: Whether to show cost axis below budget axis
        short: If True, only display curves between 10^2 and 10^9 budget
        annotate: If True, display number of names leaked (exp_tp) or false positives (exp_fp) near experimental dots
    """
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    print(f"\nFound {len(ds_model_list)} unique (dataset_size, model_size) combinations")
    print("Groups:")
    for ds, model in ds_model_list:
        print(f"  {ds}_{model}")
    
    # Get unique PII rates for color gradient assignment
    pii_rates = sorted(set(data['pii_rate'] for data in theory_data))
    
    # Use a red-based colormap with evenly spaced colors (based on order, not PII value)
    # Start from 0.5 to avoid pale colors, go to 1.0 for dark red
    pii_colors = {}
    n_pii = len(pii_rates)
    if n_pii > 0:
        for idx, pii_rate in enumerate(pii_rates):
            # Evenly space colors: first gets 0.5, last gets 1.0
            if n_pii == 1:
                colormap_val = 0.75
            else:
                colormap_val = 0.5 + 0.5 * (idx / (n_pii - 1))
            pii_colors[pii_rate] = plt.cm.Reds(colormap_val)
    
    # Create figure with rows for each (dataset_size, model_size) and 4 columns
    n_rows = len(ds_model_list)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each (dataset_size, model_size) group
    for row_idx, (ds_size, model) in enumerate(ds_model_list):
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        one_category = len({epoch_category_label(d['n_epochs']) for d in group_data}) <= 1
        
        # Plot 1: Total Recall vs budget
        ax = axes[row_idx, 0]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            # Format legend: epochs -> overfit/no overfit, PII -> percentage
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, recall_plot = filter_budget_data(data['budget'], data['recall'], min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                recall_plot = data['recall']
            
            ax.plot(budget_plot, recall_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_total_recall')
            y_err = data.get('exp_total_recall_err')
            # Debug output for missing experimental points
            if exp_budget is not None and (y_exp is None or not np.isfinite(y_exp)):
                print(f"  [Total Recall Plot] Warning: Experimental point at budget={exp_budget} missing value (y_exp={y_exp}) for {data.get('directory', 'unknown')}")
            if exp_budget is None:
                print(f"  [Total Recall Plot] Warning: No exp_budget found for {data.get('directory', 'unknown')} (params: {data.get('dataset_size')}_{data.get('model_size')}_{data.get('pii_rate')}_{data.get('n_epochs')})")
            if y_exp is not None and np.isfinite(y_exp) and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for names leaked if requested
                if annotate:
                    exp_tp = data.get('exp_tp')
                    if exp_tp is not None and np.isfinite(exp_tp):
                        ax.annotate(f'{int(exp_tp)}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for titles
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('True names extracted (% of finetuning set)', fontsize=12)
        ax.set_title(f'Total Recall vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1))  # display 0-1 as 0%-100%
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Extracted FPR vs budget
        ax = axes[row_idx, 1]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            # Format legend: epochs -> overfit/no overfit, PII -> percentage
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, fpr_plot = filter_budget_data(data['budget'], data['fpr_extracted'], min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                fpr_plot = data['fpr_extracted']
            
            ax.plot(budget_plot, fpr_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_fpr')
            y_err = data.get('exp_fpr_err')
            if y_exp is not None and np.isfinite(y_exp) and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for false positives if requested
                if annotate:
                    exp_fp = data.get('exp_fp')
                    if exp_fp is not None and np.isfinite(exp_fp):
                        ax.annotate(f'{int(exp_fp)}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Extracted FPR', fontsize=12)
        ax.set_title(f'Extracted FPR vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(bottom=-0.01)
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Extracted TPR vs budget
        ax = axes[row_idx, 2]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            # Format legend: epochs -> overfit/no overfit, PII -> percentage
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, tpr_plot = filter_budget_data(data['budget'], data['tpr_extracted'], min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                tpr_plot = data['tpr_extracted']
            
            ax.plot(budget_plot, tpr_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_tpr')
            y_err = data.get('exp_tpr_err')
            if y_exp is not None and np.isfinite(y_exp) and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Extracted TPR', fontsize=12)
        ax.set_title(f'Extracted TPR vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(bottom=-0.01)
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Precision vs budget
        ax = axes[row_idx, 3]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            # Format legend: epochs -> overfit/no overfit, PII -> percentage
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Check if precision data is available (not all NaN)
            if data['precision'] is not None and not np.all(np.isnan(data['precision'])):
                # Filter budget data if short mode is enabled
                if short:
                    budget_plot, precision_plot = filter_budget_data(data['budget'], data['precision'], min_budget=1e2, max_budget=1e9)
                else:
                    budget_plot = data['budget']
                    precision_plot = data['precision']
                
                ax.plot(budget_plot, precision_plot,
                        color=color, linestyle=linestyle,
                        marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_precision')
            y_err = data.get('exp_precision_err')
            if y_exp is not None and np.isfinite(y_exp) and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for names leaked if requested (precision plot shows TP, so show TP)
                if annotate:
                    exp_tp = data.get('exp_tp')
                    if exp_tp is not None and np.isfinite(exp_tp):
                        ax.annotate(f'{int(exp_tp)}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(bottom=-0.01, top=1.05)
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Theoretical Curves Summary (With Verification)', fontsize=18, fontweight='bold', y=0.995)
    
    # Adjust bottom margin if cost axis is shown (cost labels are at bottom)
    if show_cost:
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])  # More space at bottom for cost labels
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if output_path is None:
        suffix = "_filtered" if use_filtered_exp else ""
        output_path = os.path.join(BASE_DIR, f"summary_theo_plot{suffix}.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved summary plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved summary plot to: {pdf_path}")
    
    return fig

def get_dataset_size_label(ds_size):
    """Convert dataset size number to label."""
    if ds_size == 1:
        return "small dataset"
    elif ds_size == 10:
        return "medium dataset"
    elif ds_size == 100:
        return "large dataset"
    else:
        return f"ds={ds_size}"

def deduplicate_legend(handles, labels):
    """
    Remove duplicate legend entries based on label text.
    Keeps the first occurrence of each unique label.
    
    Args:
        handles: List of legend handles
        labels: List of legend labels
    
    Returns:
        Tuple of (deduplicated_handles, deduplicated_labels)
    """
    seen_labels = {}
    dedup_handles = []
    dedup_labels = []
    
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            seen_labels[label] = True
            dedup_handles.append(handle)
            dedup_labels.append(label)
    
    return dedup_handles, dedup_labels

def epoch_category_label(n_ep: int) -> str:
    """Map epoch count to legend category."""
    if n_ep in [2, 3]:
        return "no overfit"
    if n_ep == 10:
        return "overfit"
    return f"ep={n_ep}"

def format_tau_str(tau_val: float) -> str:
    """Format tau for legend display."""
    if tau_val < 1:
        return f"τ={tau_val:.4f}"
    return f"τ={tau_val:.2f}"

def format_theory_legend_label(
    *,
    n_ep: int,
    pii_rate: float,
    tau_val: float,
    include_category: bool = True,
    include_tau: bool = True,
) -> str:
    """
    Build legend label for theoretical (estimated) curves.

    Rules:
    - Always start with "Estimated".
    - If only one category is plotted, category can be omitted by setting include_category=False.
    - Tau can be hidden via include_tau=False.
    - When include_tau=False and include_category=False: "Estimated, PII=%x" (requested format).
    """
    parts = ["Estimated"]
    if include_category:
        parts.append(epoch_category_label(n_ep))
    parts.append(f"PII={pii_rate * 100:.0f}%")
    label = ", ".join(parts)
    if include_tau:
        label += f" ({format_tau_str(tau_val)})"
    return label

def filter_budget_data(budget, *y_arrays, min_budget=1e2, max_budget=1e9):
    """
    Filter budget and corresponding y-value arrays to only include data within budget range.
    
    Args:
        budget: Array of budget values
        *y_arrays: Variable number of y-value arrays to filter (e.g., recall, fpr, tpr)
        min_budget: Minimum budget value (default: 10^2)
        max_budget: Maximum budget value (default: 10^9)
    
    Returns:
        Tuple of (filtered_budget, filtered_y1, filtered_y2, ...)
    """
    mask = (budget >= min_budget) & (budget <= max_budget)
    filtered_budget = budget[mask]
    filtered_arrays = tuple(arr[mask] if arr is not None else None for arr in y_arrays)
    return (filtered_budget,) + filtered_arrays

def format_cost_readable(cost):
    """
    Format cost as readable string: $100, $1K, $10K, $100K, $1M, etc.
    
    Args:
        cost: Cost value in dollars
    
    Returns:
        Formatted string like "$100", "$1K", "$10K", etc.
    """
    if cost < 1:
        return f"${cost:.2f}"
    elif cost < 1000:
        return f"${cost:.0f}"
    elif cost < 1_000_000:
        cost_k = cost / 1000
        # Check if it's a whole number to avoid "1.0K", show as "1K" instead
        if abs(cost_k - round(cost_k)) < 1e-6:
            return f"${int(round(cost_k))}K"
        else:
            return f"${cost_k:.1f}K"
    else:
        cost_m = cost / 1_000_000
        # Check if it's a whole number to avoid "1.0M", show as "1M" instead
        if abs(cost_m - round(cost_m)) < 1e-6:
            return f"${int(round(cost_m))}M"
        else:
            return f"${cost_m:.1f}M"

def add_cost_axis(ax, show_cost=False, tokens_per_query=20, cost_per_million_tokens=5):
    """
    Add cost information to the x-axis labels at the bottom.
    When show_cost is True, modifies primary axis labels to show "budget - cost" format.
    
    Args:
        ax: Matplotlib axis object
        show_cost: Whether to show cost axis
        tokens_per_query: Number of tokens per query (default: 20)
        cost_per_million_tokens: Cost per million tokens in dollars (default: 5)
    
    Returns:
        None (modifies primary axis in place)
    """
    if not show_cost:
        return None
    
    # Cost = budget * tokens_per_query * (cost_per_million_tokens / 1,000,000)
    # Example: budget=100, tokens_per_query=20, cost_per_million_tokens=5
    # cost_per_query = 20 * (5 / 1,000,000) = 0.0001
    cost_per_query = tokens_per_query * (cost_per_million_tokens / 1_000_000)
    
    # Store original formatter to get budget labels
    original_formatter = LogFormatterMathtext()
    
    # Create custom formatter that shows both budget and cost
    def format_budget_with_cost(x, pos):
        if x <= 0:
            return ''
        # Get budget label from original formatter (returns something like "$10^{4}$")
        budget_label = original_formatter(x, pos)
        # Extract math content (remove $ delimiters if present)
        if budget_label.startswith('$') and budget_label.endswith('$'):
            budget_math = budget_label[1:-1]
        else:
            budget_math = budget_label
        # Calculate cost
        cost = x * cost_per_query
        # Format cost (already includes $ sign, e.g., "$10K")
        cost_str = format_cost_readable(cost)
        # Return combined label with newline: budget (in math mode) on first line, cost (plain text) on second line
        # Use \n for line break (matplotlib handles this in tick labels)
        return f"${budget_math}$\n{cost_str}"
    
    # Apply the combined formatter to primary axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_budget_with_cost))
    
    # Add extra padding for multi-line tick labels
    ax.tick_params(axis='x', pad=8)
    
    return None

def create_absolute_plots(theory_data, output_path=None, use_filtered_exp=False, show_cost=False, short=False, annotate=False, hide_tau_in_legend=False):
    """
    Create summary plots with absolute values: total names leaked and total false positives.
    Organized by (dataset_size, model_size).
    
    Args:
        theory_data: List of theoretical data dictionaries
        output_path: Optional path to save the plot
        use_filtered_exp: Whether to use filtered experimental metrics
        show_cost: Whether to show cost axis below budget axis
        annotate: If True, display number of names leaked (exp_tp) or false positives (exp_fp) near experimental dots
    """
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    # Get unique PII rates for color gradient assignment
    pii_rates = sorted(set(data['pii_rate'] for data in theory_data))
    
    # Use a red-based colormap with evenly spaced colors (based on order, not PII value)
    # Start from 0.5 to avoid pale colors, go to 1.0 for dark red
    pii_colors = {}
    n_pii = len(pii_rates)
    if n_pii > 0:
        for idx, pii_rate in enumerate(pii_rates):
            # Evenly space colors: first gets 0.5, last gets 1.0
            if n_pii == 1:
                colormap_val = 0.75
            else:
                colormap_val = 0.5 + 0.5 * (idx / (n_pii - 1))
            pii_colors[pii_rate] = plt.cm.Reds(colormap_val)
    
    # Create figure with rows for each (dataset_size, model_size) and 2 columns
    n_rows = len(ds_model_list)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each (dataset_size, model_size) group
    for row_idx, (ds_size, model) in enumerate(ds_model_list):
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        one_category = len({epoch_category_label(d['n_epochs']) for d in group_data}) <= 1
        
        # First pass: collect all values to determine y-axis range
        all_values_row = []
        
        # Collect names leaked values
        for data in group_data:
            if data['total_train_names'] is not None:
                names_leaked = data['recall'] * data['total_train_names']
                all_values_row.extend(names_leaked)
        
        # Collect false positives values
        for data in group_data:
            if data['cumulative_fp'] is not None:
                all_values_row.extend(data['cumulative_fp'])
        
        # Determine y-axis limits for this row
        if len(all_values_row) > 0:
            y_min = max(-0.01, min(all_values_row) * 0.95)  # Add 5% padding at bottom
            y_max = max(all_values_row) * 1.05  # Add 5% padding at top
        else:
            y_min = -0.01
            y_max = 1.0
        
        # Plot 1: Total Names Leaked vs budget
        ax = axes[row_idx, 0]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            
            # Calculate absolute number of names leaked
            # Recall = (1/|M|) sum_{i in M} P(E_i;N) q_i
            # So names_leaked = recall * |M| = sum_{i in M} P(E_i;N) q_i (expected TP)
            if data['total_train_names'] is not None:
                names_leaked = data['recall'] * data['total_train_names']
                
                # Debug: Check consistency
                sample_idx = min(10, len(data['recall']) - 1) if len(data['recall']) > 0 else 0
                if len(data['recall']) > 0 and len(data['fpr_extracted']) > 0:
                    sample_recall = data['recall'][sample_idx]
                    sample_fpr = data['fpr_extracted'][sample_idx]
                    sample_budget = data['budget'][sample_idx]
                    sample_names_leaked = names_leaked[sample_idx]
                    sample_cum_fp = data['cumulative_fp'][sample_idx] if data['cumulative_fp'] is not None else None
                    if sample_cum_fp is not None:
                        print(f"    Debug {data['directory']} at budget={sample_budget:.0f}: recall={sample_recall:.4f}, FPR={sample_fpr:.4f}, names_leaked={sample_names_leaked:.2f}, cum_FP={sample_cum_fp:.2f}, train_names={data['total_train_names']}, val_names={data['total_val_names']}")
            else:
                print(f"  Warning: No total_train_names for {data['directory']}, skipping absolute recall")
                continue
            
            # Format legend (without dataset size, as it's in the title)
            # Epochs: 2/3 -> no overfit, 10 -> overfit
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, names_leaked_plot = filter_budget_data(data['budget'], names_leaked, min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                names_leaked_plot = names_leaked
            
            ax.plot(budget_plot, names_leaked_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_tp')
            y_err = data.get('exp_tp_err')
            if y_exp is not None and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for names leaked if requested
                if annotate:
                    exp_tp = data.get('exp_tp')
                    if exp_tp is not None and np.isfinite(exp_tp):
                        ax.annotate(f'{int(exp_tp)}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for title
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Total names extracted and classified correctly as from finetuning', fontsize=12)
        ax.set_title(f'Total True Names Extracted vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(y_min, y_max)
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Total False Positives vs budget
        ax = axes[row_idx, 1]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            
            # Calculate cumulative false positives (must be computed from scores file)
            if data['cumulative_fp'] is not None:
                false_positives = data['cumulative_fp']
            else:
                print(f"  Warning: No cumulative_fp data for {data['directory']}, skipping absolute FPR")
                continue
            
            # Format legend (without dataset size, as it's in the title)
            # Epochs: 2/3 -> no overfit, 10 -> overfit
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, false_positives_plot = filter_budget_data(data['budget'], false_positives, min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                false_positives_plot = false_positives
            
            ax.plot(budget_plot, false_positives_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_fp')
            y_err = data.get('exp_fp_err')
            if y_exp is not None and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for false positives if requested
                if annotate:
                    exp_fp = data.get('exp_fp')
                    if exp_fp is not None and np.isfinite(exp_fp):
                        ax.annotate(f'{int(exp_fp)}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for title
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Total names extracted and classified wrongly as from finetuning', fontsize=12)
        ax.set_title(f'Total False Positives vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(y_min, y_max)  # Use same y-axis limits as first plot
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Theoretical Curves Summary - Absolute Values (With Verification)', fontsize=18, fontweight='bold', y=0.995)
    
    # Adjust bottom margin if cost axis is shown (cost labels are at bottom)
    if show_cost:
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])  # More space at bottom for cost labels
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if output_path is None:
        suffix = "_filtered" if use_filtered_exp else ""
        output_path = os.path.join(BASE_DIR, f"summary_theo_plot_absolute{suffix}.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved absolute values plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved absolute values plot to: {pdf_path}")
    
    return fig

def create_percentage_plots(theory_data, output_path=None, use_filtered_exp=False, show_cost=False, short=False, annotate=False, hide_tau_in_legend=False):
    """
    Create summary plots with percentage values: total names leaked and total false positives
    as percentages of total names in finetuning set (training set).
    Organized by (dataset_size, model_size).
    
    Args:
        theory_data: List of theoretical data dictionaries
        output_path: Optional path to save the plot
        use_filtered_exp: Whether to use filtered experimental metrics
        show_cost: Whether to show cost axis below budget axis
        annotate: If True, display number of names leaked (exp_tp) or false positives (exp_fp) near experimental dots
    """
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    # Get unique PII rates for color gradient assignment
    pii_rates = sorted(set(data['pii_rate'] for data in theory_data))
    
    # Use a red-based colormap with evenly spaced colors (based on order, not PII value)
    # Start from 0.5 to avoid pale colors, go to 1.0 for dark red
    pii_colors = {}
    n_pii = len(pii_rates)
    if n_pii > 0:
        for idx, pii_rate in enumerate(pii_rates):
            # Evenly space colors: first gets 0.5, last gets 1.0
            if n_pii == 1:
                colormap_val = 0.75
            else:
                colormap_val = 0.5 + 0.5 * (idx / (n_pii - 1))
            pii_colors[pii_rate] = plt.cm.Reds(colormap_val)
    
    # Create figure with rows for each (dataset_size, model_size) and 2 columns
    n_rows = len(ds_model_list)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each (dataset_size, model_size) group
    for row_idx, (ds_size, model) in enumerate(ds_model_list):
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        one_category = len({epoch_category_label(d['n_epochs']) for d in group_data}) <= 1
        
        # First pass: collect all values to determine y-axis range
        all_values_row = []
        
        # Collect names leaked percentages
        for data in group_data:
            if data['total_train_names'] is not None and data['total_train_names'] > 0:
                names_leaked_pct = data['recall'] * 100
                all_values_row.extend(names_leaked_pct)
        
        # Collect false positives percentages
        for data in group_data:
            if data['cumulative_fp'] is not None and data['total_train_names'] is not None and data['total_train_names'] > 0:
                false_positives_pct = (data['cumulative_fp'] / data['total_train_names']) * 100
                all_values_row.extend(false_positives_pct)
        
        # Determine y-axis limits for this row
        if len(all_values_row) > 0:
            # y_min = max(-1, min(all_values_row) * 0.95)  # Add 5% padding at bottom
            y_min = -max(all_values_row) * 0.05
            # y_min = max()
            y_max = max(all_values_row) * 1.05  # Add 5% padding at top
        else:
            y_min = -0.01
            y_max = 1.0

        # y_min = -1
        # y_max = 12
        
        # Plot 1: Total Names Leaked (as percentage) vs budget
        ax = axes[row_idx, 0]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            
            # Calculate percentage of names leaked (recall is already a fraction)
            if data['total_train_names'] is not None and data['total_train_names'] > 0:
                names_leaked_pct = data['recall'] * 100
            else:
                print(f"  Warning: No total_train_names for {data['directory']}, skipping percentage recall")
                continue
            
            # Format legend (without dataset size, as it's in the title)
            # Epochs: 2/3 -> no overfit, 10 -> overfit
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, names_leaked_pct_plot = filter_budget_data(data['budget'], names_leaked_pct, min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                names_leaked_pct_plot = names_leaked_pct
            
            ax.plot(budget_plot, names_leaked_pct_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_tp')
            y_err = data.get('exp_tp_err')
            if y_exp is not None and data['total_train_names'] is not None and data['total_train_names'] > 0 and exp_budget is not None:
                exp_tp_pct = (y_exp / data['total_train_names']) * 100
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Convert error to percentage
                    y_err_pct = [y_err[0] / data['total_train_names'] * 100, y_err[1] / data['total_train_names'] * 100]
                    # Use errorbar when CI data is available
                    eb = ax.errorbar([exp_budget], [exp_tp_pct], yerr=[[y_err_pct[0]], [y_err_pct[1]]], 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [exp_tp_pct], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for names leaked if requested
                if annotate:
                    exp_tp = data.get('exp_tp')
                    if exp_tp is not None and np.isfinite(exp_tp):
                        ax.annotate(f'{int(exp_tp)}', 
                                   xy=(exp_budget, exp_tp_pct), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for title
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('True names extracted (% of finetuning set)', fontsize=12)
        ax.set_title(f'True Names Extracted vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(y_min, y_max)
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Total False Positives (as percentage) vs budget
        ax = axes[row_idx, 1]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            
            # Calculate percentage of false positives (normalized by training set size for comparison)
            if data['cumulative_fp'] is not None and data['total_train_names'] is not None and data['total_train_names'] > 0:
                false_positives = data['cumulative_fp']
                false_positives_pct = (false_positives / data['total_train_names']) * 100
            else:
                print(f"  Warning: No cumulative_fp data for {data['directory']}, skipping percentage FPR")
                continue
            
            # Format legend (without dataset size, as it's in the title)
            # Epochs: 2/3 -> no overfit, 10 -> overfit
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Filter budget data if short mode is enabled
            if short:
                budget_plot, false_positives_pct_plot = filter_budget_data(data['budget'], false_positives_pct, min_budget=1e2, max_budget=1e9)
            else:
                budget_plot = data['budget']
                false_positives_pct_plot = false_positives_pct
            
            ax.plot(budget_plot, false_positives_pct_plot,
                    color=color, linestyle=linestyle,
                    marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_fp')
            y_err = data.get('exp_fp_err')
            if y_exp is not None and data['total_train_names'] is not None and data['total_train_names'] > 0 and exp_budget is not None:
                exp_fp_pct = (y_exp / data['total_train_names']) * 100
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Convert error to percentage
                    y_err_pct = [y_err[0] / data['total_train_names'] * 100, y_err[1] / data['total_train_names'] * 100]
                    # Use errorbar when CI data is available
                    eb = ax.errorbar([exp_budget], [exp_fp_pct], yerr=[[y_err_pct[0]], [y_err_pct[1]]], 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [exp_fp_pct], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for false positives if requested
                if annotate:
                    exp_fp = data.get('exp_fp')
                    if exp_fp is not None and np.isfinite(exp_fp):
                        ax.annotate(f'{int(exp_fp)}', 
                                   xy=(exp_budget, exp_fp_pct), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for title
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('False positives (% of finetuning set)', fontsize=12)
        ax.set_title(f'False Positives vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(y_min, y_max)  # Use same y-axis limits as first plot
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (train/val only) (with 95% CI)' if use_filtered_exp else 'Experimental results (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Theoretical Curves Summary - Percentage of Finetuning Set (With Verification)', fontsize=18, fontweight='bold', y=0.995)
    
    # Adjust bottom margin if cost axis is shown (cost labels are at bottom)
    if show_cost:
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])  # More space at bottom for cost labels
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if output_path is None:
        suffix = "_filtered" if use_filtered_exp else ""
        # for ax in axes.flat:
            # ax.set_ylim(bottom=-0.05, top=0.5)
        output_path = os.path.join(BASE_DIR, f"summary_theo_plot_percentage{suffix}.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved percentage values plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved percentage values plot to: {pdf_path}")
    
    return fig

def create_summary_extraction_plots(theory_data, output_path=None, use_filtered_exp=False, show_cost=False, short=False, annotate=False, hide_tau_in_legend=False):
    """
    Create summary plots for extraction recall (without verification).
    Plots total extraction recall (all train names extracted, regardless of classification)
    as a function of budget, comparing theoretical and experimental results.
    Organized by (dataset_size, model_size).
    
    Args:
        theory_data: List of theoretical data dictionaries
        output_path: Optional path to save the plot
        use_filtered_exp: Whether to use filtered experimental metrics
        show_cost: Whether to show cost axis below budget axis
        annotate: If True, display number of names extracted near experimental dots
    """
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    print(f"\nFound {len(ds_model_list)} unique (dataset_size, model_size) combinations for extraction plots")
    print("Groups:")
    for ds, model in ds_model_list:
        print(f"  {ds}_{model}")
    
    # Get unique PII rates for color gradient assignment
    pii_rates = sorted(set(data['pii_rate'] for data in theory_data))
    
    # Use a red-based colormap with evenly spaced colors (based on order, not PII value)
    # Start from 0.5 to avoid pale colors, go to 1.0 for dark red
    pii_colors = {}
    n_pii = len(pii_rates)
    if n_pii > 0:
        for idx, pii_rate in enumerate(pii_rates):
            # Evenly space colors: first gets 0.5, last gets 1.0
            if n_pii == 1:
                colormap_val = 0.75
            else:
                colormap_val = 0.5 + 0.5 * (idx / (n_pii - 1))
            pii_colors[pii_rate] = plt.cm.Reds(colormap_val)
    
    # Create figure with rows for each (dataset_size, model_size) and 1 column
    n_rows = len(ds_model_list)
    n_cols = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8 * n_rows))
    
    # Handle case where there's only one row
    if n_rows == 1:
        axes = [axes]
    
    # Plot for each (dataset_size, model_size) group
    for row_idx, (ds_size, model) in enumerate(ds_model_list):
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        one_category = len({epoch_category_label(d['n_epochs']) for d in group_data}) <= 1
        
        # Plot: Extraction Recall (without verification) vs budget
        if n_rows == 1:
            ax = axes[0]
        else:
            ax = axes[row_idx]
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            
            # Color based on PII rate only
            color = pii_colors[pii_rate]
            
            # Line style: solid for overfit (10 epochs), dashed for no overfit (2/3 epochs)
            if n_ep == 10:
                linestyle = '--'
            elif n_ep in [2, 3]:
                linestyle = '-'
            else:
                linestyle = '-'
            
            tau_val = data['tau']
            # Format legend: epochs -> overfit/no overfit, PII -> percentage
            if n_ep in [2, 3]:
                ep_label = "no overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep={n_ep}"
            
            # PII rate as percentage
            pii_pct = pii_rate * 100
            
            # Format tau
            if tau_val < 1:
                tau_str = f"τ={tau_val:.4f}"
            else:
                tau_str = f"τ={tau_val:.2f}"
            
            label = format_theory_legend_label(
                n_ep=n_ep,
                pii_rate=pii_rate,
                tau_val=tau_val,
                include_category=not one_category,
                include_tau=not hide_tau_in_legend,
            )
            
            # Plot theoretical extraction recall (without verification)
            recall_extraction = data.get('recall_without_verification')
            if recall_extraction is not None and not np.all(np.isnan(recall_extraction)):
                # Filter budget data if short mode is enabled
                if short:
                    budget_plot, recall_extraction_plot = filter_budget_data(data['budget'], recall_extraction, min_budget=1e2, max_budget=1e9)
                else:
                    budget_plot = data['budget']
                    recall_extraction_plot = recall_extraction
                
                ax.plot(budget_plot, recall_extraction_plot,
                        color=color, linestyle=linestyle,
                        marker='o', linewidth=2, markersize=4, label=label, alpha=0.8)
            else:
                print(f"  Warning: No extraction recall data for {data['directory']}")
            
            # Experimental point at the budget from directory name
            exp_budget = data.get('exp_budget')
            y_exp = data.get('exp_extraction_recall')
            y_err = data.get('exp_extraction_recall_err')
            if y_exp is not None and np.isfinite(y_exp) and exp_budget is not None:
                if y_err is not None and not any(pd.isna(v) for v in y_err):
                    # Use errorbar when CI data is available
                    # Format: yerr as tuple (lower_errors, upper_errors) for asymmetric error bars
                    eb = ax.errorbar([exp_budget], [y_exp], yerr=([y_err[0]], [y_err[1]]), 
                               marker='x', markersize=10, markeredgewidth=2.5, 
                               color=color, zorder=6, capsize=4, capthick=1.5, elinewidth=1.5)
                    # Make error bar lines dashed if no overfitting (n_ep in [2, 3]) to match theoretical curves
                    if n_ep in [2, 3]:
                        # Access error bar lines through the container
                        # Error bar lines are stored as LineCollection objects or Line2D objects
                        # Try to access through container attributes
                        if hasattr(eb, 'lines'):
                            for line in eb.lines:
                                if hasattr(line, 'set_linestyle'):
                                    line.set_linestyle('--')
                        # Access through children (LineCollection for error bars)
                        for child in eb.get_children():
                            if isinstance(child, LineCollection):
                                # For LineCollection, set dashes on all segments
                                child.set_dashes([(0, (2, 2))])
                            elif isinstance(child, plt.Line2D):
                                child.set_linestyle('--')
                            elif hasattr(child, 'set_linestyle'):
                                child.set_linestyle('--')
                else:
                    # Fall back to scatter if no error bars
                    ax.scatter([exp_budget], [y_exp], marker='x', s=120, linewidths=2.5, color=color, zorder=6)
                
                # Add annotation for number of names extracted if requested
                if annotate:
                    # Calculate number of names extracted from extraction recall
                    total_train_names = data.get('total_train_names')
                    if total_train_names is not None and total_train_names > 0 and y_exp is not None and np.isfinite(y_exp):
                        names_extracted = int(y_exp * total_train_names)
                        ax.annotate(f'{names_extracted}', 
                                   xy=(exp_budget, y_exp), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9, 
                                   color=color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color),
                                   zorder=7)
        
        # Get dataset size label for title
        ds_label = get_dataset_size_label(ds_size)
        
        ax.set_xlabel('Budget (number of queries) - Cost' if show_cost else 'Budget (number of queries)', fontsize=12)
        ax.set_ylabel('Names extracted but not verified (% of finetuning set)', fontsize=12)
        ax.set_title(f'Extraction Recall vs Budget ({ds_label}, {model})', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        if short:
            ax.set_xlim(left=1e2, right=1e9)
        ax.set_ylim(bottom=-0.01, top=1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))  # display 0-1 as 0%-100%
        add_cost_axis(ax, show_cost=show_cost)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='x', linestyle='None', color='black',
                              markersize=10, markeredgewidth=2.5))
        exp_label = 'Experimental (extraction, no verification) (with 95% CI)' if use_filtered_exp else 'Experimental extraction (no verification) (with 95% CI)'
        labels.append(exp_label)
        handles, labels = deduplicate_legend(handles, labels)
        ax.legend(handles, labels, fontsize=11, loc='best', framealpha=0.9, handlelength=3, handletextpad=0.5)
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Extraction Recall Summary (Without Verification)', fontsize=18, fontweight='bold', y=0.995)
    
    # Adjust bottom margin if cost axis is shown (cost labels are at bottom)
    if show_cost:
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])  # More space at bottom for cost labels
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if output_path is None:
        suffix = "_filtered" if use_filtered_exp else ""
        output_path = os.path.join(BASE_DIR, f"summary_extraction{suffix}.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved extraction recall plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved extraction recall plot to: {pdf_path}")
    
    return fig

def main():
    """Main function to collect data and create plots."""
    parser = argparse.ArgumentParser(description='Create summary plots for theoretical curves')
    parser.add_argument('--use-filtered-exp', action='store_true',
                        help='Use filtered experimental metrics (train/val only) instead of all values')
    parser.add_argument('--show-cost', action='store_true',
                        help='Show cost axis at top of plot (assumes 20 tokens per query, $5 per million tokens)')
    parser.add_argument('--short', action='store_true',
                        help='Only display curves between 10^2 and 10^9 budget')
    parser.add_argument('--annotate', action='store_true',
                        help='Display number of names leaked (exp_tp) or false positives (exp_fp) near experimental dots')
    parser.add_argument('--hide-tau-legend', action='store_true',
                        help='Hide τ (tau) in legend labels')
    args = parser.parse_args()
    
    use_filtered_exp = args.use_filtered_exp
    show_cost = args.show_cost
    short = args.short
    annotate = args.annotate
    hide_tau_in_legend = args.hide_tau_legend
    
    print("="*80)
    print("Creating Summary Plots for Theoretical Curves")
    if use_filtered_exp:
        print("Using filtered experimental metrics (train/val only)")
    else:
        print("Using all experimental metrics")
    if show_cost:
        print("Showing cost axis (20 tokens/query, $5 per million tokens = $0.0001/query)")
    print("="*80)
    
    if os.path.exists(BASE_DIR):
        # Load threshold data
        print("\nLoading threshold data...")
        threshold_map = load_threshold_data(THRESHOLD_CSV)
        
        if len(threshold_map) == 0:
            print("ERROR: No threshold data found. Cannot proceed.")
            return 1
        
        # Load threshold DataFrame for file paths
        threshold_df = None
        if os.path.exists(THRESHOLD_CSV):
            try:
                threshold_df = pd.read_csv(THRESHOLD_CSV)
            except Exception as e:
                print(f"Warning: Could not load threshold CSV as DataFrame: {e}")
        
        # Collect theoretical data
        print("\nCollecting theoretical curve data...")
        theory_data = collect_theoretical_data(BASE_DIR, threshold_map, threshold_df=threshold_df, use_filtered_exp=use_filtered_exp)
        
        print(f"\nCollected theoretical data for {len(theory_data)} directories")

        # save theory_data to a pickle file
        with open('theory_exp_data.pkl', 'wb') as f:
            pickle.dump(theory_data, f)

    # read theory_data from the pickle file
    with open('theory_exp_data.pkl', 'rb') as f:
        theory_data = pickle.load(f)
    
    # Create plots (rates)
    print("\nCreating summary plots (rates)...")
    fig = create_summary_plots(theory_data, use_filtered_exp=use_filtered_exp, show_cost=show_cost, short=short, annotate=annotate, hide_tau_in_legend=hide_tau_in_legend)
    
    # Create plots (absolute values)
    print("\nCreating summary plots (absolute values)...")
    fig_absolute = create_absolute_plots(theory_data, use_filtered_exp=use_filtered_exp, show_cost=show_cost, short=short, annotate=annotate, hide_tau_in_legend=hide_tau_in_legend)
    
    # Create plots (percentage values)
    print("\nCreating summary plots (percentage values)...")
    fig_percentage = create_percentage_plots(theory_data, use_filtered_exp=use_filtered_exp, show_cost=show_cost, short=short, annotate=annotate, hide_tau_in_legend=hide_tau_in_legend)
    
    # Create plots (extraction recall without verification)
    print("\nCreating summary plots (extraction recall without verification)...")
    fig_extraction = create_summary_extraction_plots(theory_data, use_filtered_exp=use_filtered_exp, show_cost=show_cost, short=short, annotate=annotate, hide_tau_in_legend=hide_tau_in_legend)
    
    # Show plots
    plt.show()
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    exit(main())
