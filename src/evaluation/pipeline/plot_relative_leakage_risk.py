#!/usr/bin/env python3
"""
Plot relative leakage risk: average relative risk (mean(pi_i/qi_i)) for names in finetuning set (split train)
for the prompt 'Name: ' only.

This matches the calculation in overall_proba.py for average_relative_risk_ci_name.

x-axis: pii_rate
y-axis: average relative risk (mean(pi_i/qi_i)) with 95% bootstrap CI
lines: overfit (n_epochs=10) and no-overfit (n_epochs=3, dashed)
datasets: medium (10) and large (100)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle

def load_and_filter_data(file_path, splits=['train'], prompt='Name: '):
    """Load CSV and filter for split(s) and prompt."""
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for specified split(s) and Name: prompt
    if isinstance(splits, str):
        splits = [splits]
    df = df[df['split'].isin(splits)]
    df = df[df['prompt'] == prompt]
    
    return df

def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Calculate bootstrap confidence interval for median.
    Matches the implementation in overall_proba.py.
    
    Parameters:
    -----------
    data : array-like
        Data to bootstrap
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    median : float
        Median of the data
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    np.random.seed(random_state)
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    func = np.median
    median = func(data)
    
    # Bootstrap sampling
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians.append(func(bootstrap_sample))
    
    bootstrap_medians = np.array(bootstrap_medians)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
    
    return median, ci_lower, ci_upper

def bootstrap_ci_percentile(data, percentile=90, n_bootstrap=1000, confidence=0.95, random_state=42):
    """
    Calculate bootstrap confidence interval for a specified percentile.
    
    Parameters:
    -----------
    data : array-like
        Data to bootstrap
    percentile : float
        Percentile to calculate (e.g., 90 for 90th percentile)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    percentile_value : float
        Percentile of the data
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    np.random.seed(random_state)
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    n = len(data)
    
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Calculate actual percentile
    p_value = np.percentile(data, percentile)
    
    # Bootstrap sampling
    bootstrap_percentiles = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_percentiles.append(np.percentile(bootstrap_sample, percentile))
    
    bootstrap_percentiles = np.array(bootstrap_percentiles)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_percentiles, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_percentiles, 100 * (1 - alpha / 2))
    
    return p_value, ci_lower, ci_upper

def calculate_factor(df_pi, df_qi, percentile=90, filter_proba=None, groupby_cols=['pii_rate', 'n_epochs', 'dataset_size', 'split']):
    """
    Calculate average relative risk: median(pi_i / qi_i) for each group.
    This matches the calculation in overall_proba.py for average_relative_risk_ci_name.
    
    pi = exp(ll) from df_pi, qi = exp(ll) from df_qi (base model)
    
    For each (pii_rate, n_epochs, dataset_size, split) group in pi:
    - Get all unique names in that group (set of names in finetuning set)
    - For each unique name, calculate relative_risk_i = pi_i / qi_i
    - Calculate median(relative_risk_i) and bootstrap CI
    - Calculate specified percentile
    
    Parameters:
    -----------
    percentile : float
        Percentile to calculate (default: 90 for 90th percentile)
    filter_proba : float, optional
        If provided, filter to only keep rows where pi > filter_proba (default: None)
    
    Returns:
    --------
    df_results : DataFrame with aggregated statistics
    df_individual : DataFrame with individual name-level relative risks
    """
    # Calculate pi = exp(ll) for each row
    df_pi = df_pi.copy()
    df_pi['pi'] = np.exp(df_pi['ll'])
    
    # Filter by probability threshold if specified
    if filter_proba is not None:
        initial_count = len(df_pi)
        df_pi = df_pi[df_pi['pi'] > filter_proba].copy()
        filtered_count = len(df_pi)
        print(f"  Filtered pi > {filter_proba}: {initial_count} -> {filtered_count} rows ({filtered_count/initial_count*100:.1f}% kept)")
    
    # For qi (base model), filter to pii_rate=0 and n_epochs=0
    # and calculate qi = exp(ll)
    df_qi = df_qi.copy()
    df_qi_base = df_qi[(df_qi['pii_rate'] == 0.0) & (df_qi['n_epochs'] == 0)].copy()
    df_qi_base['qi'] = np.exp(df_qi_base['ll'])
    
    # Deduplicate qi_base on (value, prompt) - take first occurrence
    df_qi_base = df_qi_base.drop_duplicates(subset=['value', 'prompt'], keep='first')
    
    results = []
    individual_risks = []  # Store individual name-level risks
    
    # Group pi by (pii_rate, n_epochs, dataset_size, split)
    for group_key, df_pi_group in df_pi.groupby(groupby_cols):
        # Unpack group key based on number of columns
        if len(groupby_cols) == 4:
            pii_rate, n_epochs, dataset_size, split_val = group_key
        else:
            pii_rate, n_epochs, dataset_size = group_key
            split_val = df_pi_group['split'].iloc[0] if 'split' in df_pi_group.columns else 'train'
        # Get unique names in this pi group (set of names in finetuning set)
        # Take first occurrence for each unique (value, prompt) pair
        df_pi_unique = df_pi_group.drop_duplicates(subset=['value', 'prompt'], keep='first')
        
        # Get the set of names
        pi_names = df_pi_unique[['value', 'prompt']].copy()
        
        # Match with qi base model
        df_qi_matched = df_qi_base.merge(
            pi_names,
            on=['value', 'prompt'],
            how='inner'
        )
        
        # Merge pi and qi to calculate individual relative risks
        df_merged = df_pi_unique.merge(
            df_qi_matched[['value', 'prompt', 'qi']],
            on=['value', 'prompt'],
            how='inner'
        )
        
        # Calculate individual relative risks: pi_i / qi_i
        df_merged['relative_risk'] = df_merged['pi'] / df_merged['qi']
        
        # Store individual risks with metadata
        for _, row in df_merged.iterrows():
            if not pd.isna(row['relative_risk']):
                individual_risks.append({
                    'pii_rate': pii_rate,
                    'n_epochs': n_epochs,
                    'dataset_size': dataset_size,
                    'value': row['value'],
                    'prompt': row['prompt'],
                    'relative_risk': row['relative_risk'],
                    'pi': row['pi'],
                    'qi': row['qi']
                })
        
        # Remove NaN values
        relative_risks = df_merged['relative_risk'].dropna().values
        
        if len(relative_risks) > 0:
            # Calculate median and bootstrap CI
            median, ci_lower, ci_upper = bootstrap_ci(relative_risks)
            
            # Calculate specified percentile and its bootstrap CI
            p_percentile, p_ci_lower, p_ci_upper = bootstrap_ci_percentile(relative_risks, percentile=percentile)
            
            results.append({
                'pii_rate': pii_rate,
                'n_epochs': n_epochs,
                'dataset_size': dataset_size,
                'split': split_val,
                'factor': median,  # Median relative risk
                'factor_ci_lower': ci_lower,
                'factor_ci_upper': ci_upper,
                'percentile': p_percentile,  # Specified percentile
                'percentile_ci_lower': p_ci_lower,
                'percentile_ci_upper': p_ci_upper,
                'n_names': len(relative_risks)
            })
        else:
            print(f"Warning: No valid relative risks for pii_rate={pii_rate}, n_epochs={n_epochs}, dataset_size={dataset_size}")
    
    return pd.DataFrame(results), pd.DataFrame(individual_risks)

def plot_relative_leakage_risk(df_results, output_path, percentile=90, include_val=False, show_median=True, prompt='Name: ', no_overfit_only=False):
    """
    Plot relative leakage risk with:
    - x-axis: pii_rate
    - y-axis: median and specified percentile relative risk (pi_i/qi_i) with 95% CI
    - lines: overfit (solid) and no-overfit (dashed)
    - datasets: medium (10) and large (100) with different colors
    - splits: train and val (if include_val=True) with different line styles
    - markers: circles for median, triangles for percentile
    
    Parameters:
    -----------
    percentile : float
        Percentile being plotted (for labeling)
    include_val : bool
        Whether to include validation split in the plot
    """
    # Filter by split
    splits_to_plot = ['train']
    if include_val:
        splits_to_plot.append('val')
    
    df_results = df_results[df_results['split'].isin(splits_to_plot)].copy()
    
    # Separate by dataset size
    df_medium = df_results[df_results['dataset_size'] == 10].copy()
    df_large = df_results[df_results['dataset_size'] == 100].copy()
    
    # Filter to only no-overfit if requested
    if no_overfit_only:
        df_medium = df_medium[df_medium['n_epochs'] == 3].copy()
        df_large = df_large[df_large['n_epochs'] == 3].copy()
    
    # Create figure - smaller size so labels appear larger relative to plot
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Define colors for different splits
    split_styles = {
        'train': {'color_medium': 'blue', 'color_large': 'red'},
        'val': {'color_medium': 'cyan', 'color_large': 'orange'}
    }
    
    # Plot for each split
    for split in splits_to_plot:
        df_medium_split = df_medium[df_medium['split'] == split].copy()
        df_large_split = df_large[df_large['split'] == split].copy()
        
        # Separate by overfit condition
        if no_overfit_only:
            # Only show no-overfit curves
            df_medium_overfit = pd.DataFrame()  # Empty
            df_medium_no_overfit = df_medium_split.sort_values('pii_rate')
            df_large_overfit = pd.DataFrame()  # Empty
            df_large_no_overfit = df_large_split.sort_values('pii_rate')
        else:
            # Show both overfit and no-overfit
            df_medium_overfit = df_medium_split[df_medium_split['n_epochs'] == 10].sort_values('pii_rate')
            df_medium_no_overfit = df_medium_split[df_medium_split['n_epochs'] == 3].sort_values('pii_rate')
            df_large_overfit = df_large_split[df_large_split['n_epochs'] == 10].sort_values('pii_rate')
            df_large_no_overfit = df_large_split[df_large_split['n_epochs'] == 3].sort_values('pii_rate')
        
        style = split_styles[split]
        split_label = f"{split.capitalize()} set"
        
        # Build simplified label: only split and percentile/median
        def build_label(metric_type='percentile'):
            if metric_type == 'percentile':
                return f"{split_label} - {percentile:.0f}th percentile" # (emission probability)"
            elif metric_type == 'median':
                return f"{split_label} - Median"
            return split_label
        
        # Plot medium dataset - MEDIAN with error bars (if show_median is True)
        # Overfit: solid line, No-overfit: dotted line
        if show_median:
            if len(df_medium_overfit) > 0:
                x = df_medium_overfit['pii_rate'] * 100
                y = df_medium_overfit['factor']
                yerr_lower = df_medium_overfit['factor'] - df_medium_overfit['factor_ci_lower']
                yerr_upper = df_medium_overfit['factor_ci_upper'] - df_medium_overfit['factor']
                
                ax.errorbar(
                    x, y,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='o--',  # Solid line for overfit
                    color=style['color_medium'],
                    linewidth=2.5,
                    markersize=9,
                    capsize=5,
                    capthick=2,
                    label=build_label(metric_type='median')
                )
            
            if len(df_medium_no_overfit) > 0:
                x = df_medium_no_overfit['pii_rate'] * 100
                y = df_medium_no_overfit['factor']
                yerr_lower = df_medium_no_overfit['factor'] - df_medium_no_overfit['factor_ci_lower']
                yerr_upper = df_medium_no_overfit['factor_ci_upper'] - df_medium_no_overfit['factor']
                
                ax.errorbar(
                    x, y,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='o-',  # Dotted line for no-overfit
                    color=style['color_medium'],
                    linewidth=2.5,
                    markersize=9,
                    capsize=5,
                    capthick=2,
                    label=build_label(metric_type='median'),
                    alpha=0.7
                )
            
            # Plot large dataset - MEDIAN with error bars
            if len(df_large_overfit) > 0:
                x = df_large_overfit['pii_rate'] * 100
                y = df_large_overfit['factor']
                yerr_lower = df_large_overfit['factor'] - df_large_overfit['factor_ci_lower']
                yerr_upper = df_large_overfit['factor_ci_upper'] - df_large_overfit['factor']
                
                ax.errorbar(
                    x, y,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='s--',  # Solid line for overfit
                    color=style['color_large'],
                    linewidth=2.5,
                    markersize=9,
                    capsize=5,
                    capthick=2,
                    label=build_label(metric_type='median')
                )
            
            if len(df_large_no_overfit) > 0:
                x = df_large_no_overfit['pii_rate'] * 100
                y = df_large_no_overfit['factor']
                yerr_lower = df_large_no_overfit['factor'] - df_large_no_overfit['factor_ci_lower']
                yerr_upper = df_large_no_overfit['factor_ci_upper'] - df_large_no_overfit['factor']
                
                ax.errorbar(
                    x, y,
                    yerr=[yerr_lower, yerr_upper],
                    fmt='s-',  # Dotted line for no-overfit
                    color=style['color_large'],
                    linewidth=2.5,
                    markersize=9,
                    capsize=5,
                    capthick=2,
                    label=build_label(metric_type='median'),
                    alpha=0.7
                )
        
        # Plot medium dataset - PERCENTILE with error bars
        if len(df_medium_overfit) > 0:
            x = df_medium_overfit['pii_rate'] * 100
            y = df_medium_overfit['percentile']
            yerr_lower = df_medium_overfit['percentile'] - df_medium_overfit['percentile_ci_lower']
            yerr_upper = df_medium_overfit['percentile_ci_upper'] - df_medium_overfit['percentile']
            
            ax.errorbar(
                x, y,
                yerr=[yerr_lower, yerr_upper],
                fmt='^--',  # Triangle markers, solid line for overfit
                color=style['color_medium'],
                linewidth=2.5,
                markersize=9,
                capsize=5,
                capthick=2,
                label=build_label(metric_type='percentile'),
                alpha=0.8
            )
        
        if len(df_medium_no_overfit) > 0:
            x = df_medium_no_overfit['pii_rate'] * 100
            y = df_medium_no_overfit['percentile']
            yerr_lower = df_medium_no_overfit['percentile'] - df_medium_no_overfit['percentile_ci_lower']
            yerr_upper = df_medium_no_overfit['percentile_ci_upper'] - df_medium_no_overfit['percentile']
            
            ax.errorbar(
                x, y,
                yerr=[yerr_lower, yerr_upper],
                fmt='^-',  # Triangle markers, dotted line for no-overfit
                color=style['color_medium'],
                linewidth=2.5,
                markersize=9,
                capsize=5,
                capthick=2,
                label=build_label(metric_type='percentile'),
                alpha=0.5
            )
        
        # Plot large dataset - PERCENTILE with error bars
        if len(df_large_overfit) > 0:
            x = df_large_overfit['pii_rate'] * 100
            y = df_large_overfit['percentile']
            yerr_lower = df_large_overfit['percentile'] - df_large_overfit['percentile_ci_lower']
            yerr_upper = df_large_overfit['percentile_ci_upper'] - df_large_overfit['percentile']
            
            ax.errorbar(
                x, y,
                yerr=[yerr_lower, yerr_upper],
                fmt='v--',  # Inverted triangle markers, solid line for overfit
                color=style['color_large'],
                linewidth=2.5,
                markersize=9,
                capsize=5,
                capthick=2,
                label=build_label(metric_type='percentile'),
                alpha=0.8
            )
        
        if len(df_large_no_overfit) > 0:
            x = df_large_no_overfit['pii_rate'] * 100
            y = df_large_no_overfit['percentile']
            yerr_lower = df_large_no_overfit['percentile'] - df_large_no_overfit['percentile_ci_lower']
            yerr_upper = df_large_no_overfit['percentile_ci_upper'] - df_large_no_overfit['percentile']
            
            ax.errorbar(
                x, y,
                yerr=[yerr_lower, yerr_upper],
                fmt='v-',  # Inverted triangle markers, dotted line for no-overfit
                color=style['color_large'],
                linewidth=2.5,
                markersize=9,
                capsize=5,
                capthick=2,
                label=build_label(metric_type='percentile'),
                alpha=0.5
            )
    
    # Formatting - larger font sizes since figure is smaller
    ax.set_xlabel('PII Rate (%)', fontsize=16)
    ax.set_ylabel('Emission Probability Multiplier', fontsize=16)
    split_text = 'Train and Val Splits' if include_val else 'Train Split'
    metric_text = f'Median and {percentile:.0f}th Percentile' if show_median else f'{percentile:.0f}th Percentile'
    # Two-line title - include prompt info if not default
    prompt_label = prompt.strip() if prompt != 'Name: ' else 'Names'
    title_line1 = f'{metric_text} Emission Probability Multiplier'
    title_line2 = f'for {prompt_label} in Finetuning Set ({split_text})'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Larger tick labels to match axis labels
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Larger legend with better spacing to see solid vs dashed lines - one column
    ax.legend(fontsize=16, loc='best', ncol=1, framealpha=0.9, 
              handlelength=3, handletextpad=0.5)
    
    # Set y-axis to log scale if values span orders of magnitude
    all_values = pd.concat([df_results['factor'], df_results['percentile']])
    if all_values.max() / all_values.min() > 10:
        ax.set_yscale('log')
    
    # Set x-axis to log scale for pii_rate
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot relative leakage risk')
    parser.add_argument('--output', type=str,
                        default='/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/relative_emission_probability_change_leakage_risk.png',
                        help='Output path for the plot')
    parser.add_argument('--top_names_output', type=str,
                        default='/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/top_100_names_by_factor.csv',
                        help='Output path for top 100 names CSV')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top names to output (default: 100)')
    parser.add_argument('--percentile', type=float, default=90.0,
                        help='Percentile to calculate and plot (default: 90.0 for 90th percentile)')
    parser.add_argument('--include_val', action='store_true',
                        help='Include validation split in addition to train split')
    parser.add_argument('--dataset', type=str, default='10,100',
                        help='Comma-separated list of dataset sizes to process (default: 10,100). Example: --dataset 10,100 or --dataset 100')
    parser.add_argument('--no_median', action='store_true',
                        help='Do not show median plots, only show percentile')
    parser.add_argument('--filter-proba', type=float, default=None, dest='filter_proba',
                        help='Filter to only keep rows where pi > value (e.g., 1e-7). Default: no filtering')
    parser.add_argument('--prompt', type=str, default='Name: ',
                        help='Prompt to filter by (default: "Name: "). Example: --prompt "MRN: "')
    parser.add_argument('--no_overfit_only', action='store_true',
                        help='Only show no-overfit curves (n_epochs=3), hide overfit curves (n_epochs=10)')
    args = parser.parse_args()
    
    # Parse dataset sizes
    try:
        dataset_sizes = [int(x.strip()) for x in args.dataset.split(',')]
        dataset_sizes = sorted(set(dataset_sizes))  # Remove duplicates and sort
        if not dataset_sizes:
            raise ValueError("At least one dataset size must be specified")
        # Validate dataset sizes
        valid_sizes = [10, 100]
        invalid_sizes = [ds for ds in dataset_sizes if ds not in valid_sizes]
        if invalid_sizes:
            raise ValueError(f"Invalid dataset sizes: {invalid_sizes}. Valid sizes are: {valid_sizes}")
    except ValueError as e:
        print(f"Error parsing dataset sizes: {e}")
        print(f"Received: {args.dataset}")
        return
    
    print(f"Processing dataset sizes: {dataset_sizes}")
    
    # File paths
    base_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline'

    if os.path.exists(base_dir):
        
        # Load data for specified dataset sizes
        pi_files = {}
        qi_files = {}
        for ds in dataset_sizes:
            pi_files[ds] = f'{base_dir}/ll_all_output_False_1B_{ds}_batch.csv'
            qi_files[ds] = f'{base_dir}/ll_all_output_True_1B_{ds}_batch.csv'
        
        # Determine which splits to process
        splits_to_process = ['train']
        if args.include_val:
            splits_to_process.append('val')
        
        all_results = []
        all_individual_risks = []
        
        for dataset_size in dataset_sizes:
            print(f"\nProcessing dataset size: {dataset_size}")
            
            # Load and filter data for all splits
            df_pi = load_and_filter_data(pi_files[dataset_size], splits=splits_to_process, prompt=args.prompt)
            df_qi = load_and_filter_data(qi_files[dataset_size], splits=splits_to_process, prompt=args.prompt)
            
            # Calculate factor for each (pii_rate, n_epochs, split) combination
            df_factor, df_individual = calculate_factor(df_pi, df_qi, percentile=args.percentile, 
                                                    filter_proba=args.filter_proba)
            
            all_results.append(df_factor)
            all_individual_risks.append(df_individual)
        
        # Combine results
        df_all_results = pd.concat(all_results, ignore_index=True)
        df_all_individual = pd.concat(all_individual_risks, ignore_index=True)
        
        print("\nAggregated Results:")
        print(df_all_results)
        
        # Get top N names by relative risk
        # For names that appear in multiple configurations, take the maximum relative risk
        df_top_names = df_all_individual.groupby('value').agg({
            'relative_risk': 'max',  # Take max across all configurations
            'pii_rate': 'first',
            'n_epochs': 'first',
            'dataset_size': 'first',
            'pi': 'first',
            'qi': 'first',
            'prompt': 'first'
        }).reset_index()
        
        # Sort by relative risk descending and take top N
        df_top_names = df_top_names.sort_values('relative_risk', ascending=False).head(args.top_n)
        
        print(f"\nTop {args.top_n} names by relative risk factor:")
        print(df_top_names[['value', 'relative_risk', 'pii_rate', 'n_epochs', 'dataset_size', 'pi', 'qi']].to_string())
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save top names to CSV
        if args.top_names_output:
            os.makedirs(os.path.dirname(args.top_names_output), exist_ok=True)
            df_top_names.to_csv(args.top_names_output, index=False)
            print(f"\nTop {args.top_n} names saved to: {args.top_names_output}")
        
        # save to pick
        with open('df_all_results_relative_leakage_risk.pkl', 'wb') as f:
            pickle.dump(df_all_results, f)

    # read from pickle file
    with open('df_all_results_relative_leakage_risk.pkl', 'rb') as f:
        df_all_results = pickle.load(f)
    
    # Plot
    plot_relative_leakage_risk(df_all_results, args.output, percentile=args.percentile, 
                              include_val=args.include_val, show_median=not args.no_median, 
                              prompt=args.prompt, no_overfit_only=args.no_overfit_only)

if __name__ == '__main__':
    main()
