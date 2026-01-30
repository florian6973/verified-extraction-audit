"""
Script to replot combined metrics from CSV files:
- Average absolute leakage VS PII level
- Average Relative leakage vs PII level  
- MIA AUC vs PII level

One plot per overfit/no overfit and metric, with two subplots (name and MRN),
and lines for each model/dataset combination.
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_overall_proba_filename(filename):
    """Extract model, pii_type, and dataset_size from overall_proba filename."""
    basename = os.path.basename(filename)
    # Pattern: overall_proba_{model}_{pii_type}_{dataset_size}.csv
    match = re.match(r'overall_proba_([^_]+)_([^_]+)_(\d+)\.csv', basename)
    if match:
        model, pii_type, dataset_size = match.groups()
        return model, pii_type, int(dataset_size)
    return None, None, None


def parse_risk_stats_filename(filename):
    """Extract model, pii_type, and dataset_size from risk_stats filename."""
    basename = os.path.basename(filename)
    # Pattern: absolute_risk_stats_{model}_{pii_type}_{dataset_size}.csv or relative_risk_stats_{model}_{pii_type}_{dataset_size}.csv
    match = re.match(r'(?:absolute|relative)_risk_stats_([^_]+)_([^_]+)_(\d+)\.csv', basename)
    if match:
        model, pii_type, dataset_size = match.groups()
        return model, pii_type, int(dataset_size)
    return None, None, None


def parse_mia_filename(filename):
    """Extract model and dataset_size from MIA filename."""
    basename = os.path.basename(filename)
    # Pattern: mia_results_all_all_{model}_{dataset_size}.csv
    match = re.match(r'mia_results_all_all_([^_]+)_(\d+)\.csv', basename)
    if match:
        model, dataset_size = match.groups()
        return model, int(dataset_size)
    return None, None


def load_overall_proba_data(plots_dir):
    """Load all overall_proba CSV files and combine them."""
    pattern = os.path.join(plots_dir, 'overall_proba_*.csv')
    csv_files = glob.glob(pattern)
    
    all_data = []
    for csv_file in csv_files:
        model, pii_type, dataset_size = parse_overall_proba_filename(csv_file)
        if model is None:
            continue
        
        df = pd.read_csv(csv_file)
        df['model'] = model
        df['source_pii_type'] = pii_type  # Keep original pii_type from filename
        df['source_dataset_size'] = dataset_size
        
        # Map pii_type from filename to standard names
        if pii_type == 'name':
            df['pii_type_standard'] = 'name-patient'
            # Filter to only keep 'Name: ' prompt for names
            if 'prompt' in df.columns:
                df = df[df['prompt'] == 'Name: '].copy()
        elif pii_type == 'mrn':
            df['pii_type_standard'] = 'unit_no'
        else:
            df['pii_type_standard'] = df['pii_type'].iloc[0] if 'pii_type' in df.columns else pii_type
        
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Determine overfit/no overfit based on n_epochs
    df_combined['epoch_label'] = df_combined['n_epochs'].map({
        2: 'no overfit',
        3: 'no overfit', 
        10: 'overfit'
    })
    
    return df_combined


def load_risk_stats_data(plots_dir, risk_type='absolute'):
    """Load all risk_stats CSV files (absolute or relative) and combine them.
    
    Args:
        plots_dir: Directory containing the CSV files
        risk_type: 'absolute' or 'relative'
    
    Returns:
        Combined DataFrame with columns: pii_rate, epoch_label, mean, ci_lower, ci_upper, n_samples,
        plus model, source_pii_type, source_dataset_size, pii_type_standard
    """
    pattern = os.path.join(plots_dir, f'{risk_type}_risk_stats_*.csv')
    csv_files = glob.glob(pattern)
    
    all_data = []
    for csv_file in csv_files:
        model, pii_type, dataset_size = parse_risk_stats_filename(csv_file)
        if model is None:
            continue
        
        df = pd.read_csv(csv_file)
        df['model'] = model
        df['source_pii_type'] = pii_type
        df['source_dataset_size'] = dataset_size
        
        # Map pii_type from filename to standard names
        if pii_type == 'name':
            df['pii_type_standard'] = 'name-patient'
        elif pii_type == 'mrn':
            df['pii_type_standard'] = 'unit_no'
        else:
            df['pii_type_standard'] = pii_type
        
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def load_mia_data(mia_dir):
    """Load all MIA CSV files (all_all files) and combine them.
    The all_all files are already aggregated across prompts, so no prompt filtering needed.
    """
    pattern = os.path.join(mia_dir, 'mia_results_all_all_*.csv')
    csv_files = glob.glob(pattern)
    
    all_data = []
    for csv_file in csv_files:
        model, dataset_size = parse_mia_filename(csv_file)
        if model is None:
            continue
        
        df = pd.read_csv(csv_file)
        df['model'] = model
        df['source_dataset_size'] = dataset_size
        
        # The all_all files are already aggregated across prompts (no prompt column)
        # So we can use them directly
        
        # Determine overfit/no overfit
        df['epoch_label'] = df['n_epochs'].map({
            2: 'no overfit',
            3: 'no overfit',
            10: 'overfit'
        })
        
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def create_combined_plot_ci(df_data, metric_name, y_col, y_err_lower_col, y_err_upper_col, y_label, epoch_label, output_path):
    """
    Create a plot with two subplots (name and MRN) showing lines for each model/dataset combination.
    Uses asymmetric error bars from CI.
    
    Args:
        df_data: DataFrame with the data
        metric_name: Name of the metric (for title)
        y_col: Column name for y-axis values
        y_err_lower_col: Column name for lower error (mean - ci_lower)
        y_err_upper_col: Column name for upper error (ci_upper - mean)
        y_label: Label for y-axis
        epoch_label: 'overfit' or 'no overfit'
        output_path: Path to save the plot
    """
    # Filter by epoch_label
    df_filtered = df_data[df_data['epoch_label'] == epoch_label].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for {metric_name} - {epoch_label}")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define PII types for subplots
    pii_types = ['name-patient', 'unit_no']
    pii_labels = ['Name', 'MRN']
    
    # Line styles: 1B = solid, 8B = dashed
    model_linestyles = {'1B': '-', '8B': '--'}
    
    # Colors for dataset sizes (consistent across models)
    unique_dataset_sizes = sorted(df_filtered['source_dataset_size'].unique())
    dataset_colors = {ds: plt.cm.tab10(i / max(len(unique_dataset_sizes), 1)) 
                      for i, ds in enumerate(unique_dataset_sizes)}
    
    for subplot_idx, (pii_type, pii_label) in enumerate(zip(pii_types, pii_labels)):
        ax = axes[subplot_idx]
        
        # Filter data for this PII type
        if 'pii_type_standard' in df_filtered.columns:
            df_pii = df_filtered[df_filtered['pii_type_standard'] == pii_type].copy()
        else:
            df_pii = df_filtered[df_filtered['pii_type'] == pii_type].copy()
        
        if len(df_pii) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{pii_label}', fontsize=14, fontweight='bold')
            continue
        
        # Plot lines for each model/dataset combination
        unique_models = sorted(df_pii['model'].unique())
        for model in unique_models:
            for dataset_size in unique_dataset_sizes:
                # Filter data for this combination
                df_combo = df_pii[
                    (df_pii['model'] == model) & 
                    (df_pii['source_dataset_size'] == dataset_size)
                ].copy()
                
                if len(df_combo) == 0:
                    continue
                
                # Sort by pii_rate for proper line plotting
                df_combo = df_combo.sort_values('pii_rate')
                
                # Create label
                label = f'{model} (DS={dataset_size})'
                
                # Get line style based on model (1B = solid, 8B = dashed)
                linestyle = model_linestyles.get(model, '-')
                # Get color based on dataset size (consistent across models)
                color = dataset_colors.get(dataset_size, 'black')
                
                # Plot with asymmetric error bars
                yerr = [df_combo[y_err_lower_col].values, df_combo[y_err_upper_col].values]
                ax.errorbar(
                    df_combo['pii_rate'] * 100,  # Convert to percentage
                    df_combo[y_col],
                    yerr=yerr,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker='o',
                    capsize=3,
                    capthick=1,
                    markersize=5,
                    linewidth=2
                )
        
        # Customize subplot
        ax.set_xlabel('PII Level (%)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{pii_label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Set log scale for y-axis for relative risk metrics
        if 'relative' in metric_name.lower():
            ax.set_yscale('log')
        
        # Add legend with longer handles to show dashed lines properly
        ax.legend(fontsize=9, loc='best', handlelength=3)
    
    # Overall title
    epoch_display = 'Overfit' if epoch_label == 'overfit' else 'No Overfit'
    fig.suptitle(f'{metric_name} vs PII Level - {epoch_display}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def create_combined_plot(df_data, metric_name, y_col, y_err_col, y_label, epoch_label, output_path):
    """
    Create a plot with two subplots (name and MRN) showing lines for each model/dataset combination.
    
    Args:
        df_data: DataFrame with the data
        metric_name: Name of the metric (for title)
        y_col: Column name for y-axis values
        y_err_col: Column name for error bars (can be None)
        y_label: Label for y-axis
        epoch_label: 'overfit' or 'no overfit'
        output_path: Path to save the plot
    """
    # Filter by epoch_label
    df_filtered = df_data[df_data['epoch_label'] == epoch_label].copy()
    
    if len(df_filtered) == 0:
        print(f"No data for {metric_name} - {epoch_label}")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define PII types for subplots
    pii_types = ['name-patient', 'unit_no']
    pii_labels = ['Name', 'MRN']
    
    # Line styles: 1B = solid, 8B = dashed
    model_linestyles = {'1B': '-', '8B': '--'}
    
    # Colors for dataset sizes (consistent across models)
    unique_dataset_sizes = sorted(df_filtered['source_dataset_size'].unique())
    dataset_colors = {ds: plt.cm.tab10(i / max(len(unique_dataset_sizes), 1)) 
                      for i, ds in enumerate(unique_dataset_sizes)}
    
    for subplot_idx, (pii_type, pii_label) in enumerate(zip(pii_types, pii_labels)):
        ax = axes[subplot_idx]
        
        # Filter data for this PII type
        if 'pii_type_standard' in df_filtered.columns:
            df_pii = df_filtered[df_filtered['pii_type_standard'] == pii_type].copy()
        else:
            df_pii = df_filtered[df_filtered['pii_type'] == pii_type].copy()
        
        if len(df_pii) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{pii_label}', fontsize=14, fontweight='bold')
            continue
        
        # Plot lines for each model/dataset combination
        unique_models = sorted(df_pii['model'].unique())
        for model in unique_models:
            for dataset_size in unique_dataset_sizes:
                # Filter data for this combination
                df_combo = df_pii[
                    (df_pii['model'] == model) & 
                    (df_pii['source_dataset_size'] == dataset_size)
                ].copy()
                
                if len(df_combo) == 0:
                    continue
                
                # Sort by pii_rate for proper line plotting
                df_combo = df_combo.sort_values('pii_rate')
                
                # Create label
                label = f'{model} (DS={dataset_size})'
                
                # Get line style based on model (1B = solid, 8B = dashed)
                linestyle = model_linestyles.get(model, '-')
                # Get color based on dataset size (consistent across models)
                color = dataset_colors.get(dataset_size, 'black')
                
                # Plot with or without error bars
                if y_err_col and y_err_col in df_combo.columns:
                    ax.errorbar(
                        df_combo['pii_rate'] * 100,  # Convert to percentage
                        df_combo[y_col],
                        yerr=df_combo[y_err_col],
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker='o',
                        capsize=3,
                        capthick=1,
                        markersize=5,
                        linewidth=2
                    )
                else:
                    ax.plot(
                        df_combo['pii_rate'] * 100,  # Convert to percentage
                        df_combo[y_col],
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        marker='o',
                        markersize=5,
                    linewidth=2
                )
        
        # Customize subplot
        ax.set_xlabel('PII Level (%)', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{pii_label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Set log scale for x-axis for MRN subplots
        if pii_type == 'unit_no' or pii_label == 'MRN':
            ax.set_xscale('log')
        
        # Set log scale for y-axis for leakage metrics
        if 'leakage' in metric_name.lower() or 'relative' in metric_name.lower():
            ax.set_yscale('log')
        ax.set_xscale('log')
        
        # For MIA AUC plots: set y-axis limits and add chance line at 0.5
        if 'mia' in metric_name.lower() and 'auc' in metric_name.lower():
            ax.set_ylim(0.35, 1.05)
            # Add horizontal dashed line at chance level (0.5)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Chance (0.5)')
        
        # Add legend (after chance line if MIA AUC) with longer handles to show dashed lines properly
        ax.legend(fontsize=9, loc='best', handlelength=3)
    
    # Overall title
    epoch_display = 'Overfit' if epoch_label == 'overfit' else 'No Overfit'
    fig.suptitle(f'{metric_name} vs PII Level - {epoch_display}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    # Define directories
    plots_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/base-analysis'
    mia_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia'
    output_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/combined'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load overall_proba data (aggregate probability sums)
    print("Loading overall_proba data...")
    df_overall = load_overall_proba_data(plots_dir)
    
    # Load risk_stats data (average across individual samples with bootstrap CI)
    print("Loading absolute risk stats data...")
    df_abs_risk = load_risk_stats_data(plots_dir, risk_type='absolute')
    
    print("Loading relative risk stats data...")
    df_rel_risk = load_risk_stats_data(plots_dir, risk_type='relative')
    
    print("Loading MIA data...")
    df_mia = load_mia_data(mia_dir)
    
    if len(df_overall) == 0 and len(df_abs_risk) == 0 and len(df_rel_risk) == 0 and len(df_mia) == 0:
        print("No data found in CSV files!")
        return
    
    # === Plots from overall_proba files (aggregate probability sums) ===
    # These are "Absolute Leakage" and "Relative Leakage" (without "Average")
    if len(df_overall) > 0:
        # Aggregate train_pi_prob_mean and train_factor_mean by model, dataset_size, pii_rate, epoch_label, and pii_type_standard
        # This averages across prompts if multiple prompts exist (following overall_proba.py logic)
        def combine_bootstrap_stats_pi(group):
            # Average of bootstrap means
            mean = group['train_pi_prob_mean'].mean()
            # Overall std: sqrt of (mean of variances + variance of means)
            mean_of_vars = (group['train_pi_prob_std'] ** 2).mean()
            var_of_means = group['train_pi_prob_mean'].var(ddof=1) if len(group) > 1 else 0
            std = np.sqrt(mean_of_vars + var_of_means)
            return pd.Series({'train_pi_prob_mean': mean, 'train_pi_prob_std': std})
        
        def combine_bootstrap_stats_factor(group):
            # Average of bootstrap means
            mean = group['train_factor_mean'].mean()
            # Overall std: sqrt of (mean of variances + variance of means)
            mean_of_vars = (group['train_factor_std'] ** 2).mean()
            var_of_means = group['train_factor_mean'].var(ddof=1) if len(group) > 1 else 0
            std = np.sqrt(mean_of_vars + var_of_means)
            return pd.Series({'train_factor_mean': mean, 'train_factor_std': std})
        
        # Aggregate absolute leakage
        df_overall_abs = df_overall.groupby([
            'model', 'source_dataset_size', 'pii_rate', 'epoch_label', 'pii_type_standard'
        ]).apply(combine_bootstrap_stats_pi).reset_index()
        
        # Aggregate relative leakage
        df_overall_rel = df_overall.groupby([
            'model', 'source_dataset_size', 'pii_rate', 'epoch_label', 'pii_type_standard'
        ]).apply(combine_bootstrap_stats_factor).reset_index()
        
        # Create plots for absolute leakage (without "Average")
        print("\nCreating absolute leakage plots...")
        for epoch_label in ['no overfit', 'overfit']:
            output_path = os.path.join(output_dir, f'absolute_leakage_{epoch_label.replace(" ", "_")}.png')
            create_combined_plot(
                df_overall_abs,
                'Absolute Leakage',
                'train_pi_prob_mean',
                'train_pi_prob_std',
                'Absolute Leakage',
                epoch_label,
                output_path
            )
        
        # Create plots for relative leakage (without "Average")
        print("\nCreating relative leakage plots...")
        for epoch_label in ['no overfit', 'overfit']:
            output_path = os.path.join(output_dir, f'relative_leakage_{epoch_label.replace(" ", "_")}.png')
            create_combined_plot(
                df_overall_rel,
                'Relative Leakage',
                'train_factor_mean',
                'train_factor_std',
                'Relative Leakage',
                epoch_label,
                output_path
            )
    
    # === Plots from risk_stats files (average across individual samples) ===
    # These are "Average Absolute Leakage" and "Average Relative Leakage"
    
    # Create plots for absolute risk (average across individual samples) -> Average Absolute Leakage
    if len(df_abs_risk) > 0:
        print("\nCreating average absolute leakage plots...")
        # Compute error bars from CI (asymmetric)
        df_abs_risk['err_lower'] = df_abs_risk['mean'] - df_abs_risk['ci_lower']
        df_abs_risk['err_upper'] = df_abs_risk['ci_upper'] - df_abs_risk['mean']
        
        for epoch_label in ['no overfit', 'overfit']:
            output_path = os.path.join(output_dir, f'average_absolute_leakage_{epoch_label.replace(" ", "_")}.png')
            create_combined_plot_ci(
                df_abs_risk,
                'Average Absolute Leakage',
                'mean',
                'err_lower',
                'err_upper',
                'Average Absolute Leakage',
                epoch_label,
                output_path
            )
    
    # Create plots for relative risk (average across individual samples) -> Average Relative Leakage
    if len(df_rel_risk) > 0:
        print("\nCreating average relative leakage plots...")
        # Compute error bars from CI (asymmetric)
        df_rel_risk['err_lower'] = df_rel_risk['mean'] - df_rel_risk['ci_lower']
        df_rel_risk['err_upper'] = df_rel_risk['ci_upper'] - df_rel_risk['mean']
        
        for epoch_label in ['no overfit', 'overfit']:
            output_path = os.path.join(output_dir, f'average_relative_leakage_{epoch_label.replace(" ", "_")}.png')
            create_combined_plot_ci(
                df_rel_risk,
                'Average Relative Leakage',
                'mean',
                'err_lower',
                'err_upper',
                'Average Relative Leakage',
                epoch_label,
                output_path
            )
    
    # Create plots for MIA AUC
    if len(df_mia) > 0:
        print("\nCreating MIA AUC plots...")
        # The all_all files are already aggregated across prompts, so no prompt filtering needed
        # Group by model, dataset_size, pii_rate, epoch_label, and pii_type
        # (all_all files are already aggregated, but we group to handle any duplicates)
        df_mia_agg = df_mia.groupby([
            'model', 'source_dataset_size', 'pii_rate', 'epoch_label', 'pii_type'
        ]).agg({
            'auc': 'mean',
            'auc_std': lambda x: np.sqrt(np.mean(x**2))
        }).reset_index()
        
        # Map pii_type to standard names
        df_mia_agg['pii_type_standard'] = df_mia_agg['pii_type'].map({
            'name-patient': 'name-patient',
            'unit_no': 'unit_no'
        })
        
        for epoch_label in ['no overfit', 'overfit']:
            output_path = os.path.join(output_dir, f'mia_auc_{epoch_label.replace(" ", "_")}.png')
            create_combined_plot(
                df_mia_agg,
                'MIA AUC',
                'auc',
                'auc_std',
                'MIA AUC',
                epoch_label,
                output_path
            )
    
    print("\nAll plots created successfully!")


if __name__ == "__main__":
    main()
