from src._repo import REPO_ROOT
"""
Create summary plots for experimental recall outputs.
Plots TPR, FPR, and total recall as a function of pii_rate.
Different lines for different models (colors) and different taus (line styles).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# Base directory containing all experimental recall outputs
BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output"

# Path to threshold_fpr5_results.csv
# THRESHOLD_FPR5_CSV = os.path.join(BASE_DIR, "threshold_fpr5_results.csv")
THRESHOLD_FPR5_CSV = os.path.join(BASE_DIR, "threshold_extracted_fpr0.05_results.csv")

# Tau values to extract (fallback if threshold_fpr5_results.csv not available)
# TAU_VALUES = [0.3, 0.5, 0.7]
TAU_VALUES = [0.7]

# Line styles for different taus
TAU_LINESTYLES = {
    0.3: '-',      # solid
    0.5: '--',     # dashed
    0.7: '-.',     # dash-dot
}

# Markers for different (dataset_size, n_epochs) combinations
# Available markers: 'o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd'
# Note: Currently using simple 'o' marker for all lines
DS_EP_MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']

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

def get_metrics_at_tau(metrics_df, target_tau, tolerance=0.05):
    """
    Extract metrics at a specific tau threshold.
    Returns the row with threshold closest to target_tau.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return None
    
    # Find the threshold closest to target_tau
    metrics_df = metrics_df.copy()
    metrics_df['tau_diff'] = np.abs(metrics_df['threshold'] - target_tau)
    
    # Get the row with minimum difference
    closest_idx = metrics_df['tau_diff'].idxmin()
    closest_row = metrics_df.loc[closest_idx]
    
    # Check if the difference is within tolerance
    if closest_row['tau_diff'] > tolerance:
        # If no close threshold found, try interpolation
        # For now, just use the closest one
        pass
    
    return {
        'tau': target_tau,
        'actual_threshold': closest_row['threshold'],
        'TPR': closest_row['TPR'],
        'FPR': closest_row['FPR'],
        'total_recall': closest_row['total_recall'],
    }

def load_threshold_fpr5_results(csv_path):
    """
    Load threshold_fpr5_results.csv and return a dictionary mapping directory names to avg_threshold.
    
    Returns:
        Dictionary: {directory_name: avg_threshold}
    """
    threshold_map = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: threshold_fpr5_results.csv not found at {csv_path}")
        print("Will use fallback TAU_VALUES")
        return threshold_map
    
    try:
        df_fpr5 = pd.read_csv(csv_path)
        if 'directory' in df_fpr5.columns and 'avg_threshold' in df_fpr5.columns:
            # Group by directory and take the first avg_threshold (or average if multiple)
            for directory in df_fpr5['directory'].unique():
                matching_rows = df_fpr5[df_fpr5['directory'] == directory]
                if len(matching_rows) > 0:
                    # Take the first match (or average if multiple)
                    avg_thr = matching_rows.iloc[0]['avg_threshold']
                    threshold_map[directory] = avg_thr
            print(f"Loaded {len(threshold_map)} optimal thresholds from threshold_fpr5_results.csv")
        else:
            print(f"Warning: threshold_fpr5_results.csv missing required columns")
    except Exception as e:
        print(f"Error reading threshold_fpr5_results.csv: {e}")
        print("Will use fallback TAU_VALUES")
    
    return threshold_map

def collect_all_data(base_dir):
    """
    Collect all metrics data from all directories.
    Uses optimal thresholds from threshold_fpr5_results.csv when available.
    Returns a DataFrame with columns: model_size, pii_rate, tau, TPR, FPR, total_recall
    """
    all_data = []
    
    # Load optimal thresholds from threshold_fpr5_results.csv
    threshold_map = load_threshold_fpr5_results(THRESHOLD_FPR5_CSV)
    
    # Get all subdirectories
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} directories to process")
    
    for subdir in subdirs:
        dir_name = subdir.name
        params = parse_directory_name(dir_name)
        
        if params is None:
            print(f"Warning: Could not parse directory name: {dir_name}")
            continue
        
        # Look for metrics file
        metrics_file = subdir / "all_names_ll_computed_with_scores_metrics_by_threshold.csv"
        
        if not metrics_file.exists():
            print(f"Warning: Metrics file not found in {dir_name}")
            continue
        
        print(f"Processing {dir_name}...")
        
        try:
            metrics_df = pd.read_csv(metrics_file)
            
            # Determine which tau value to use for this directory
            if dir_name in threshold_map:
                # Use optimal threshold (avg_thr) directly from threshold_fpr5_results.csv
                tau_to_use = threshold_map[dir_name]
                print(f"  Using avg_thr from FPR5: {tau_to_use:.6f}")
            else:
                # Default to 0.7 if optimal threshold doesn't exist
                tau_to_use = 0.7
                print(f"  Using default tau: {tau_to_use}")
            
            # Find metrics at this threshold
            metrics_at_tau = get_metrics_at_tau(metrics_df, tau_to_use, tolerance=0.1)
            
            if metrics_at_tau is not None:
                all_data.append({
                    'dataset_size': params['dataset_size'],
                    'model_size': params['model_size'],
                    'pii_rate': params['pii_rate'],
                    'n_epochs': params['n_epochs'],
                    'tau': tau_to_use,
                    'actual_threshold': metrics_at_tau['actual_threshold'],
                    'TPR': metrics_at_tau['TPR'],
                    'FPR': metrics_at_tau['FPR'],
                    'total_recall': metrics_at_tau['total_recall'],
                })
            else:
                print(f"  Warning: Could not find metrics at threshold {tau_to_use:.6f}")
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data collected. Check that metrics files exist.")
    
    df = pd.DataFrame(all_data)
    return df

def create_summary_plots(df, output_path=None, threshold_map=None):
    """
    Create summary plots with 3 subplots: TPR, FPR, total_recall vs pii_rate.
    Different colors for (model_size, dataset_size, n_epochs) combinations.
    Different line styles for tau values.
    
    Args:
        df: DataFrame with metrics data
        output_path: Path to save the plot
        threshold_map: Dictionary mapping directory names to optimal thresholds (optional)
    """
    # Get unique values
    pii_rates = sorted(df['pii_rate'].unique())
    
    # Get all unique (model_size, dataset_size, n_epochs) combinations
    model_configs = df[['model_size', 'dataset_size', 'n_epochs']].drop_duplicates()
    model_configs = model_configs.sort_values(['model_size', 'dataset_size', 'n_epochs'])
    model_configs_list = model_configs.values.tolist()
    
    print(f"\nPII rates found: {pii_rates}")
    print(f"\nModel configurations found:")
    for model, ds, ep in model_configs_list:
        print(f"  {model}-{ds}-{ep}")
    
    # Assign colors to (model_size, dataset_size, n_epochs) combinations
    # Use a colormap to generate distinct colors
    n_configs = len(model_configs_list)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_configs)))
    if n_configs > 10:
        # Use a different colormap for more colors
        colors = plt.cm.tab20(np.linspace(0, 1, n_configs))
    
    model_config_colors = {}
    for idx, (model, ds, ep) in enumerate(model_configs_list):
        model_config_colors[(model, ds, ep)] = colors[idx]
    
    print(f"\nColor assignments:")
    for (model, ds, ep), color in model_config_colors.items():
        print(f"  {model}-{ds}-{ep}: {color}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: TPR vs pii_rate
    ax = axes[0]
    for model, ds_size, n_ep in model_configs_list:
        # Get all data for this model configuration (one line per configuration)
        model_data = df[(df['model_size'] == model) & 
                       (df['dataset_size'] == ds_size) & 
                       (df['n_epochs'] == n_ep)]
        if len(model_data) > 0:
            # Sort by pii_rate
            model_data = model_data.sort_values('pii_rate')
            color = model_config_colors[(model, ds_size, n_ep)]
            # Get threshold value (use first one as representative)
            tau_val = model_data.iloc[0]['tau']
            # Create label with threshold: model-dataset_size-n_epochs (τ=threshold)
            if tau_val < 1:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.4f})"
            else:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.2f})"
            ax.plot(model_data['pii_rate'], model_data['TPR'],
                    color=color, linestyle='-',
                    marker='o', linewidth=2, markersize=5, label=label)
    
    ax.set_xlabel('PII Rate', fontsize=12)
    ax.set_ylabel('TPR (True Positive Rate)', fontsize=12)
    ax.set_title('TPR vs PII Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: FPR vs pii_rate
    ax = axes[1]
    for model, ds_size, n_ep in model_configs_list:
        # Get all data for this model configuration (one line per configuration)
        model_data = df[(df['model_size'] == model) & 
                       (df['dataset_size'] == ds_size) & 
                       (df['n_epochs'] == n_ep)]
        if len(model_data) > 0:
            # Sort by pii_rate
            model_data = model_data.sort_values('pii_rate')
            color = model_config_colors[(model, ds_size, n_ep)]
            # Get threshold value (use first one as representative)
            tau_val = model_data.iloc[0]['tau']
            # Create label with threshold: model-dataset_size-n_epochs (τ=threshold)
            if tau_val < 1:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.4f})"
            else:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.2f})"
            ax.plot(model_data['pii_rate'], model_data['FPR'],
                    color=color, linestyle='-',
                    marker='o', linewidth=2, markersize=5, label=label)
    
    ax.set_xlabel('PII Rate', fontsize=12)
    ax.set_ylabel('FPR (False Positive Rate)', fontsize=12)
    ax.set_title('FPR vs PII Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: Total Recall vs pii_rate
    ax = axes[2]
    for model, ds_size, n_ep in model_configs_list:
        # Get all data for this model configuration (one line per configuration)
        model_data = df[(df['model_size'] == model) & 
                       (df['dataset_size'] == ds_size) & 
                       (df['n_epochs'] == n_ep)]
        if len(model_data) > 0:
            # Sort by pii_rate
            model_data = model_data.sort_values('pii_rate')
            color = model_config_colors[(model, ds_size, n_ep)]
            # Get threshold value (use first one as representative)
            tau_val = model_data.iloc[0]['tau']
            # Create label with threshold: model-dataset_size-n_epochs (τ=threshold)
            if tau_val < 1:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.4f})"
            else:
                label = f"{model}-{ds_size}-{n_ep} (τ={tau_val:.2f})"
            ax.plot(model_data['pii_rate'], model_data['total_recall'],
                    color=color, linestyle='-',
                    marker='o', linewidth=2, markersize=5, label=label)
    
    ax.set_xlabel('PII Rate', fontsize=12)
    ax.set_ylabel('Total Recall', fontsize=12)
    ax.set_title('Total Recall vs PII Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_ylim(-0.005, 0.1)
    
    # Add overall title
    # Check if we're using optimal thresholds
    if threshold_map is not None and len(threshold_map) > 0:
        title = 'Experimental Recall Metrics Summary (Using Optimal Thresholds from FPR5)'
    else:
        title = 'Experimental Recall Metrics Summary'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "summary_plot_tpr_fpr_recall.png")
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved summary plot to: {output_path}")
    
    return fig

def main():
    """Main function to collect data and create plots."""
    print("="*80)
    print("Creating Summary Plots for Experimental Recall Outputs")
    print("="*80)
    
    # Collect all data
    print("\nCollecting data from all directories...")
    df = collect_all_data(BASE_DIR)
    
    print(f"\nCollected {len(df)} data points")
    print("\nData summary:")
    print(df.groupby(['model_size', 'pii_rate', 'tau']).size().unstack(fill_value=0))
    
    # Save aggregated data to CSV
    output_csv = os.path.join(BASE_DIR, "summary_data.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved aggregated data to: {output_csv}")
    
    # Reload threshold map for title
    threshold_map = load_threshold_fpr5_results(THRESHOLD_FPR5_CSV)
    
    # Create plots
    print("\nCreating summary plots...")
    fig = create_summary_plots(df, threshold_map=threshold_map)
    
    # Show plot
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
