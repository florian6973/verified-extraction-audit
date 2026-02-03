#!/usr/bin/env python3
from src._repo import REPO_ROOT
"""
Plot FPR vs TPR for specific configurations at two budgets (10^4 and 10^7).
For 1B model, large (100), no overfit (3) & overfit (10), 0.01 PII and 1 PII.
Computes FPR/TPR for tau values from 0 to 1 in 0.1 increments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import json
import pickle

# Base directory containing all experimental recall outputs
BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output"

# Output directory for plots
OUTPUT_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output/fpr_tpr_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target budgets
BUDGETS = [1e4, 1e7]

# Tau values to compute (0 to 1 in 0.1 increments)
TAU_VALUES = np.arange(0, 1.1, 0.1)

# Configurations to plot
# Format: (dataset_size, model_size, pii_rate, n_epochs, k)
CONFIGURATIONS = [
    (100, "1B", 0.01, 3, 10000),   # no overfit, 0.01 PII
    # (100, "1B", 0.01, 10, 10000),  # overfit, 0.01 PII
    (100, "1B", 0.05, 3, 10000),    # no overfit, 1 PII
    (100, "1B", 0.1, 3, 10000),    # no overfit, 1 PII
    (100, "1B", 1.0, 3, 10000),    # no overfit, 1 PII
    # (100, "1B", 1.0, 10, 10000),   # overfit, 1 PII
]

# Column names
PI_COL = "p_ft_Name: "
SPLIT_COL = "split_x"  # "train" = member, "val" = non-member
SCORE_COL = "score_oof_member_proba"  # verifier score


def get_directory_name(dataset_size, model_size, pii_rate, n_epochs, k):
    """Generate directory name from configuration."""
    return f"{dataset_size}_{model_size}_{pii_rate}_{n_epochs}_{k}"


def get_scores_file_path(directory, model_size, dataset_size, pii_rate, n_epochs):
    """Get the path to the scores CSV file."""
    base_path = Path(BASE_DIR) / directory
    # Format pii_rate to match file naming (0.01 -> "0.01", 1.0 -> "1.0")
    if pii_rate == 0.01:
        pii_rate_str = "0.01"
    elif pii_rate == 0.05:
        pii_rate_str = "0.05"
    elif pii_rate == 0.1:
        pii_rate_str = "0.1"
    elif pii_rate == 1.0:
        pii_rate_str = "1.0"
    else:
        pii_rate_str = f"{pii_rate:.2f}" if pii_rate < 1.0 else f"{pii_rate:.1f}"
    scores_file = base_path / f"scores_{model_size}_{dataset_size}_pii_rate_{pii_rate_str}_n_epochs_{n_epochs}_p.csv"
    return scores_file


def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    """P(name i appears at least once) after N i.i.d. draws."""
    return 1.0 - np.power((1.0 - pi), N)


def compute_fpr_curve_extracted(df_nonmem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Extracted-stream FPR (selection-aware):
      sum_{i in V} P(E_i;N) q_i  /  sum_{i in V} P(E_i;N)
    """
    pi_nm = df_nonmem[PI_COL].to_numpy(dtype=float)
    q_nm = df_nonmem[q_col].to_numpy(dtype=float)

    fpr_ext = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        denom = np.sum(pE_nm)
        num = np.sum(pE_nm * q_nm)
        fpr_ext.append(num / denom if denom > 0 else np.nan)

    return np.array(fpr_ext)


def compute_tpr_curve_extracted(df_mem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Extracted-stream TPR (selection-aware):
      sum_{i in M} P(E_i;N) q_i  /  sum_{i in M} P(E_i;N)
    """
    pi_m = df_mem[PI_COL].to_numpy(dtype=float)
    q_m = df_mem[q_col].to_numpy(dtype=float)

    tpr_ext = []
    for N in budgets:
        pE_m = prob_extracted_at_least_once(pi_m, N)
        denom = np.sum(pE_m)
        num = np.sum(pE_m * q_m)
        tpr_ext.append(num / denom if denom > 0 else np.nan)

    return np.array(tpr_ext)


def collect_fpr_tpr_curve(config, target_budgets):
    """
    Collect FPR and TPR values for all tau values for a given configuration.
    
    Args:
        config: Tuple of (dataset_size, model_size, pii_rate, n_epochs, k)
        target_budgets: List of target budget values
        
    Returns:
        Dictionary with config info and list of (tau, fpr, tpr) tuples for each budget
    """
    dataset_size, model_size, pii_rate, n_epochs, k = config
    directory = get_directory_name(dataset_size, model_size, pii_rate, n_epochs, k)
    
    # Get scores file path
    scores_file = get_scores_file_path(directory, model_size, dataset_size, pii_rate, n_epochs)
    
    if not scores_file.exists():
        print(f"Warning: Scores file not found: {scores_file}")
        return None
    
    print(f"Processing {directory}: loading {scores_file.name}")
    
    # Load and clean data
    try:
        df = pd.read_csv(scores_file).copy()
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return None
    
    # Coerce types
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    df[PI_COL] = pd.to_numeric(df[PI_COL], errors="coerce")
    
    # Drop rows with missing pi
    df = df.dropna(subset=[PI_COL])
    
    # Split members/nonmembers
    split_lower = df[SPLIT_COL].astype(str).str.lower()
    members_mask = split_lower == "train"
    nonmembers_mask = split_lower == "val"
    
    if members_mask.sum() == 0:
        print(f"  Warning: No member rows found")
        return None
    if nonmembers_mask.sum() == 0:
        print(f"  Warning: No non-member rows found")
        return None
    
    print(f"  Loaded {len(df)} rows ({members_mask.sum()} members, {nonmembers_mask.sum()} non-members)")
    
    # Compute FPR and TPR for each tau value at each budget
    results = {}
    for budget in target_budgets:
        results[budget] = []
    
    for tau in TAU_VALUES:
        # Compute q for this tau
        q_col = f"q_tau_{tau}"
        df[q_col] = (df[SCORE_COL] >= tau).astype(int)
        
        # Get members and nonmembers with q column
        members_tau = df[members_mask].copy()
        nonmembers_tau = df[nonmembers_mask].copy()
        
        # Compute FPR and TPR at each target budget
        budgets_array = np.array(target_budgets)
        fpr_values = compute_fpr_curve_extracted(nonmembers_tau, budgets_array, q_col)
        tpr_values = compute_tpr_curve_extracted(members_tau, budgets_array, q_col)
        
        # Store results for each budget
        for budget_idx, budget in enumerate(target_budgets):
            fpr = fpr_values[budget_idx]
            tpr = tpr_values[budget_idx]
            if not (np.isnan(fpr) or np.isnan(tpr)):
                results[budget].append((tau, fpr, tpr))
    
    # Sort by tau for each budget
    for budget in target_budgets:
        results[budget].sort(key=lambda x: x[0])
    
    return {
        'config': config,
        'directory': directory,
        'results': results  # Dictionary mapping budget -> list of (tau, fpr, tpr) tuples
    }


def plot_fpr_tpr_curves(all_data, budgets, output_dir):
    """
    Plot TPR vs FPR curves for two budgets side by side.
    
    Args:
        all_data: List of model data dictionaries
        budgets: List of budget values
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Colors for different PII rates
    # colors = {
    #     0.01: '#1f77b4',  # blue

    #     1.0: '#d62728'     # red
    # }

    colors = {}
    n_pii = 4
    if n_pii > 0:
        for idx, pii_rate in enumerate([0.01, 0.05, 0.1, 1.0]):
            # Evenly space colors: first gets 0.5, last gets 1.0
            if n_pii == 1:
                colormap_val = 0.75
            else:
                colormap_val = 0.5 + 0.5 * (idx / (n_pii - 1))
            colors[pii_rate] = plt.cm.Reds(colormap_val)
    
    # Linestyles for overfit vs no overfit
    linestyles = {
        3: '-',   # no overfit: solid
        10: '--'  # overfit: dashed
    }
    
    # Check if only one unique n_epochs value exists
    unique_epochs = set()
    for model_data in all_data:
        if model_data is not None:
            config = model_data['config']
            unique_epochs.add(config[3])  # n_epochs is at index 3
    single_epoch_type = len(unique_epochs) == 1
    
    for budget_idx, budget in enumerate(budgets):
        ax = axes[budget_idx]
        
        for model_data in all_data:
            if model_data is None or budget not in model_data['results']:
                continue
            
            config = model_data['config']
            curve_data = model_data['results'][budget]
            
            if len(curve_data) == 0:
                continue
            
            dataset_size, model_size, pii_rate, n_epochs, k = config
            
            # Extract FPR and TPR values
            fprs = np.array([x[1] for x in curve_data])
            tprs = np.array([x[2] for x in curve_data])
            
            # Sort by FPR for proper curve plotting and AUC calculation
            sort_idx = np.argsort(fprs)
            fprs_sorted = fprs[sort_idx]
            tprs_sorted = tprs[sort_idx]
            
            # Compute AUC (Area Under Curve) using trapezoidal integration
            auc = np.trapz(tprs_sorted, fprs_sorted)
            
            # Create label
            if pii_rate == 0.01:
                pii_label = "PII=1%"
            elif pii_rate == 1.0:
                pii_label = "PII=100%"
            else:
                pii_label = f"PII={pii_rate*100:.0f}%"
            
            # If only one epoch type, don't include epoch info in legend
            if single_epoch_type:
                label = f"Estimated, {pii_label} (AUC={auc:.2f})"
            else:
                if n_epochs == 3:
                    ep_label = "no overfit"
                elif n_epochs == 10:
                    ep_label = "overfit"
                else:
                    ep_label = f"ep{n_epochs}"
                label = f"{ep_label}, {pii_label} (AUC={auc:.2f})"
            
            # Plot the curve (sorted by FPR for proper ROC-like curve)
            color = colors[pii_rate]
            linestyle = linestyles[n_epochs]
            
            ax.plot(fprs_sorted, tprs_sorted, marker='o', linestyle=linestyle, label=label,
                    color=color, linewidth=3, markersize=6, alpha=0.9, markevery=1)
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random classifier')
        
        # Format budget for title
        if budget >= 1e6:
            budget_str = f"10⁷"
        elif budget >= 1e4:
            budget_str = f"10⁴"
        elif budget >= 1e3:
            budget_str = f"{int(budget/1e3)}×10³"
        else:
            budget_str = f"{int(budget)}"
        
        # Formatting
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=16)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=16)
        ax.set_title(f'Budget = {budget_str} queries', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='lower right', fontsize=13, framealpha=0.95, handlelength=3)  # Larger legend with longer handles for dashed lines
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
    
    fig.suptitle('FPR vs TPR for 1B model, large (100)', 
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save as PNG, PDF and SVG
    base_name = "fpr_tpr_budgets_1B_large"
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    svg_path = os.path.join(output_dir, f"{base_name}.svg")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSaved plots to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  SVG: {svg_path}")
    
    plt.close()


def main():
    if os.path.exists(BASE_DIR):
        print("Collecting FPR/TPR data for specified configurations...")
        print(f"Computing for tau values: {TAU_VALUES}")
        print("="*80)
        
        # Collect data for all configurations
        all_model_data = []
        
        for config in CONFIGURATIONS:
            model_data = collect_fpr_tpr_curve(config, BUDGETS)
            if model_data is not None:
                all_model_data.append(model_data)
        
        if len(all_model_data) == 0:
            print("Error: No valid data collected for any configuration")
            return 1

        print(all_model_data)
        # save all_model_data to a pickle file
        with open('all_model_data_fpr_tpr_budgets.pkl', 'wb') as f:
            pickle.dump(all_model_data, f)
    
    # read all_model_data from the pickle file
    with open('all_model_data_fpr_tpr_budgets.pkl', 'rb') as f:
        all_model_data = pickle.load(f)

    # exit()
    
    # Plot the curves
    print("\n" + "="*80)
    print("Generating plots...")
    plot_fpr_tpr_curves(all_model_data, BUDGETS, OUTPUT_DIR)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
