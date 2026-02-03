#!/usr/bin/env python3
from src._repo import REPO_ROOT
"""
Plot TPR as a function of FPR at a given budget (like AUROC curve).
For a list of models, varying tau values to plot the curve.

Usage:
    python fpr_tpr_theo.py --budget 10000 --models 10_1B_0.1_3_500000 10_1B_0.05_3_500000
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.lines import Line2D

# Base directory containing all experimental recall outputs
BASE_DIR = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-output"


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


def find_tau_files(plots_dir):
    """
    Find all theoretical_curves_tau_*.csv files in the plots directory.
    
    Returns:
        List of tuples: [(tau_value, file_path), ...]
    """
    tau_files = []
    if not plots_dir.exists():
        return tau_files
    
    pattern = re.compile(r'theoretical_curves_tau_([\d.]+)\.csv')
    
    for file_path in plots_dir.glob("theoretical_curves_tau_*.csv"):
        match = pattern.match(file_path.name)
        if match:
            try:
                tau_val = float(match.group(1))
                tau_files.append((tau_val, file_path))
            except ValueError:
                continue
    
    # Sort by tau value
    tau_files.sort(key=lambda x: x[0])
    return tau_files


def get_fpr_tpr_at_budget(theory_file, target_budget):
    """
    Get FPR and TPR at a specific budget from a theoretical curves file.
    
    Args:
        theory_file: Path to theoretical_curves_tau_*.csv file
        target_budget: Target budget value
        
    Returns:
        Tuple of (fpr, tpr) or (None, None) if not found
    """
    try:
        df = pd.read_csv(theory_file)
        
        # Filter out infinite budgets
        df_finite = df[np.isfinite(df['budget'])].copy()
        
        if len(df_finite) == 0:
            return None, None
        
        # Find the row with budget closest to target_budget
        budget_idx = np.argmin(np.abs(df_finite['budget'] - target_budget))
        row = df_finite.iloc[budget_idx]
        
        # Get FPR and TPR (with verification)
        fpr = row.get('fpr_extracted_with_verification', np.nan)
        tpr = row.get('tpr_extracted_with_verification', np.nan)
        
        if pd.isna(fpr) or pd.isna(tpr):
            return None, None
        
        return float(fpr), float(tpr)
    except Exception as e:
        print(f"  Error reading {theory_file}: {e}")
        return None, None


def collect_fpr_tpr_curve(directory, base_dir, target_budget):
    """
    Collect FPR and TPR values for all tau values for a given model directory.
    
    Args:
        directory: Directory name (e.g., "10_1B_0.1_3_500000")
        base_dir: Base directory path
        target_budget: Target budget value
        
    Returns:
        Dictionary with model info and list of (tau, fpr, tpr) tuples
    """
    base_path = Path(base_dir)
    plots_dir = base_path / directory / "plots_theory"
    
    if not plots_dir.exists():
        print(f"Warning: plots_theory directory not found for {directory}")
        return None
    
    # Find all tau files
    tau_files = find_tau_files(plots_dir)
    
    if len(tau_files) == 0:
        print(f"Warning: No theoretical curves files found for {directory}")
        return None
    
    print(f"Processing {directory}: found {len(tau_files)} tau files")
    
    # Parse directory name for model info
    params = parse_directory_name(directory)
    if params is None:
        print(f"Warning: Could not parse directory name: {directory}")
        return None
    
    # Collect FPR and TPR for each tau
    curve_data = []
    for tau_val, theory_file in tau_files:
        fpr, tpr = get_fpr_tpr_at_budget(theory_file, target_budget)
        if fpr is not None and tpr is not None:
            curve_data.append((tau_val, fpr, tpr))
    
    if len(curve_data) == 0:
        print(f"Warning: No valid FPR/TPR data found for {directory}")
        return None
    
    # Sort by tau (should already be sorted, but just in case)
    curve_data.sort(key=lambda x: x[0])
    
    return {
        'directory': directory,
        'params': params,
        'curve_data': curve_data  # List of (tau, fpr, tpr) tuples
    }


def plot_fpr_tpr_curves(model_data_list, target_budget, output_path=None):
    """
    Plot TPR vs FPR curves for multiple models.
    
    Args:
        model_data_list: List of dictionaries with model data and curve points
        target_budget: Target budget value (for title)
        output_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data_list)))
    
    for idx, model_data in enumerate(model_data_list):
        if model_data is None or len(model_data['curve_data']) == 0:
            continue
        
        params = model_data['params']
        curve_data = model_data['curve_data']
        
        # Extract FPR and TPR values
        taus = [x[0] for x in curve_data]
        fprs = [x[1] for x in curve_data]
        tprs = [x[2] for x in curve_data]
        
        # Create model label
        n_ep = params['n_epochs']
        pii_rate = params['pii_rate']
        model_size = params['model_size']
        dataset_size = params['dataset_size']
        
        if n_ep in [2, 3]:
            ep_label = "no_overfit"
        elif n_ep == 10:
            ep_label = "overfit"
        else:
            ep_label = f"ep{n_ep}"
        
        pii_pct = int(pii_rate * 100)
        model_label = f"{dataset_size}_{model_size}_{ep_label}_PII{pii_pct}%"
        
        # Plot the curve
        color = colors[idx]
        ax.plot(fprs, tprs, marker='o', linestyle='-', label=model_label, 
                color=color, linewidth=2, markersize=4)
        
        # Add tau annotations for a few key points
        if len(taus) > 0:
            # Annotate min, max, and middle tau values
            indices_to_annotate = [0, len(taus)//2, len(taus)-1]
            for i in indices_to_annotate:
                if i < len(taus):
                    ax.annotate(f'τ={taus[i]:.3f}', 
                              xy=(fprs[i], tprs[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')
    
    # Format budget for title
    if target_budget >= 1e6:
        budget_str = f"{target_budget/1e6:.0f}×10⁶"
    elif target_budget >= 1e3:
        budget_str = f"{target_budget/1e3:.0f}×10³"
    else:
        budget_str = f"{target_budget:.0f}"
    
    # Formatting
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'TPR vs FPR at Budget = {budget_str}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot TPR vs FPR curves at a given budget by varying tau',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot for specific models at budget 10000
  python fpr_tpr_theo.py --budget 10000 --models 10_1B_0.1_3_500000 10_1B_0.05_3_500000
  
  # Plot for all models matching a pattern
  python fpr_tpr_theo.py --budget 100000 --pattern "10_1B_0.*_3_500000"
        """
    )
    parser.add_argument('--budget', type=float, required=True,
                       help='Target budget value (e.g., 10000, 100000)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='List of model directory names')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Pattern to match directory names (regex)')
    parser.add_argument('--base-dir', type=str, default=BASE_DIR,
                       help=f'Base directory (default: {BASE_DIR})')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the plot (default: fpr_tpr_budget_{budget}.png)')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Error: Base directory not found: {args.base_dir}")
        return 1
    
    # Determine which directories to process
    directories = []
    if args.models:
        directories = args.models
    elif args.pattern:
        import re
        pattern = re.compile(args.pattern)
        for dir_path in base_path.iterdir():
            if dir_path.is_dir() and pattern.match(dir_path.name):
                directories.append(dir_path.name)
    else:
        print("Error: Must provide either --models or --pattern")
        return 1
    
    if len(directories) == 0:
        print("Error: No directories found to process")
        return 1
    
    print(f"Processing {len(directories)} model directories at budget={args.budget}")
    print("="*80)
    
    # Collect data for each model
    model_data_list = []
    for directory in directories:
        model_data = collect_fpr_tpr_curve(directory, args.base_dir, args.budget)
        if model_data is not None:
            model_data_list.append(model_data)
    
    if len(model_data_list) == 0:
        print("Error: No valid data collected for any model")
        return 1
    
    # Determine output path
    if args.output is None:
        budget_str = f"{args.budget:.0e}".replace('+', '')
        args.output = f"fpr_tpr_budget_{budget_str}.png"
    
    # Plot the curves
    plot_fpr_tpr_curves(model_data_list, args.budget, args.output)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
