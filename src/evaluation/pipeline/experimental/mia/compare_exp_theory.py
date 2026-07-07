"""
Compare experimental and theoretical results for TPR, FPR, and Recall as a function of tau.
Reads outputs from evaluate_scores.py and attack_curves.py.
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.evaluation.pipeline.experimental.config_loader import load_config
from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir

parser = argparse.ArgumentParser(description='Compare experimental and theoretical results')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

model = config['filters']['model']
dataset_size = config['filters']['dataset_size']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']
budget = config['inputs']['budget']

output_dir = get_output_dir(config)

# Paths to input CSVs
# There are only 4 possible files:
# 1. all_names_ll_computed_with_scores_metrics_by_threshold.csv (experiment all, no bootstrap)
# 2. all_names_ll_computed_with_scores_metrics_by_threshold_with_bootstrap.csv (experiment all, with bootstrap)
# 3. all_names_ll_computed_with_scores_metrics_by_threshold_without_others.csv (experiment train/val only, no bootstrap)
# 4. all_names_ll_computed_with_scores_metrics_by_threshold_without_others_with_bootstrap.csv (experiment train/val only, with bootstrap)

# Load experiment all: prefer bootstrap version if available
exp_csv_base = os.path.join(output_dir, f"all_names_ll_computed_with_scores_metrics_by_threshold.csv")
exp_csv_bootstrap = os.path.join(output_dir, f"all_names_ll_computed_with_scores_metrics_by_threshold_with_bootstrap.csv")
if os.path.exists(exp_csv_bootstrap):
    exp_csv = exp_csv_bootstrap
    print(f"Loading experimental results (all) from: {exp_csv} (with bootstrap)")
else:
    exp_csv = exp_csv_base
    print(f"Loading experimental results (all) from: {exp_csv} (no bootstrap)")

# Load experiment train/val only: prefer bootstrap version if available
exp_csv_filtered_base = os.path.join(output_dir, f"all_names_ll_computed_with_scores_metrics_by_threshold_without_others.csv")
exp_csv_filtered_bootstrap = os.path.join(output_dir, f"all_names_ll_computed_with_scores_metrics_by_threshold_with_bootstrap_without_others.csv")
if os.path.exists(exp_csv_filtered_bootstrap):
    exp_csv_filtered = exp_csv_filtered_bootstrap
    print(f"Loading experimental results (train/val only) from: {exp_csv_filtered} (with bootstrap)")
elif os.path.exists(exp_csv_filtered_base):
    exp_csv_filtered = exp_csv_filtered_base
    print(f"Loading experimental results (train/val only) from: {exp_csv_filtered} (no bootstrap)")
else:
    exp_csv_filtered = None
    print(f"Experimental results (train/val only) not found")
# Theoretical results from attack_curves.py
# Need to find the theoretical curves CSV - it's saved with tau in the filename
# We'll need to read multiple tau files or find a way to get all tau values
plots_dir = os.path.join(output_dir, "plots_theory")
print(f"Loading theoretical results from: {plots_dir}")

# Load experimental data (normal)
if not os.path.exists(exp_csv):
    # raise FileNotFoundError(f"Experimental CSV not found: {exp_csv}")
    print(f"Warning: {exp_csv} not found. Skipping.")
    # exit(1)
    exit()

df_exp = pd.read_csv(exp_csv)
print(f"Loaded experimental data: {len(df_exp)} thresholds")
print(f"Experimental columns: {df_exp.columns.tolist()}")

# Check for bootstrap CI columns
has_bootstrap_exp = False
bootstrap_cols_exp = {}
if 'TPR_ci_lower' in df_exp.columns and 'TPR_ci_upper' in df_exp.columns:
    has_bootstrap_exp = True
    bootstrap_cols_exp = {
        'TPR': ('TPR_ci_lower', 'TPR_ci_upper'),
        'FPR': ('FPR_ci_lower', 'FPR_ci_upper') if 'FPR_ci_lower' in df_exp.columns and 'FPR_ci_upper' in df_exp.columns else None,
        'total_recall': ('total_recall_ci_lower', 'total_recall_ci_upper') if 'total_recall_ci_lower' in df_exp.columns and 'total_recall_ci_upper' in df_exp.columns else None,
    }
    print(f"Found bootstrap confidence intervals in experimental data")

# Load experimental data (filtered, without others - closer to FPR theory)
df_exp_filtered = None
has_bootstrap_filtered = False
bootstrap_cols_filtered = {}
if os.path.exists(exp_csv_filtered):
    df_exp_filtered = pd.read_csv(exp_csv_filtered)
    print(f"Loaded filtered experimental data: {len(df_exp_filtered)} thresholds")
    print(f"Filtered experimental columns: {df_exp_filtered.columns.tolist()}")
    
    # Check for bootstrap CI columns in filtered data
    if 'TPR_ci_lower' in df_exp_filtered.columns and 'TPR_ci_upper' in df_exp_filtered.columns:
        has_bootstrap_filtered = True
        bootstrap_cols_filtered = {
            'TPR': ('TPR_ci_lower', 'TPR_ci_upper'),
            'FPR': ('FPR_ci_lower', 'FPR_ci_upper') if 'FPR_ci_lower' in df_exp_filtered.columns and 'FPR_ci_upper' in df_exp_filtered.columns else None,
            'total_recall': ('total_recall_ci_lower', 'total_recall_ci_upper') if 'total_recall_ci_lower' in df_exp_filtered.columns and 'total_recall_ci_upper' in df_exp_filtered.columns else None,
        }
        print(f"Found bootstrap confidence intervals in filtered experimental data")
else:
    print(f"Warning: {exp_csv_filtered} not found. Will only plot normal experimental curve.")

# Load theoretical data for all available tau values
# Scan the plots_dir for all theoretical_curves_tau_*.csv files
theory_metrics = []
theory_files = []

# Find all theoretical curve files
if os.path.exists(plots_dir):
    plots_path = Path(plots_dir)
    for theory_file in plots_path.glob("theoretical_curves_tau_*.csv"):
        theory_files.append(str(theory_file))
    
    print(f"Found {len(theory_files)} theoretical curve files")
    
    # Extract tau values from filenames and sort
    tau_file_pairs = []
    for theory_file in theory_files:
        # Extract tau value from filename: theoretical_curves_tau_XX.csv
        match = re.search(r'theoretical_curves_tau_([\d.]+)\.csv', os.path.basename(theory_file))
        if match:
            try:
                tau_val = float(match.group(1))
                tau_file_pairs.append((tau_val, theory_file))
            except ValueError:
                print(f"Warning: Could not parse tau value from {theory_file}")
    
    # Sort by tau value
    tau_file_pairs.sort(key=lambda x: x[0])
    print(f"Found tau values: {[t[0] for t in tau_file_pairs]}")
else:
    print(f"Warning: Theoretical plots directory not found: {plots_dir}")

# Use budget = 10^4 as reference point, or asymptotic values
target_budget = budget

# Load data for each tau value found
for tau, theory_csv in tau_file_pairs:
    if os.path.exists(theory_csv):
        df_tau = pd.read_csv(theory_csv)
        print(f"Loaded theoretical data for tau={tau}: {len(df_tau)} budget points")
        print(f"  Available columns: {df_tau.columns.tolist()}")
        
        # Find the row with budget closest to target_budget
        # Exclude inf values from the search
        finite_mask = np.isfinite(df_tau['budget'])
        df_finite = df_tau[finite_mask].copy()
        
        if len(df_finite) > 0:
            budget_idx = np.argmin(np.abs(df_finite['budget'] - target_budget))
            row = df_finite.iloc[budget_idx]
            actual_budget = row['budget']
            print(f"  Using values at budget={actual_budget:.0f} (closest to target={target_budget:.0f})")
        else:
            # Fallback: use first row if no finite budgets
            row = df_tau.iloc[0]
            print(f"  Warning: No finite budgets found, using first row with budget={row['budget']}")
        
        # Get the values - check all possible column names
        tpr_val = row.get('tpr_extracted_with_verification', np.nan)
        fpr_val = row.get('fpr_extracted_with_verification', np.nan)
        
        # For recall, the theoretical recall is: (1/|M|) sum_{i in M} P(E_i;N) q_i
        # This should match the experimental total_recall which is TP/total_train_names
        # Check if the column exists
        recall_val = row.get('recall_with_verification', np.nan)
        if np.isnan(recall_val):
            # Try alternative column names
            recall_val = row.get('recall', np.nan)
        
        print(f"  Raw values - TPR: {tpr_val}, FPR: {fpr_val}, Recall: {recall_val}")
        
        theory_metrics.append({
            'tau': tau,
            'TPR': tpr_val,
            'FPR': fpr_val,
            'recall': recall_val,
        })
        
        print(f"  tau={tau}: TPR={theory_metrics[-1]['TPR']:.4f}, FPR={theory_metrics[-1]['FPR']:.4f}, Recall={theory_metrics[-1]['recall']:.4f}")
    else:
        print(f"Warning: Theoretical CSV not found: {theory_csv}")

if not theory_metrics or all(np.isnan([m['TPR'] for m in theory_metrics])):
    raise FileNotFoundError(f"No valid theoretical data found in {plots_dir}")

df_theory_summary = pd.DataFrame(theory_metrics)
# Sort by tau to ensure proper plotting order
df_theory_summary = df_theory_summary.sort_values('tau').reset_index(drop=True)

# Prepare experimental data - use tau as threshold
# The experimental CSV has 'threshold' column, we'll use that as tau
df_exp_plot = df_exp.copy()
df_exp_plot['tau'] = df_exp_plot['threshold']

# Prepare filtered experimental data
df_exp_filtered_plot = None
if df_exp_filtered is not None:
    df_exp_filtered_plot = df_exp_filtered.copy()
    df_exp_filtered_plot['tau'] = df_exp_filtered_plot['threshold']

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Colors
exp_color = 'blue'
exp_filtered_color = 'green'
theory_color = 'red'
exp_marker = 'o'
exp_filtered_marker = '^'
theory_marker = 's'

# Plot 1: TPR vs tau
# Plot experimental with CI if available
if has_bootstrap_exp and bootstrap_cols_exp['TPR'] is not None:
    ci_lower_col, ci_upper_col = bootstrap_cols_exp['TPR']
    axes[0].fill_between(df_exp_plot['tau'], 
                         df_exp_plot[ci_lower_col], 
                         df_exp_plot[ci_upper_col],
                         color=exp_color, alpha=0.2, label='Experimental (all) 95% CI')
axes[0].plot(df_exp_plot['tau'], df_exp_plot['TPR'], 
             marker=exp_marker, color=exp_color, label='Experimental (all)', linewidth=2, markersize=8)

if df_exp_filtered_plot is not None:
    if has_bootstrap_filtered and bootstrap_cols_filtered['TPR'] is not None:
        ci_lower_col, ci_upper_col = bootstrap_cols_filtered['TPR']
        axes[0].fill_between(df_exp_filtered_plot['tau'], 
                             df_exp_filtered_plot[ci_lower_col], 
                             df_exp_filtered_plot[ci_upper_col],
                             color=exp_filtered_color, alpha=0.2, label='Experimental (train/val only) 95% CI')
    axes[0].plot(df_exp_filtered_plot['tau'], df_exp_filtered_plot['TPR'], 
                 marker=exp_filtered_marker, color=exp_filtered_color, label='Experimental (train/val only)', linewidth=2, markersize=8)
axes[0].plot(df_theory_summary['tau'], df_theory_summary['TPR'], 
             marker=theory_marker, color=theory_color, label='Theoretical', linewidth=2, markersize=8)
if 'TPR_asymptote' in df_theory_summary.columns:
    axes[0].plot(df_theory_summary['tau'], df_theory_summary['TPR_asymptote'], 
                 marker=theory_marker, color=theory_color, linestyle='--', 
                 label='Theoretical (asymptote)', linewidth=2, markersize=8, alpha=0.7)
axes[0].set_xlabel('Threshold (tau)', fontsize=12)
axes[0].set_ylabel('TPR (True Positive Rate)', fontsize=12)
axes[0].set_title('TPR vs Threshold', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# Plot 2: FPR vs tau
# Plot experimental with CI if available
if has_bootstrap_exp and bootstrap_cols_exp['FPR'] is not None:
    ci_lower_col, ci_upper_col = bootstrap_cols_exp['FPR']
    axes[1].fill_between(df_exp_plot['tau'], 
                         df_exp_plot[ci_lower_col], 
                         df_exp_plot[ci_upper_col],
                         color=exp_color, alpha=0.2, label='Experimental (all) 95% CI')
axes[1].plot(df_exp_plot['tau'], df_exp_plot['FPR'], 
             marker=exp_marker, color=exp_color, label='Experimental (all)', linewidth=2, markersize=8)
if df_exp_filtered_plot is not None:
    if has_bootstrap_filtered and bootstrap_cols_filtered['FPR'] is not None:
        ci_lower_col, ci_upper_col = bootstrap_cols_filtered['FPR']
        axes[1].fill_between(df_exp_filtered_plot['tau'], 
                             df_exp_filtered_plot[ci_lower_col], 
                             df_exp_filtered_plot[ci_upper_col],
                             color=exp_filtered_color, alpha=0.2, label='Experimental (train/val only) 95% CI')
    axes[1].plot(df_exp_filtered_plot['tau'], df_exp_filtered_plot['FPR'], 
                 marker=exp_filtered_marker, color=exp_filtered_color, label='Experimental (train/val only, closer to FPR theory)', linewidth=2, markersize=8)
axes[1].plot(df_theory_summary['tau'], df_theory_summary['FPR'], 
             marker=theory_marker, color=theory_color, label='Theoretical', linewidth=2, markersize=8)
if 'FPR_asymptote' in df_theory_summary.columns:
    axes[1].plot(df_theory_summary['tau'], df_theory_summary['FPR_asymptote'], 
                 marker=theory_marker, color=theory_color, linestyle='--', 
                 label='Theoretical (asymptote)', linewidth=2, markersize=8, alpha=0.7)
axes[1].set_xlabel('Threshold (tau)', fontsize=12)
axes[1].set_ylabel('FPR (False Positive Rate)', fontsize=12)
axes[1].set_title('FPR vs Threshold', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# Plot 3: Total Recall vs tau
# Plot experimental with CI if available
if has_bootstrap_exp and bootstrap_cols_exp['total_recall'] is not None:
    ci_lower_col, ci_upper_col = bootstrap_cols_exp['total_recall']
    axes[2].fill_between(df_exp_plot['tau'], 
                         df_exp_plot[ci_lower_col], 
                         df_exp_plot[ci_upper_col],
                         color=exp_color, alpha=0.2, label='Experimental (all) 95% CI')
axes[2].plot(df_exp_plot['tau'], df_exp_plot['total_recall'], 
             marker=exp_marker, color=exp_color, label='Experimental (all)', linewidth=2, markersize=8)
if df_exp_filtered_plot is not None:
    if has_bootstrap_filtered and bootstrap_cols_filtered['total_recall'] is not None:
        ci_lower_col, ci_upper_col = bootstrap_cols_filtered['total_recall']
        axes[2].fill_between(df_exp_filtered_plot['tau'], 
                             df_exp_filtered_plot[ci_lower_col], 
                             df_exp_filtered_plot[ci_upper_col],
                             color=exp_filtered_color, alpha=0.2, label='Experimental (train/val only) 95% CI')
    axes[2].plot(df_exp_filtered_plot['tau'], df_exp_filtered_plot['total_recall'], 
                 marker=exp_filtered_marker, color=exp_filtered_color, label='Experimental (train/val only)', linewidth=2, markersize=8)
axes[2].plot(df_theory_summary['tau'], df_theory_summary['recall'], 
             marker=theory_marker, color=theory_color, label='Theoretical', linewidth=2, markersize=8)
if 'recall_asymptote' in df_theory_summary.columns:
    axes[2].plot(df_theory_summary['tau'], df_theory_summary['recall_asymptote'], 
                 marker=theory_marker, color=theory_color, linestyle='--', 
                 label='Theoretical (asymptote)', linewidth=2, markersize=8, alpha=0.7)
axes[2].set_xlabel('Threshold (tau)', fontsize=12)
axes[2].set_ylabel('Total Recall', fontsize=12)
axes[2].set_title('Total Recall vs Threshold', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
# axes[2].set_ylim(-0.05, 1.05)

fig.suptitle(f'Experimental vs Theoretical: {model} {dataset_size} pii_rate={pii_rate} n_epochs={n_epochs}', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
output_path = os.path.join(plots_dir, f"exp_vs_theory_tpr_fpr_recall_tau_{model}_{dataset_size}_{pii_rate}_{n_epochs}.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\nSaved comparison plot to: {output_path}")
plt.show()

# Print summary
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print("\nExperimental Results (all data):")
print(df_exp_plot[['tau', 'TPR', 'FPR', 'total_recall']].to_string(index=False))
if df_exp_filtered_plot is not None:
    print("\nExperimental Results (train/val only):")
    print(df_exp_filtered_plot[['tau', 'TPR', 'FPR', 'total_recall']].to_string(index=False))
print("\nTheoretical Results:")
print(df_theory_summary[['tau', 'TPR', 'FPR', 'recall']].to_string(index=False))
