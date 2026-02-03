# Plot ll distribution histogram for train/val/other (remaining) names
# Visualize if generated names not in original list differ from validation

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base paths
from src._repo import REPO_ROOT
BASE_DIR = REPO_ROOT
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/pii_leakage/experimental-recall')

# Input file
COMPUTED_LL_FILE = os.path.join(OUTPUT_DIR, 'all_names_ll_computed.csv')

# Output plot
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, 'll_distribution_by_groundtruth.png')


def plot_distribution():
    print("="*60)
    print("Plotting LL Distribution by Groundtruth")
    print("="*60)
    
    # Load computed ll
    print(f"\nLoading: {COMPUTED_LL_FILE}")
    df = pd.read_csv(COMPUTED_LL_FILE)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check groundtruth distribution
    print(f"\nGroundtruth distribution:")
    print(df['groundtruth'].value_counts())
    
    # Find ll columns
    ll_cols = [c for c in df.columns if c.startswith('ll_')]
    print(f"\nLL columns: {ll_cols}")
    
    # We'll plot for finetuned Name: (main feature for classification)
    ft_name_col = None
    base_name_col = None
    for col in ll_cols:
        if 'finetuned' in col.lower() and 'name' in col.lower():
            ft_name_col = col
        if 'base' in col.lower() and 'name' in col.lower():
            base_name_col = col
    
    print(f"\nFinetuned Name column: {ft_name_col}")
    print(f"Base Name column: {base_name_col}")
    
    # Colors for each category
    colors = {
        'train': 'red',
        'val': 'blue', 
        'other': 'green'
    }
    
    labels = {
        'train': 'Train (in training set)',
        'val': 'Val (in validation set)',
        'other': 'Other (generated, not in original)'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Finetuned Name: ll (normalized)
    ax1 = axes[0, 0]
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt][ft_name_col].dropna()
        if len(data) > 0:
            ax1.hist(data, bins=50, alpha=0.5, color=colors[gt], label=f"{labels[gt]} (n={len(data)})", density=True)
    ax1.set_xlabel('Log-Likelihood')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Finetuned Model - Name: prompt\n({ft_name_col})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Base Name: ll (normalized)
    ax2 = axes[0, 1]
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt][base_name_col].dropna()
        if len(data) > 0:
            ax2.hist(data, bins=50, alpha=0.5, color=colors[gt], label=f"{labels[gt]} (n={len(data)})", density=True)
    ax2.set_xlabel('Log-Likelihood')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Base Model - Name: prompt\n({base_name_col})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference (finetuned - base) for Name: (normalized)
    ax3 = axes[1, 0]
    df['ll_diff_name'] = df[ft_name_col] - df[base_name_col]
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt]['ll_diff_name'].dropna()
        if len(data) > 0:
            ax3.hist(data, bins=50, alpha=0.5, color=colors[gt], label=f"{labels[gt]} (n={len(data)})", density=True)
    ax3.set_xlabel('LL Difference (Finetuned - Base)')
    ax3.set_ylabel('Density')
    ax3.set_title('LL Difference (Finetuned - Base) - Name: prompt')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: KDE overlay for better comparison
    ax4 = axes[1, 1]
    from scipy import stats
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt][ft_name_col].dropna()
        if len(data) > 10:
            # KDE
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax4.plot(x_range, kde(x_range), color=colors[gt], label=f"{labels[gt]} (n={len(data)})", linewidth=2)
    ax4.set_xlabel('Log-Likelihood')
    ax4.set_ylabel('Density')
    ax4.set_title('KDE - Finetuned Model - Name: prompt')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {OUTPUT_PLOT}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics by Groundtruth")
    print("="*60)
    
    print(f"\n{ft_name_col}:")
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt][ft_name_col].dropna()
        if len(data) > 0:
            print(f"  {gt:8s}: mean={data.mean():.4f}, std={data.std():.4f}, median={data.median():.4f}, n={len(data)}")
    
    print(f"\n{base_name_col}:")
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt][base_name_col].dropna()
        if len(data) > 0:
            print(f"  {gt:8s}: mean={data.mean():.4f}, std={data.std():.4f}, median={data.median():.4f}, n={len(data)}")
    
    print(f"\nLL Difference (Finetuned - Base):")
    for gt in ['train', 'val', 'other']:
        data = df[df['groundtruth'] == gt]['ll_diff_name'].dropna()
        if len(data) > 0:
            print(f"  {gt:8s}: mean={data.mean():.4f}, std={data.std():.4f}, median={data.median():.4f}, n={len(data)}")
    
    # Statistical tests
    print("\n" + "="*60)
    print("Statistical Tests (Kolmogorov-Smirnov)")
    print("="*60)
    
    from scipy.stats import ks_2samp
    
    train_data = df[df['groundtruth'] == 'train'][ft_name_col].dropna()
    val_data = df[df['groundtruth'] == 'val'][ft_name_col].dropna()
    other_data = df[df['groundtruth'] == 'other'][ft_name_col].dropna()
    
    if len(train_data) > 0 and len(val_data) > 0:
        stat, pval = ks_2samp(train_data, val_data)
        print(f"\nTrain vs Val: KS stat={stat:.4f}, p-value={pval:.4e}")
    
    if len(train_data) > 0 and len(other_data) > 0:
        stat, pval = ks_2samp(train_data, other_data)
        print(f"Train vs Other: KS stat={stat:.4f}, p-value={pval:.4e}")
    
    if len(val_data) > 0 and len(other_data) > 0:
        stat, pval = ks_2samp(val_data, other_data)
        print(f"Val vs Other: KS stat={stat:.4f}, p-value={pval:.4e}")
    
    return df


if __name__ == "__main__":
    df = plot_distribution()
