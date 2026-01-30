# Analyze classifier performance stratified by Log-Likelihood percentiles
# Understand why names are misclassified based on their LL values

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from config_loader import load_config

# Parse arguments first
parser = argparse.ArgumentParser(description='Analyze classifier by LL percentiles')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--df_temp_sub', type=str, default=None, help='Path to df_temp_sub CSV (overrides config)')
parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
parser.add_argument('--n_bins', type=int, default=5, help='Number of percentile bins')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Get paths from config
OUTPUT_DIR = config['output_dir']
DF_TEMP_SUB_FILE = args.df_temp_sub or config['classifier']['df_temp_sub_file']


def compute_metrics(y_true, y_pred):
    """Compute FPR, TPR, and other metrics from predictions."""
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Handle edge cases where only one class is present
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 1:
                TP = (y_pred == 1).sum()
                FN = (y_pred == 0).sum()
                TN, FP = 0, 0
            else:
                TN = (y_pred == 0).sum()
                FP = (y_pred == 1).sum()
                TP, FN = 0, 0
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    TNR = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    FNR = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else np.nan
    
    return {
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'FPR': FPR, 'TPR': TPR, 'TNR': TNR, 'FNR': FNR,
        'PPV': PPV, 'accuracy': accuracy
    }


def analyze_by_ll_percentiles(df_temp_sub_path, threshold=0.5, n_bins=5):
    """
    Analyze classifier performance stratified by LL percentiles.
    
    Args:
        df_temp_sub_path: Path to df_temp_sub CSV
        threshold: Classification threshold (default 0.5)
        n_bins: Number of percentile bins (default 5 for quintiles: 0-20, 20-40, etc.)
    """
    print("="*80)
    print("Classifier Performance Analysis by Log-Likelihood Percentiles")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {df_temp_sub_path}")
    df = pd.read_csv(df_temp_sub_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Find LL columns (p_pre_*, p_ft_*)
    ll_cols = [c for c in df.columns if c.startswith('p_pre_') or c.startswith('p_ft_')]
    print(f"\nLL columns found: {ll_cols}")
    
    # Use the finetuned Name: column for stratification (main feature)
    ll_col = None
    for col in ll_cols:
        if 'ft' in col and 'Name' in col:
            ll_col = col
            break
    
    if ll_col is None:
        ll_col = ll_cols[0] if ll_cols else None
    
    print(f"Using column for stratification: {ll_col}")
    
    # Debug: show actual values range and detect if it's probability or log-probability
    is_probability = False
    if ll_col and ll_col in df.columns:
        ll_data = df[ll_col].dropna()
        print(f"\nColumn statistics ({ll_col}):")
        print(f"  Count: {len(ll_data)}")
        print(f"  Min: {ll_data.min():.2e}")
        print(f"  Max: {ll_data.max():.2e}")
        print(f"  Mean: {ll_data.mean():.2e}")
        print(f"  Sample values: {ll_data.head(5).tolist()}")
        
        # Detect if values are probabilities (0 < x <= 1) or log-probabilities (x <= 0)
        if ll_data.min() >= 0 and ll_data.max() <= 1:
            is_probability = True
            print(f"\n  -> Detected as PROBABILITY values (0 < x <= 1)")
            print(f"     Will convert to log-probability for analysis")
            # Convert to log-probability
            df[ll_col + '_log'] = np.log(df[ll_col].clip(lower=1e-300))
            ll_col_orig = ll_col
            ll_col = ll_col + '_log'
            # Show converted values
            log_data = df[ll_col].dropna()
            print(f"\n  After conversion to log-prob ({ll_col}):")
            print(f"    Min: {log_data.min():.2f}")
            print(f"    Max: {log_data.max():.2f}")
            print(f"    Mean: {log_data.mean():.2f}")
        else:
            print(f"\n  -> Detected as LOG-PROBABILITY values (x <= 0)")
    
    # Split into training set (y_pred_proba is NA) and test set
    # Note: do this after potential column conversion above
    df_train = df[df['y_pred_proba'].isna()].copy()
    df_test = df[df['y_pred_proba'].notna()].copy()
    
    print(f"\n{'='*80}")
    print("DATA SPLIT")
    print(f"{'='*80}")
    print(f"Training set (y_pred_proba is NA): {len(df_train)}")
    print(f"Test set (y_pred_proba is not NA): {len(df_test)}")
    
    # Ground truth: group column (1 = train, 0 = val)
    print(f"\nTraining set - Ground truth distribution:")
    print(df_train['group'].value_counts())
    print(f"\nTest set - Ground truth distribution:")
    print(df_test['group'].value_counts())
    
    # ============================================================
    # TRAINING SET ANALYSIS (no predictions available)
    # ============================================================
    print(f"\n{'='*80}")
    print("TRAINING SET ANALYSIS (used to train the classifier)")
    print(f"{'='*80}")
    
    if ll_col and ll_col in df_train.columns:
        print(f"\nLog-prob ({ll_col}) statistics by ground truth:")
        for group in [0, 1]:
            data = df_train[df_train['group'] == group][ll_col].dropna()
            group_name = "Train (1)" if group == 1 else "Val (0)"
            if len(data) > 0:
                print(f"  {group_name}: n={len(data)}")
                print(f"    Log-prob: mean={data.mean():.2f}, std={data.std():.2f}, min={data.min():.2f}, max={data.max():.2f}")
                print(f"    Proba:    mean={np.exp(data.mean()):.2e}, min={np.exp(data.min()):.2e}, max={np.exp(data.max()):.2e}")
    
    # ============================================================
    # TEST SET ANALYSIS (stratified by LL percentiles)
    # ============================================================
    print(f"\n{'='*80}")
    print("TEST SET ANALYSIS (stratified by LL percentiles)")
    print(f"{'='*80}")
    
    if len(df_test) == 0:
        print("No test samples available!")
        return
    
    # Create predictions at threshold
    df_test['y_pred'] = (df_test['y_pred_proba'] >= threshold).astype(int)
    df_test['y_true'] = df_test['group']
    
    # Overall test set metrics
    print(f"\n--- Overall Test Set (threshold={threshold}) ---")
    metrics = compute_metrics(df_test['y_true'].values, df_test['y_pred'].values)
    
    # Compute AUC
    try:
        auc = roc_auc_score(df_test['y_true'].values, df_test['y_pred_proba'].values)
    except ValueError:
        auc = np.nan
    
    print(f"  n={len(df_test)}")
    print(f"  Confusion: TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")
    print(f"  FPR={metrics['FPR']:.4f}, TPR={metrics['TPR']:.4f}, Accuracy={metrics['accuracy']:.4f}, AUC={auc:.4f}")
    
    if ll_col is None or ll_col not in df_test.columns:
        print(f"\nLL column not found, cannot stratify by percentiles")
        return
    
    # Compute percentile bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    ll_values = df_test[ll_col].dropna()
    bin_edges = np.percentile(ll_values, percentiles)
    
    print(f"\n{'='*80}")
    print(f"STRATIFICATION BY LL PERCENTILES ({ll_col})")
    print(f"{'='*80}")
    print(f"\nPercentile bin edges (LL values):")
    for i in range(len(bin_edges)):
        print(f"  {percentiles[i]:5.1f}%: LL={bin_edges[i]:.2f} (proba={np.exp(bin_edges[i]):.2e})")
    
    # Create percentile labels
    df_test['ll_percentile'] = pd.cut(
        df_test[ll_col], 
        bins=bin_edges, 
        labels=[f"{int(percentiles[i])}-{int(percentiles[i+1])}%" for i in range(n_bins)],
        include_lowest=True
    )
    
    # Analyze each percentile bin
    print(f"\n{'='*80}")
    print("METRICS BY LL PERCENTILE BIN")
    print(f"{'='*80}")
    
    results = []
    for i in range(n_bins):
        bin_label = f"{int(percentiles[i])}-{int(percentiles[i+1])}%"
        ll_min = bin_edges[i]
        ll_max = bin_edges[i+1]
        
        df_bin = df_test[df_test['ll_percentile'] == bin_label]
        
        if len(df_bin) == 0:
            continue
        
        n_train = (df_bin['y_true'] == 1).sum()
        n_val = (df_bin['y_true'] == 0).sum()
        
        if len(df_bin) > 0:
            metrics = compute_metrics(df_bin['y_true'].values, df_bin['y_pred'].values)
            # Compute AUC for this bin (need both classes present)
            try:
                if len(df_bin['y_true'].unique()) > 1:
                    bin_auc = roc_auc_score(df_bin['y_true'].values, df_bin['y_pred_proba'].values)
                else:
                    bin_auc = np.nan
            except ValueError:
                bin_auc = np.nan
        else:
            metrics = {'FPR': np.nan, 'TPR': np.nan, 'accuracy': np.nan, 
                      'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}
            bin_auc = np.nan
        
        results.append({
            'bin': bin_label,
            'll_min': ll_min,
            'll_max': ll_max,
            'n_total': len(df_bin),
            'n_train': n_train,
            'n_val': n_val,
            'auc': bin_auc,
            **metrics
        })
        
        print(f"\n--- {bin_label} ---")
        print(f"  LL range: {ll_min:.2f} to {ll_max:.2f}")
        print(f"  Proba range: {np.exp(ll_min):.2e} to {np.exp(ll_max):.2e}")
        print(f"  n={len(df_bin)} (train={n_train}, val={n_val})")
        print(f"  Confusion: TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")
        print(f"  FPR={metrics['FPR']:.4f} (val predicted as train)")
        print(f"  TPR={metrics['TPR']:.4f} (train correctly identified)")
        print(f"  Accuracy={metrics['accuracy']:.4f}, AUC={bin_auc:.4f}" if not np.isnan(bin_auc) else f"  Accuracy={metrics['accuracy']:.4f}, AUC=N/A (single class)")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Percentile':<12} {'LL Range':<20} {'Proba Range':<25} {'n':>6} {'Train':>6} {'Val':>6} {'FPR':>8} {'TPR':>8} {'Acc':>8} {'AUC':>8}")
    print("-" * 130)
    for r in results:
        proba_min = np.exp(r['ll_min'])
        proba_max = np.exp(r['ll_max'])
        auc_str = f"{r['auc']:>8.4f}" if not np.isnan(r['auc']) else "     N/A"
        print(f"{r['bin']:<12} {r['ll_min']:.1f} to {r['ll_max']:.1f}      {proba_min:.2e} to {proba_max:.2e}   {r['n_total']:>6} {r['n_train']:>6} {r['n_val']:>6} "
              f"{r['FPR']:>8.4f} {r['TPR']:>8.4f} {r['accuracy']:>8.4f} {auc_str}")
    
    # Show misclassified examples
    print(f"\n{'='*80}")
    print("MISCLASSIFIED EXAMPLES")
    print(f"{'='*80}")
    
    # False Positives (val predicted as train)
    fp_mask = (df_test['y_true'] == 0) & (df_test['y_pred'] == 1)
    df_fp = df_test[fp_mask].sort_values('y_pred_proba', ascending=False)
    
    print(f"\nFalse Positives (Val predicted as Train): {len(df_fp)}")
    if len(df_fp) > 0:
        print("\nTop 10 False Positives (highest confidence):")
        cols_to_show = ['value', ll_col, 'y_pred_proba', 'll_percentile']
        cols_available = [c for c in cols_to_show if c in df_fp.columns]
        print(df_fp[cols_available].head(10).to_string())
    
    # False Negatives (train predicted as val)
    fn_mask = (df_test['y_true'] == 1) & (df_test['y_pred'] == 0)
    df_fn = df_test[fn_mask].sort_values('y_pred_proba', ascending=True)
    
    print(f"\nFalse Negatives (Train predicted as Val): {len(df_fn)}")
    if len(df_fn) > 0:
        print("\nTop 10 False Negatives (lowest confidence):")
        cols_to_show = ['value', ll_col, 'y_pred_proba', 'll_percentile']
        cols_available = [c for c in cols_to_show if c in df_fn.columns]
        print(df_fn[cols_available].head(10).to_string())
    
    # Save results
    df_results = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'clf_analysis_by_ll_percentile.csv')
    df_results.to_csv(output_file, index=False)
    print(f"\n\nSaved results to: {output_file}")
    
    return df_results


if __name__ == "__main__":
    print(f"Config: {args.config or 'default'}")
    print(f"df_temp_sub: {DF_TEMP_SUB_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Threshold: {args.threshold}")
    print(f"N bins: {args.n_bins}")
    
    df_results = analyze_by_ll_percentiles(
        DF_TEMP_SUB_FILE,
        threshold=args.threshold,
        n_bins=args.n_bins
    )
