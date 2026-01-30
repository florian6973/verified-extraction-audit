"""
Bootstrap resampling for experimental metrics to compute confidence intervals.

This module provides functions to bootstrap the extracted names dataset
and compute confidence intervals for MIA metrics (TP, FP, recall, FPR, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from src.evaluation.pipeline.experimental.mia.name_filter import name_mask

def bootstrap_metrics(
    df: pd.DataFrame,
    score_col: str = "score_oof_member_proba",
    groundtruth_col: str = "groundtruth",
    thresholds: Optional[List[float]] = None,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    filter_groundtruth: bool = False,
    stratify_by: Optional[str] = None,
    total_train_names: Optional[int] = None,
    filter_language: bool = True
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Bootstrap resampling to compute confidence intervals for metrics.
    
    Args:
        df: DataFrame with scores and ground truth labels
        score_col: Column name for scores
        groundtruth_col: Column name for ground truth labels
        thresholds: List of thresholds to evaluate. If None, uses optimal threshold.
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        filter_groundtruth: If True, filter to only 'train' and 'val'
        stratify_by: Column name to stratify bootstrap by (e.g., 'groundtruth' or 'split_x')
                    This ensures each bootstrap sample has similar class distribution
        total_train_names: Total number of train names (for total_recall calculation)
        filter_language: If True, filter to only likely names using language heuristics (default: True)
    
    Returns:
        Dictionary mapping metric names to dicts with:
        - 'mean': Mean value across bootstrap samples
        - 'ci_lower': Lower bound of 95% confidence interval
        - 'ci_upper': Upper bound of 95% confidence interval
        - 'std': Standard deviation across bootstrap samples
    """
    np.random.seed(random_state)

    import warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="DataFrameGroupBy.apply operated on the grouping columns"
    )
    
    # Start with input dataframe
    df_work = df.copy()
    
    # Apply language filter if requested
    if filter_language:
        if 'value' in df_work.columns:
            # Diagnostic: Check distribution before filtering
            rows_before_lang = len(df_work)
            if rows_before_lang > 0 and score_col in df_work.columns:
                # Check distribution by groundtruth
                before_dist = df_work[groundtruth_col].value_counts()
                print(f"\nBefore language filter - Distribution by groundtruth:")
                print(before_dist)
                
                # Calculate FPR at common thresholds BEFORE filtering
                df_other_before = df_work[df_work[groundtruth_col].isin(['val', 'other'])].copy()
                if len(df_other_before) > 0:
                    print(f"\nFPR at different thresholds BEFORE language filter (on val+other entries):")
                    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        fp_before = (df_other_before[score_col] >= thr).sum()
                        tn_before = (df_other_before[score_col] < thr).sum()
                        fpr_before = fp_before / (fp_before + tn_before) if (fp_before + tn_before) > 0 else 0
                        print(f"  Threshold {thr:.1f}: FPR = {fpr_before:.4f} (FP={fp_before}, TN={tn_before})")
            
            name_mask_result = name_mask(df_work, column='value')
            df_work = df_work[name_mask_result].copy()
            rows_after_lang = len(df_work)
            
            print(f"\nLanguage filter applied: {rows_before_lang} -> {rows_after_lang} rows ({rows_before_lang - rows_after_lang} filtered out)")
            
            # Diagnostic: Check distribution after filtering
            if rows_after_lang > 0 and score_col in df_work.columns:
                after_dist = df_work[groundtruth_col].value_counts()
                print(f"\nAfter language filter - Distribution by groundtruth:")
                print(after_dist)
                
                # Calculate FPR at common thresholds AFTER filtering
                df_other_after = df_work[df_work[groundtruth_col].isin(['val', 'other'])].copy()
                if len(df_other_after) > 0:
                    print(f"\nFPR at different thresholds AFTER language filter (on val+other entries):")
                    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        fp_after = (df_other_after[score_col] >= thr).sum()
                        tn_after = (df_other_after[score_col] < thr).sum()
                        fpr_after = fp_after / (fp_after + tn_after) if (fp_after + tn_after) > 0 else 0
                        print(f"  Threshold {thr:.1f}: FPR = {fpr_after:.4f} (FP={fp_after}, TN={tn_after})")
                    
                    # Show the change
                    print(f"\nFPR change due to language filter:")
                    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        fp_before = (df_other_before[score_col] >= thr).sum()
                        tn_before = (df_other_before[score_col] < thr).sum()
                        fpr_before = fp_before / (fp_before + tn_before) if (fp_before + tn_before) > 0 else 0
                        fp_after = (df_other_after[score_col] >= thr).sum()
                        tn_after = (df_other_after[score_col] < thr).sum()
                        fpr_after = fp_after / (fp_after + tn_after) if (fp_after + tn_after) > 0 else 0
                        change = fpr_after - fpr_before
                        print(f"  Threshold {thr:.1f}: {fpr_before:.4f} -> {fpr_after:.4f} (change: {change:+.4f})")
        else:
            print(f"Warning: 'value' column not found, skipping language filter")
    
    # Filter if requested
    if filter_groundtruth:
        df_work = df_work[df_work[groundtruth_col].isin(['train', 'val'])].copy()
    else:
        pass  # Already have df_work
    
    # Remove rows with missing scores
    df_valid = df_work.dropna(subset=[score_col]).copy()
    df_valid['y_true'] = (df_valid[groundtruth_col] == 'train').astype(int)
    
    if len(df_valid) == 0:
        raise ValueError("No valid rows with scores found")
    
    # Compute original total_train_names (fixed value, not from bootstrap sample)
    # This is needed for correct total_recall calculation
    # Use provided value if available, otherwise compute from data
    if total_train_names is not None:
        original_total_train_names = total_train_names
    else:
        original_total_train_names = (df_valid['y_true'] == 1).sum()
    
    # If no thresholds provided, compute optimal threshold from full data
    if thresholds is None:
        from sklearn.metrics import roc_curve
        fpr, tpr, thrs = roc_curve(df_valid['y_true'], df_valid[score_col])
        # Use Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        thresholds = [thrs[optimal_idx]]
    
    # Store all bootstrap results
    bootstrap_results = {thr: [] for thr in thresholds}
    
    print(f"\nRunning {n_bootstrap} bootstrap samples...")
    print(f"Original total train names: {original_total_train_names}")
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Resample with replacement
        if stratify_by is not None and stratify_by in df_valid.columns:
            # Stratified bootstrap: maintain class distribution
            bootstrap_df = df_valid.groupby(stratify_by).apply(
                lambda x: x.sample(n=len(x), replace=True, random_state=random_state + b if random_state else None)
            ).reset_index(drop=True)
        else:
            # Simple bootstrap
            bootstrap_df = df_valid.sample(n=len(df_valid), replace=True, random_state=random_state + b if random_state else None)
        
        # Compute metrics for each threshold
        for thr in thresholds:
            y_pred = (bootstrap_df[score_col] >= thr).astype(int)
            y_true = bootstrap_df['y_true'].values
            
            # Compute confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
            else:
                # Handle edge cases
                if len(np.unique(y_pred)) == 1:
                    if y_pred[0] == 1:
                        TP = (y_true == 1).sum()
                        FP = (y_true == 0).sum()
                        TN, FN = 0, 0
                    else:
                        TN = (y_true == 0).sum()
                        FN = (y_true == 1).sum()
                        TP, FP = 0, 0
                else:
                    TN, FP, FN, TP = 0, 0, 0, 0
            
            # Compute metrics
            total_train = (y_true == 1).sum()
            total_val = (y_true == 0).sum()
            
            tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0
            # FPR = FP / (FP + TN)
            # FP = False Positives: predicted as train (score >= threshold) but actually val/other
            # TN = True Negatives: predicted as val/other (score < threshold) and actually val/other
            # Note: Language filtering removes non-name entries. If it removes FP and TN proportionally,
            # FPR may not change much. FPR will decrease if more high-scoring non-names (potential FPs)
            # are filtered out compared to low-scoring non-names (potential TNs).
            fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = tpr_val
            # total_recall should use the ORIGINAL total_train_names, not the bootstrap sample's count
            # This is because total_recall = TP / total_train_names (fixed denominator)
            total_recall = TP / original_total_train_names if original_total_train_names > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            
            bootstrap_results[thr].append({
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'TN': TN,
                'TPR': tpr_val,
                'FPR': fpr_val,
                'precision': precision,
                'recall': recall,
                'total_recall': total_recall,
                'accuracy': accuracy,
            })
    
    # Compute statistics for each metric
    results_summary = {}
    for thr in thresholds:
        results_summary[thr] = {}
        metrics_list = bootstrap_results[thr]
        
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            
            results_summary[thr][metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
    
    return results_summary


def add_bootstrap_to_metrics_csv(
    metrics_csv_path: str,
    df_scores: pd.DataFrame,
    score_col: str = "score_oof_member_proba",
    groundtruth_col: str = "groundtruth",
    n_bootstrap: int = 1000,
    output_suffix: Optional[str] = None,
    filter_groundtruth: bool = False,
    filter_language: bool = True
) -> str:
    """
    Add bootstrap confidence intervals to existing metrics CSV.
    
    Args:
        metrics_csv_path: Path to existing metrics CSV
        df_scores: DataFrame with scores (for bootstrapping)
        score_col: Column name for scores
        groundtruth_col: Column name for ground truth
        n_bootstrap: Number of bootstrap samples
        output_suffix: Suffix for output file (if None, auto-generates based on filter_groundtruth)
        filter_groundtruth: Whether to filter to train/val only
    
    Returns:
        Path to output CSV with bootstrap CIs
    """
    # Check if input file already has "_without_others" in the name (needed for output path generation)
    base_name = os.path.basename(metrics_csv_path)
    has_without_others_in_name = "_without_others" in base_name
    
    # Auto-generate suffix based on filter_groundtruth if not provided
    # The suffix should reflect whether filtering was applied during bootstrap
    if output_suffix is None:
        if filter_groundtruth:
            # If filtering was applied during bootstrap, always use "_with_bootstrap_without_others"
            # This ensures consistent naming regardless of input file name
            output_suffix = "_with_bootstrap_without_others"
        else:
            # No filtering during bootstrap, just add "_with_bootstrap"
            output_suffix = "_with_bootstrap"
    # Load existing metrics
    df_metrics = pd.read_csv(metrics_csv_path)
    
    # IMPORTANT: Make a copy of original metric values to ensure they're preserved
    # The bootstrap should only add CI columns, not modify original metrics
    original_metrics = df_metrics[['threshold', 'TP', 'FP', 'FN', 'TN', 'TPR', 'FPR', 'precision', 'recall', 'total_recall', 'accuracy']].copy()
    
    # Get thresholds from metrics
    thresholds = df_metrics['threshold'].unique().tolist()
    
    # Get total_train_names from metrics CSV if available (should be consistent across all rows)
    total_train_names = None
    if 'total_train_names' in df_metrics.columns:
        total_train_names = df_metrics['total_train_names'].iloc[0]  # Should be same for all rows
        print(f"Using total_train_names from metrics CSV: {total_train_names}")
    
    # Run bootstrap
    print(f"\nComputing bootstrap confidence intervals for {len(thresholds)} thresholds...")
    bootstrap_results = bootstrap_metrics(
        df_scores,
        score_col=score_col,
        groundtruth_col=groundtruth_col,
        thresholds=thresholds,
        n_bootstrap=n_bootstrap,
        filter_groundtruth=filter_groundtruth,
        stratify_by='groundtruth',  # Maintain class distribution
        total_train_names=total_train_names,
        filter_language=filter_language
    )
    
    # Add CI columns to metrics dataframe (ONLY CI columns, preserve original metrics)
    for metric in ['TP', 'FP', 'TPR', 'FPR', 'precision', 'recall', 'total_recall', 'accuracy']:
        ci_lower_col = f'{metric}_ci_lower'
        ci_upper_col = f'{metric}_ci_upper'
        std_col = f'{metric}_std'
        
        df_metrics[ci_lower_col] = np.nan
        df_metrics[ci_upper_col] = np.nan
        df_metrics[std_col] = np.nan
        
        for idx, row in df_metrics.iterrows():
            thr = row['threshold']
            if thr in bootstrap_results and metric in bootstrap_results[thr]:
                df_metrics.at[idx, ci_lower_col] = bootstrap_results[thr][metric]['ci_lower']
                df_metrics.at[idx, ci_upper_col] = bootstrap_results[thr][metric]['ci_upper']
                df_metrics.at[idx, std_col] = bootstrap_results[thr][metric]['std']
    
    # Ensure original metric values are preserved (safety check)
    for idx, row in df_metrics.iterrows():
        thr = row['threshold']
        orig_row = original_metrics[original_metrics['threshold'] == thr]
        if len(orig_row) > 0:
            orig_row = orig_row.iloc[0]
            # Restore original values to ensure they match the input metrics CSV
            df_metrics.at[idx, 'TP'] = orig_row['TP']
            df_metrics.at[idx, 'FP'] = orig_row['FP']
            df_metrics.at[idx, 'FN'] = orig_row['FN']
            df_metrics.at[idx, 'TN'] = orig_row['TN']
            df_metrics.at[idx, 'TPR'] = orig_row['TPR']
            df_metrics.at[idx, 'FPR'] = orig_row['FPR']
            df_metrics.at[idx, 'precision'] = orig_row['precision']
            df_metrics.at[idx, 'recall'] = orig_row['recall']
            df_metrics.at[idx, 'total_recall'] = orig_row['total_recall']
            df_metrics.at[idx, 'accuracy'] = orig_row['accuracy']
    
    # Save output
    # If input has "_without_others" and we're adding "_with_bootstrap_without_others",
    # we need to replace "_without_others" with the new suffix to avoid duplication
    if has_without_others_in_name and filter_groundtruth and output_suffix == "_with_bootstrap_without_others":
        # Replace "_without_others.csv" with "_with_bootstrap_without_others.csv"
        output_path = metrics_csv_path.replace('_without_others.csv', f'{output_suffix}.csv')
    else:
        output_path = metrics_csv_path.replace('.csv', f'{output_suffix}.csv')
    df_metrics.to_csv(output_path, index=False)
    print(f"\nSaved metrics with bootstrap CIs to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Bootstrap metrics for MIA evaluation')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (alternative to --scores-csv and --metrics-csv)')
    parser.add_argument('--scores-csv', type=str, default=None,
                        help='Path to scores CSV (all_names_ll_computed_with_scores.csv)')
    parser.add_argument('--metrics-csv', type=str, default=None,
                        help='Path to metrics CSV to add bootstrap CIs to')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Number of bootstrap samples (default: 100)')
    parser.add_argument('--filter-groundtruth', action='store_true',
                        help='Filter to train/val only')
    parser.add_argument('--no-filter-language', action='store_true',
                        help='Disable language filtering (filter_language defaults to True)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # If config is provided, use it to determine file paths
    if args.config:
        from src.evaluation.pipeline.experimental.config_loader import load_config
        from src.evaluation.pipeline.experimental.config_helper import get_output_dir
        
        config = load_config(args.config)
        model = config['filters']['model']
        dataset_size = config['filters']['dataset_size']
        pii_rate = config['filters']['pii_rate']
        n_epochs = config['filters']['n_epochs']
        
        output_dir = get_output_dir(config)
        scores_csv = os.path.join(output_dir, "all_names_ll_computed_with_scores.csv")
        
        # Determine which metrics CSV to use based on filter_groundtruth
        if args.filter_groundtruth:
            metrics_csv = os.path.join(output_dir, "all_names_ll_computed_with_scores_metrics_by_threshold_without_others.csv")
        else:
            metrics_csv = os.path.join(output_dir, "all_names_ll_computed_with_scores_metrics_by_threshold.csv")
    else:
        # Use provided paths
        if args.scores_csv is None or args.metrics_csv is None:
            parser.error("Either --config must be provided, or both --scores-csv and --metrics-csv must be provided")
        scores_csv = args.scores_csv
        metrics_csv = args.metrics_csv
    
    # Check if files exist
    if not os.path.exists(scores_csv):
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    
    # Load scores
    print(f"Loading scores from: {scores_csv}")
    df_scores = pd.read_csv(scores_csv)
    
    # Add bootstrap CIs
    print(f"Adding bootstrap CIs to metrics from: {metrics_csv}")
    output_path = add_bootstrap_to_metrics_csv(
        metrics_csv,
        df_scores,
        n_bootstrap=args.n_bootstrap,
        filter_groundtruth=args.filter_groundtruth,
        filter_language=not args.no_filter_language,  # Default True, disable with --no-filter-language
    )
    
    print(f"\nDone! Bootstrap results saved to: {output_path}")
