#!/usr/bin/env python3
"""
Compute the best threshold for FPR < 5% using cross-fitting correctly.

For each scores CSV file in experimental-recall-output directories:
1. Load the scores file (should have fold_id, y_true, score_oof_member_proba columns)
2. For each CV fold, compute the threshold that gives FPR < 5% using only samples from that fold
3. Average the thresholds across all folds
4. Report results

This ensures proper cross-fitting: each sample is only evaluated using the model
trained on other folds (out-of-fold scores).
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, confusion_matrix


def compute_fpr_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    """Compute FPR at a given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        # Handle edge cases
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:
                TN, FP, FN, TP = 0, (y_true == 0).sum(), 0, (y_true == 1).sum()
            else:
                TN, FP, FN, TP = (y_true == 0).sum(), 0, (y_true == 1).sum(), 0
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
    
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return fpr


def find_threshold_for_fpr(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    target_fpr: float = 0.05,
    max_threshold: float = 1.0,
    min_threshold: float = 0.0,
    n_points: int = 1000
) -> Tuple[float, float]:
    """
    Find the threshold that gives FPR <= target_fpr (or closest to it).
    
    Returns:
        (threshold, actual_fpr) tuple
    """
    # Generate candidate thresholds
    thresholds = np.linspace(min_threshold, max_threshold, n_points)
    
    # Compute FPR for each threshold
    fprs = []
    for thr in thresholds:
        fpr = compute_fpr_at_threshold(y_true, y_scores, thr)
        fprs.append(fpr)
    # print(fprs)
    # input()
    
    fprs = np.array(fprs)
    
    # Find thresholds that achieve FPR <= target_fpr
    valid_mask = fprs <= target_fpr
    valid_indices = np.where(valid_mask)[0]
    # print(valid_indices)
    
    if len(valid_indices) > 0:
        # Among valid thresholds, choose the one with highest threshold (most permissive)
        # This maximizes TPR while keeping FPR <= target
        best_idx = valid_indices[np.argmin(thresholds[valid_indices])]
        best_threshold = thresholds[best_idx]
        actual_fpr = fprs[best_idx]
    else:
        # If no threshold achieves FPR <= target, find the one closest to target
        best_idx = np.argmin(np.abs(fprs - target_fpr))
        best_threshold = thresholds[best_idx]
        actual_fpr = fprs[best_idx]
    
    return best_threshold, actual_fpr


def compute_threshold_per_fold(
    df: pd.DataFrame,
    fold_id_col: str = "fold_id",
    y_true_col: str = "y_true",
    score_col: str = "score_oof_member_proba",
    target_fpr: float = 0.05
) -> Dict[int, Dict[str, float]]:
    """
    Compute threshold for FPR < target_fpr for each fold separately.
    
    Returns:
        Dictionary mapping fold_id to {'threshold': float, 'fpr': float, 'n_samples': int}
    """
    results = {}
    
    # Get unique fold IDs
    if fold_id_col not in df.columns:
        raise ValueError(f"Column '{fold_id_col}' not found. Available columns: {df.columns.tolist()}")
    
    fold_ids = sorted(df[fold_id_col].dropna().unique())
    
    if len(fold_ids) == 0:
        raise ValueError(f"No valid fold IDs found in column '{fold_id_col}'")
    
    print(f"  Found {len(fold_ids)} folds: {fold_ids}")
    
    for fold_id in fold_ids:
        # Get samples from this fold only (cross-fitting: these are test samples for this fold)
        df_fold = df[df[fold_id_col] == fold_id].copy()
        
        if len(df_fold) == 0:
            print(f"    Warning: Fold {fold_id} has no samples, skipping")
            continue
        
        # Check required columns
        if y_true_col not in df_fold.columns:
            raise ValueError(f"Column '{y_true_col}' not found for fold {fold_id}")
        if score_col not in df_fold.columns:
            raise ValueError(f"Column '{score_col}' not found for fold {fold_id}")
        
        # Get valid samples (non-null scores and labels)
        df_fold_valid = df_fold.dropna(subset=[score_col, y_true_col])
        
        if len(df_fold_valid) == 0:
            print(f"    Warning: Fold {fold_id} has no valid samples after dropping NaN, skipping")
            continue
        
        y_true = df_fold_valid[y_true_col].values.astype(int)
        y_scores = df_fold_valid[score_col].values.astype(float)
        
        # Check if we have both classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            print(f"    Warning: Fold {fold_id} has only one class ({unique_labels}), skipping")
            continue
        
        # Find threshold for target FPR
        threshold, actual_fpr = find_threshold_for_fpr(y_true, y_scores, target_fpr=target_fpr)
        
        
        # Compute additional metrics at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 1:
                    TN, FP, FN, TP = 0, (y_true == 0).sum(), 0, (y_true == 1).sum()
                else:
                    TN, FP, FN, TP = (y_true == 0).sum(), 0, (y_true == 1).sum(), 0
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        results[fold_id] = {
            'threshold': threshold,
            'fpr': actual_fpr,
            'tpr': tpr,
            'n_samples': len(df_fold_valid),
            'n_negatives': (y_true == 0).sum(),
            'n_positives': (y_true == 1).sum(),
            'TP': int(TP),
            'FP': int(FP),
            'TN': int(TN),
            'FN': int(FN)
        }
        
        print(f"    Fold {fold_id}: threshold={threshold:.4f}, FPR={actual_fpr:.4f}, TPR={tpr:.4f}, n={len(df_fold_valid)}")
    
    return results


def process_scores_file(
    scores_path: str,
    target_fpr: float = 0.05,
    fold_id_col: str = "fold_id",
    y_true_col: str = "y_true",
    score_col: str = "score_oof_member_proba"
) -> Optional[Dict]:
    """
    Process a single scores CSV file and compute averaged threshold.
    
    Returns:
        Dictionary with results or None if processing failed
    """
    print(f"\n{'='*80}")
    print(f"Processing: {scores_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(scores_path):
        print(f"  ERROR: File does not exist")
        return None
    
    try:
        df = pd.read_csv(scores_path)
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"  ERROR: Failed to load CSV: {e}")
        return None
    
    # Check required columns
    required_cols = [fold_id_col, y_true_col, score_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ERROR: Missing required columns: {missing_cols}")
        print(f"  Available columns: {df.columns.tolist()}")
        return None
    
    # Compute threshold per fold
    try:
        fold_results = compute_threshold_per_fold(
            df, 
            fold_id_col=fold_id_col,
            y_true_col=y_true_col,
            score_col=score_col,
            target_fpr=target_fpr
        )
    except Exception as e:
        print(f"  ERROR: Failed to compute thresholds: {e}")
        return None
    
    if len(fold_results) == 0:
        print(f"  ERROR: No valid fold results")
        return None
    
    # Average thresholds across folds
    thresholds = [r['threshold'] for r in fold_results.values()]
    fprs = [r['fpr'] for r in fold_results.values()]
    tprs = [r['tpr'] for r in fold_results.values()]
    
    avg_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    avg_fpr = np.mean(fprs)
    avg_tpr = np.mean(tprs)
    
    print(f"\n  Summary:")
    print(f"    Number of folds: {len(fold_results)}")
    print(f"    Average threshold: {avg_threshold:.4f} ± {std_threshold:.4f}")
    print(f"    Average FPR: {avg_fpr:.4f}")
    print(f"    Average TPR: {avg_tpr:.4f}")
    print(f"    Threshold range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
    
    return {
        'file_path': scores_path,
        'n_folds': len(fold_results),
        'avg_threshold': avg_threshold,
        'std_threshold': std_threshold,
        'avg_fpr': avg_fpr,
        'avg_tpr': avg_tpr,
        'min_threshold': min(thresholds),
        'max_threshold': max(thresholds),
        'fold_results': fold_results
    }


def find_scores_files(base_dir: str) -> List[str]:
    """
    Find all scores CSV files in the experimental-recall-output directory structure.
    
    Pattern: {base_dir}/*/scores_*_p.csv
    """
    scores_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return scores_files
    
    # Find all scores_*.csv files in subdirectories
    for scores_file in base_path.glob("*/scores_*_p.csv"):
        scores_files.append(str(scores_file))
    
    return sorted(scores_files)


def main():
    parser = argparse.ArgumentParser(
        description='Compute best threshold for FPR < 5% using cross-fitting correctly'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output',
        help='Base directory containing experimental-recall-output subdirectories'
    )
    parser.add_argument(
        '--target-fpr',
        type=float,
        default=0.05,
        help='Target FPR threshold (default: 0.05 for 5%%)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Path to save results CSV (default: threshold_fpr5_results.csv in base_dir)'
    )
    parser.add_argument(
        '--fold-id-col',
        type=str,
        default='fold_id',
        help='Column name for fold ID (default: fold_id)'
    )
    parser.add_argument(
        '--y-true-col',
        type=str,
        default='y_true',
        help='Column name for true labels (default: y_true)'
    )
    parser.add_argument(
        '--score-col',
        type=str,
        default='score_oof_member_proba',
        help='Column name for scores (default: score_oof_member_proba)'
    )
    
    args = parser.parse_args()
    
    # Find all scores files
    print(f"Searching for scores files in: {args.base_dir}")
    scores_files = find_scores_files(args.base_dir)
    
    if len(scores_files) == 0:
        print(f"ERROR: No scores files found in {args.base_dir}")
        return 1
    
    print(f"Found {len(scores_files)} scores files")
    
    # Process each file
    all_results = []
    failed_files = []
    
    for scores_file in scores_files:
        result = process_scores_file(
            scores_file,
            target_fpr=args.target_fpr,
            fold_id_col=args.fold_id_col,
            y_true_col=args.y_true_col,
            score_col=args.score_col
        )
        
        if result is not None:
            all_results.append(result)
        else:
            failed_files.append(scores_file)
    
    # Create summary DataFrame
    if len(all_results) == 0:
        print("\nERROR: No files processed successfully")
        return 1
    
    summary_data = []
    for result in all_results:
        # Extract directory name from file path
        dir_name = os.path.basename(os.path.dirname(result['file_path']))
        file_name = os.path.basename(result['file_path'])
        
        summary_data.append({
            'directory': dir_name,
            'scores_file': file_name,
            'n_folds': result['n_folds'],
            'avg_threshold': result['avg_threshold'],
            'std_threshold': result['std_threshold'],
            'avg_fpr': result['avg_fpr'],
            'avg_tpr': result['avg_tpr'],
            'min_threshold': result['min_threshold'],
            'max_threshold': result['max_threshold'],
            'file_path': result['file_path']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save results
    if args.output_csv is None:
        output_csv = os.path.join(args.base_dir, 'threshold_fpr5_results.csv')
    else:
        output_csv = args.output_csv
    
    df_summary.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Total files processed: {len(all_results)}")
    print(f"  Failed files: {len(failed_files)}")
    print(f"\nResults summary:")
    print(df_summary[['directory', 'n_folds', 'avg_threshold', 'std_threshold', 'avg_fpr', 'avg_tpr']].to_string(index=False))
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    return 0


if __name__ == '__main__':
    exit(main())
