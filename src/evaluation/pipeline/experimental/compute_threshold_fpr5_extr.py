#!/usr/bin/env python3
from src._repo import REPO_ROOT
"""
Compute the best threshold for FPR < 5% using cross-fitting correctly.

NEW (optional): budgeted extracted-stream FPR constraint
------------------------------------------------------
If --budget-N is provided (float/int), we compute thresholds that enforce:

    FPR_extracted(N, tau) <= target_fpr

where extracted-stream FPR is weighted by extraction probability mass:
  w_i(N) = P(extracted at least once) = 1 - (1 - pi_i)^N
and pi_i is taken from --pi-col (default: "p_ft_Name: ").

For each CV fold, we compute tau using ONLY samples from that fold (OOF scoring),
and ONLY negatives (y_true==0) in that fold, with weights w_i(N).

Then we average tau across folds.

If --budget-N is not provided, the script reverts to the original "population FPR"
constraint using confusion-matrix FPR on the fold's samples.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix


# ---------------------------
# Basic helpers (population FPR)
# ---------------------------
def compute_fpr_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    """Compute (population) FPR = FP/(FP+TN) at a given threshold over provided samples."""
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

    return FP / (FP + TN) if (FP + TN) > 0 else 0.0


def find_threshold_for_population_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fpr: float = 0.05,
    max_threshold: float = 1.0,
    min_threshold: float = 0.0,
    n_points: int = 1000
) -> Tuple[float, float]:
    """
    Find threshold such that population FPR <= target_fpr (or closest).
    Returns: (threshold, actual_fpr)
    """
    thresholds = np.linspace(min_threshold, max_threshold, n_points)
    fprs = np.array([compute_fpr_at_threshold(y_true, y_scores, thr) for thr in thresholds])

    valid = np.where(fprs <= target_fpr)[0]
    if len(valid) > 0:
        # choose smallest threshold among valid -> most permissive (maximizes TPR)
        best_idx = valid[np.argmin(thresholds[valid])]
        return float(thresholds[best_idx]), float(fprs[best_idx])

    best_idx = int(np.argmin(np.abs(fprs - target_fpr)))
    return float(thresholds[best_idx]), float(fprs[best_idx])


# ---------------------------
# Budgeted extracted-stream FPR (weighted)
# ---------------------------
def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    """P(extracted at least once after N iid draws)."""
    # safe-ish for typical pi. If you have pi extremely tiny and N huge, this is still fine.
    return 1.0 - np.power((1.0 - pi), N)


def weighted_upper_tail_mass(scores: np.ndarray, weights: np.ndarray, tau: float) -> float:
    """Compute sum(weights[s>=tau]) / sum(weights)."""
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return 0.0
    return float(np.sum(weights[scores >= tau]) / wsum)


def find_threshold_for_extracted_fpr(
    neg_scores: np.ndarray,
    neg_weights: np.ndarray,
    target_fpr: float = 0.05
) -> Tuple[float, float]:
    """
    Find tau so that weighted extracted-stream FPR <= target_fpr, i.e.
        sum w_i 1[s_i >= tau] / sum w_i <= target_fpr

    We choose the *largest* tau that still satisfies the constraint (most permissive).
    Implemented via weighted (1-target_fpr) quantile on scores.

    Returns: (tau, achieved_fpr)
    """
    scores = np.asarray(neg_scores, dtype=float)
    weights = np.asarray(neg_weights, dtype=float)

    # drop non-positive weights (they contribute nothing)
    m = weights > 0
    scores = scores[m]
    weights = weights[m]

    if len(scores) == 0:
        # no negatives with weight: trivially safe, pick tau=1
        return 1.0, 0.0

    # Sort by score ascending
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cw = np.cumsum(w)
    total = cw[-1]

    # We want upper tail mass <= target_fpr
    # upper tail mass at threshold tau means keeping scores >= tau.
    # Equivalent: choose tau at the (1-target_fpr) weighted quantile.
    # Compute cutoff so that mass below tau is >= (1-target_fpr) * total.
    cutoff = (1.0 - target_fpr) * total
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(max(idx, 0), len(s) - 1)

    tau = float(s[idx])

    # Because of ties/discreteness, adjust tau upward if needed to be as permissive as possible
    # while still meeting constraint: try thresholds at unique score levels.
    uniq = np.unique(s)
    # candidate taus are unique scores; larger tau => fewer accepted => smaller FPR
    # We want the smallest tau that keeps FPR <= target, but "most permissive" in your earlier code
    # was smallest tau. Here, we interpret "best" as maximizing TPR while meeting FPR constraint,
    # so we want the *smallest* tau that satisfies FPR<=target.
    # However, for extracted-stream FPR on negatives only, permissive = smaller tau => larger FPR.
    # So: pick the smallest tau among those with FPR<=target.
    best_tau = None
    best_fpr = None
    for cand in uniq:
        fpr = weighted_upper_tail_mass(scores, weights, cand)
        if fpr <= target_fpr + 1e-12:
            best_tau = float(cand)
            best_fpr = float(fpr)
            break
    if best_tau is None:
        # cannot reach target even at tau=max => take max tau
        best_tau = float(uniq[-1])
        best_fpr = float(weighted_upper_tail_mass(scores, weights, best_tau))

    return best_tau, best_fpr


# ---------------------------
# Per-fold computation
# ---------------------------
def compute_threshold_per_fold(
    df: pd.DataFrame,
    fold_id_col: str = "fold_id",
    y_true_col: str = "y_true",
    score_col: str = "score_oof_member_proba",
    target_fpr: float = 0.05,
    budget_N: Optional[float] = None,
    pi_col: str = "p_ft_Name: ",
) -> Dict[int, Dict[str, float]]:
    """
    Compute threshold for each fold.

    If budget_N is None:
        enforce population FPR <= target_fpr on that fold's samples.
    Else:
        enforce extracted-stream FPR <= target_fpr on that fold's NEGATIVES,
        weighted by P(extracted|N) computed from pi_col.
    """
    results: Dict[int, Dict[str, float]] = {}

    if fold_id_col not in df.columns:
        raise ValueError(f"Column '{fold_id_col}' not found. Available columns: {df.columns.tolist()}")

    fold_ids = sorted(df[fold_id_col].dropna().unique())
    if len(fold_ids) == 0:
        raise ValueError(f"No valid fold IDs found in column '{fold_id_col}'")

    print(f"  Found {len(fold_ids)} folds: {fold_ids}")

    if budget_N is not None and pi_col not in df.columns:
        raise ValueError(f"--budget-N provided but pi_col='{pi_col}' not found. Available columns: {df.columns.tolist()}")

    for fold_id in fold_ids:
        df_fold = df[df[fold_id_col] == fold_id].copy()
        if len(df_fold) == 0:
            print(f"    Warning: Fold {fold_id} has no samples, skipping")
            continue

        # Ensure required columns
        for c in [y_true_col, score_col]:
            if c not in df_fold.columns:
                raise ValueError(f"Column '{c}' not found for fold {fold_id}")

        # Clean
        df_fold_valid = df_fold.dropna(subset=[score_col, y_true_col]).copy()
        if len(df_fold_valid) == 0:
            print(f"    Warning: Fold {fold_id} has no valid samples after dropping NaN, skipping")
            continue

        y_true = df_fold_valid[y_true_col].values.astype(int)
        y_scores = df_fold_valid[score_col].values.astype(float)

        if len(np.unique(y_true)) < 2:
            print(f"    Warning: Fold {fold_id} has only one class ({np.unique(y_true)}), skipping")
            continue

        # -----------------------
        # Pick threshold
        # -----------------------
        if budget_N is None:
            threshold, actual_fpr = find_threshold_for_population_fpr(
                y_true, y_scores, target_fpr=target_fpr
            )
            constraint_type = "population"

        else:
            # Extracted-stream constraint uses NEGATIVES only with weights P(extracted|N)
            df_neg = df_fold_valid[df_fold_valid[y_true_col].astype(int) == 0].copy()
            if len(df_neg) == 0:
                print(f"    Warning: Fold {fold_id} has no negatives, skipping")
                continue

            df_neg[pi_col] = pd.to_numeric(df_neg[pi_col], errors="coerce")
            df_neg = df_neg.dropna(subset=[pi_col])
            if len(df_neg) == 0:
                print(f"    Warning: Fold {fold_id} negatives have no valid pi, skipping")
                continue

            neg_scores = df_neg[score_col].values.astype(float)
            pi = df_neg[pi_col].values.astype(float)
            w = prob_extracted_at_least_once(pi, float(budget_N))

            threshold, actual_fpr = find_threshold_for_extracted_fpr(
                neg_scores=neg_scores,
                neg_weights=w,
                target_fpr=target_fpr
            )
            constraint_type = f"extracted@N={budget_N:g}"

        # -----------------------
        # Report fold metrics at threshold
        # -----------------------
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

        results[int(fold_id)] = {
            "threshold": float(threshold),
            "fpr": float(actual_fpr),
            "tpr": float(tpr),
            "n_samples": int(len(df_fold_valid)),
            "n_negatives": int((y_true == 0).sum()),
            "n_positives": int((y_true == 1).sum()),
            "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
            "constraint": constraint_type,
        }

        print(
            f"    Fold {fold_id}: thr={threshold:.4f}, "
            f"{constraint_type} FPR={actual_fpr:.4f}, TPR={tpr:.4f}, n={len(df_fold_valid)}"
        )

    return results


# ---------------------------
# File processing
# ---------------------------
def process_scores_file(
    scores_path: str,
    target_fpr: float = 0.05,
    fold_id_col: str = "fold_id",
    y_true_col: str = "y_true",
    score_col: str = "score_oof_member_proba",
    budget_N: Optional[float] = None,
    pi_col: str = "p_ft_Name: ",
) -> Optional[Dict]:
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

    required_cols = [fold_id_col, y_true_col, score_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        return None

    if budget_N is not None and pi_col not in df.columns:
        print(f"  ERROR: --budget-N provided but missing pi_col '{pi_col}'")
        return None

    try:
        fold_results = compute_threshold_per_fold(
            df,
            fold_id_col=fold_id_col,
            y_true_col=y_true_col,
            score_col=score_col,
            target_fpr=target_fpr,
            budget_N=budget_N,
            pi_col=pi_col,
        )
    except Exception as e:
        print(f"  ERROR: Failed to compute thresholds: {e}")
        return None

    if len(fold_results) == 0:
        print(f"  ERROR: No valid fold results")
        return None

    thresholds = [r["threshold"] for r in fold_results.values()]
    fprs = [r["fpr"] for r in fold_results.values()]
    tprs = [r["tpr"] for r in fold_results.values()]

    avg_threshold = float(np.mean(thresholds))
    std_threshold = float(np.std(thresholds))
    avg_fpr = float(np.mean(fprs))
    avg_tpr = float(np.mean(tprs))

    print(f"\n  Summary:")
    print(f"    Constraint: {'extracted-stream' if budget_N is not None else 'population'}")
    if budget_N is not None:
        print(f"    Budget N: {budget_N:g}  (weights from pi_col='{pi_col}')")
    print(f"    Number of folds: {len(fold_results)}")
    print(f"    Average threshold: {avg_threshold:.4f} ± {std_threshold:.4f}")
    print(f"    Average FPR (constraint metric): {avg_fpr:.4f}")
    print(f"    Average TPR: {avg_tpr:.4f}")
    print(f"    Threshold range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")

    return {
        "file_path": scores_path,
        "n_folds": len(fold_results),
        "avg_threshold": avg_threshold,
        "std_threshold": std_threshold,
        "avg_fpr": avg_fpr,
        "avg_tpr": avg_tpr,
        "min_threshold": float(min(thresholds)),
        "max_threshold": float(max(thresholds)),
        "fold_results": fold_results,
        "constraint": ("extracted" if budget_N is not None else "population"),
        "budget_N": (float(budget_N) if budget_N is not None else None),
        "pi_col": (pi_col if budget_N is not None else None),
    }


def find_scores_files(base_dir: str) -> List[str]:
    scores_files: List[str] = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return scores_files

    for scores_file in base_path.glob("*/scores_*_p.csv"):
        scores_files.append(str(scores_file))

    return sorted(scores_files)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute best threshold for FPR < target using cross-fitting correctly"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "outputs", "pii_leakage", "experimental-recall-output"),
        help="Base directory containing experimental-recall-output subdirectories"
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target FPR (default 0.05)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results CSV (default: threshold_results.csv in base_dir)"
    )
    parser.add_argument(
        "--fold-id-col",
        type=str,
        default="fold_id"
    )
    parser.add_argument(
        "--y-true-col",
        type=str,
        default="y_true"
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default="score_oof_member_proba"
    )

    # NEW: budgeted extracted-stream constraint
    parser.add_argument(
        "--budget-N",
        type=float,
        default=None,
        help="If set, choose thresholds to ensure extracted-stream FPR at this budget <= target-fpr "
             "(weighted by w_i(N)=1-(1-pi_i)^N on negatives)."
    )
    parser.add_argument(
        "--pi-col",
        type=str,
        default="p_ft_Name: ",
        help="Column for per-draw extraction probability pi_i (default: 'p_ft_Name: ')"
    )

    args = parser.parse_args()

    print(f"Searching for scores files in: {args.base_dir}")
    scores_files = find_scores_files(args.base_dir)

    if len(scores_files) == 0:
        print(f"ERROR: No scores files found in {args.base_dir}")
        return 1

    print(f"Found {len(scores_files)} scores files")

    all_results = []
    failed_files = []

    for scores_file in scores_files:
        result = process_scores_file(
            scores_file,
            target_fpr=args.target_fpr,
            fold_id_col=args.fold_id_col,
            y_true_col=args.y_true_col,
            score_col=args.score_col,
            budget_N=args.budget_N,
            pi_col=args.pi_col,
        )

        if result is not None:
            all_results.append(result)
        else:
            failed_files.append(scores_file)

    if len(all_results) == 0:
        print("\nERROR: No files processed successfully")
        return 1

    summary_data = []
    for result in all_results:
        dir_name = os.path.basename(os.path.dirname(result["file_path"]))
        file_name = os.path.basename(result["file_path"])
        summary_data.append({
            "directory": dir_name,
            "scores_file": file_name,
            "n_folds": result["n_folds"],
            "avg_threshold": result["avg_threshold"],
            "std_threshold": result["std_threshold"],
            "avg_fpr": result["avg_fpr"],
            "avg_tpr": result["avg_tpr"],
            "min_threshold": result["min_threshold"],
            "max_threshold": result["max_threshold"],
            "constraint": result["constraint"],
            "budget_N": result["budget_N"],
            "pi_col": result["pi_col"],
            "file_path": result["file_path"],
        })

    df_summary = pd.DataFrame(summary_data)

    if args.output_csv is None:
        suffix = "extracted" if args.budget_N is not None else "population"
        output_csv = os.path.join(args.base_dir, f"threshold_{suffix}_fpr{args.target_fpr:g}_results.csv")
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
    cols = ["directory", "n_folds", "avg_threshold", "std_threshold", "avg_fpr", "avg_tpr", "constraint", "budget_N"]
    print(df_summary[cols].to_string(index=False))

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
