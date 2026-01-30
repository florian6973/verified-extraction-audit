#!/usr/bin/env python3
"""
Pattern B: K-fold cross-fitting verifier + ensemble scoring for unseen names.

INPUT CSV must contain:
  - split column with "train" (member/positive) and "val" (non-member/negative)
  - one or MORE feature columns (numeric), passed via --feature_cols
    e.g. --feature_cols "loss_gap,logp_ft,logp_pre"

OPTIONAL:
  - name/id column (kept in output, not required for training)

OUTPUTS:
  1) augmented CSV with out-of-fold (OOF) scores for every row
  2) K fold-model files saved to disk + a manifest.json
     -> for names not in the list, compute the same features and score with ALL K models,
        aggregate (mean by default).

Usage:
  python mia_verifier_patternB.py train \
    --csv_path /path/to/data.csv \
    --out_csv_path /path/to/with_scores.csv \
    --models_dir /path/to/models_dir \
    --feature_cols "feat1,feat2,feat3" \
    --split_col split \
    --name_col value \
    --k 5 \
    --seed 0

Score unseen:
  python mia_verifier_patternB.py score_unseen \
    --models_dir /path/to/models_dir \
    --features_csv /path/to/unseen_features.csv \
    --out_csv_path /path/to/unseen_scores.csv

Where unseen_features.csv contains the SAME feature columns (and optionally name_col).
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir
from src.evaluation.pipeline.experimental.config_loader import load_config


# -----------------------
# Utilities
# -----------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_feature_cols(s: str) -> List[str]:
    cols = [c for c in s.split(",") if c != ""]
    if len(cols) == 0:
        raise ValueError("No feature columns provided. Use --feature_cols 'feat1,feat2,...'")
    return cols


def _label_from_split(split_series: pd.Series) -> np.ndarray:
    """
    Train = member = 1
    Val   = non-member = 0
    """
    s = split_series.astype(str).str.lower()
    y = np.where(s == "train", 1, np.where(s == "val", 0, -1))
    if np.any(y == -1):
        bad = split_series[y == -1].unique()
        raise ValueError(f"Found unknown split values: {bad}. Expected 'train'/'val'.")
    return y.astype(int)


def _build_model(seed: int) -> Pipeline:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])


@dataclass
class Manifest:
    k: int
    seed: int
    feature_cols: List[str]
    split_col: str
    name_col: Optional[str]
    positive_label: str
    negative_label: str
    aggregation: str
    model_paths: List[str]

    def to_dict(self) -> Dict:
        return {
            "k": self.k,
            "seed": self.seed,
            "feature_cols": self.feature_cols,
            "split_col": self.split_col,
            "name_col": self.name_col,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "aggregation": self.aggregation,
            "model_paths": self.model_paths,
        }


# -----------------------
# Core I/O + featurization
# -----------------------
def _coerce_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NA/non-numeric in {feature_cols}.")
    return df


def _get_X(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=float)


# -----------------------
# Training (Pattern B)
# -----------------------
def train_crossfit_and_save(
    csv_path: str,
    out_csv_path: str,
    models_dir: str,
    feature_cols: List[str],
    split_col: str = "split",
    name_col: Optional[str] = None,
    k: int = 5,
    seed: int = 0,
) -> None:
    _ensure_dir(models_dir)
    df = pd.read_csv(csv_path).copy()

    # Validate columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Available: {list(df.columns)}")
    if split_col not in df.columns:
        raise ValueError(f"split_col='{split_col}' not found. Columns: {list(df.columns)}")
    if name_col is not None and name_col not in df.columns:
        raise ValueError(f"name_col='{name_col}' not found. Columns: {list(df.columns)}")

    # Clean
    df = _coerce_features(df, feature_cols)
    y = _label_from_split(df[split_col])
    X = _get_X(df, feature_cols)

    # Cross-fitting
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_scores = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    fold_id = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    model_paths: List[str] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=0):
        model = _build_model(seed + fold)
        model.fit(X[train_idx], y[train_idx])

        proba = model.predict_proba(X[test_idx])[:, 1]  # P(member)
        oof_scores[test_idx] = proba
        fold_id[test_idx] = fold

        model_path = os.path.join(models_dir, f"fold_{fold}.joblib")
        joblib.dump(model, model_path)
        model_paths.append(model_path)

        print(f"[fold {fold}] saved -> {model_path} | holdout n={len(test_idx)}")

    assert np.all(np.isfinite(oof_scores)), "Some OOF scores are NaN; check folds / labels."

    # Save augmented CSV
    df_out = df.copy()
    df_out["y_true"] = y
    df_out["fold_id"] = fold_id
    df_out["score_oof_member_proba"] = oof_scores

    # Convenience decision at 0.5 (you will likely calibrate tau elsewhere)
    df_out["pred_oof_at_0p5"] = (df_out["score_oof_member_proba"] >= 0.5).astype(int)

    df_out.to_csv(out_csv_path, index=False)
    print(f"Saved augmented CSV with OOF scores -> {out_csv_path}")

    # Save manifest
    manifest = Manifest(
        k=k,
        seed=seed,
        feature_cols=feature_cols,
        split_col=split_col,
        name_col=name_col,
        positive_label="train",
        negative_label="val",
        aggregation="mean",
        model_paths=model_paths,
    )
    manifest_path = os.path.join(models_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    print(f"Saved manifest -> {manifest_path}")


# -----------------------
# Scoring unseen candidates with the fold-ensemble
# -----------------------
def load_manifest(models_dir: str) -> Dict:
    manifest_path = os.path.join(models_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {models_dir}")
    with open(manifest_path, "r") as f:
        return json.load(f)


def score_unseen_df(models_dir: str, df_unseen: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Score unseen items using ALL fold models and aggregate by mean.
    Returns (df_with_scores, per_fold_scores).
    
    Args:
        models_dir: Directory containing the trained models and manifest
        df_unseen: DataFrame with unseen data to score
        feature_cols: Optional list of feature column names. If None, uses columns from manifest.
    """
    manifest = load_manifest(models_dir)
    if feature_cols is None:
        feature_cols = manifest["feature_cols"]
    model_paths = manifest["model_paths"]

    missing = [c for c in feature_cols if c not in df_unseen.columns]

    if missing:
        raise ValueError(f"Unseen features CSV is missing columns: {missing}. Expected: {feature_cols}")

    df_unseen = _coerce_features(df_unseen.copy(), feature_cols)
    X = _get_X(df_unseen, feature_cols)

    per_fold = np.zeros((X.shape[0], len(model_paths)), dtype=float)
    for j, mp in enumerate(model_paths):
        model = joblib.load(mp)
        per_fold[:, j] = model.predict_proba(X)[:, 1]

    mean_scores = per_fold.mean(axis=1)
    df_unseen["score_member_proba_mean"] = mean_scores
    for j in range(per_fold.shape[1]):
        df_unseen[f"score_member_proba_fold{j}"] = per_fold[:, j]

    return df_unseen, per_fold


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train K-fold cross-fit verifier and save OOF scores + models.")
    p_train.add_argument("--config", type=str, default=None, help="Path to config file")

    # p_train.add_argument("--csv_path", required=True)
    # p_train.add_argument("--out_csv_path", required=True)
    # p_train.add_argument("--models_dir", required=True)
    # p_train.add_argument("--feature_cols", required=True, help="Comma-separated feature columns, e.g. 'f1,f2,f3'")
    # p_train.add_argument("--split_col", default="split")
    # p_train.add_argument("--name_col", default=None)
    # p_train.add_argument("--k", type=int, default=5)
    # p_train.add_argument("--seed", type=int, default=0)

    p_score = sub.add_parser("score_unseen", help="Score unseen features CSV with the K-fold ensemble.")
    p_score.add_argument("--models_dir", required=True)
    p_score.add_argument("--features_csv", required=True, help="CSV containing feature columns.")
    p_score.add_argument("--out_csv_path", required=True)
    p_score.add_argument("--feature_cols", default=None, help="Comma-separated feature column names (e.g. 'ft_Name: ,ft_Patient: ,qi_Name: ,qi_Patient: '). If not provided, uses columns from manifest.")

    args = parser.parse_args()

    if args.cmd == "train":
        config = load_config(args.config)
        output_dir = get_output_dir(config)
        model = config['filters']['model']
        dataset_size = config['filters']['dataset_size']
        pii_rate = config['filters']['pii_rate']
        n_epochs = config['filters']['n_epochs']
        csv_name = f"df_combined_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}.csv"
        models_dir = os.path.join(output_dir, f"models_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}")

        args.csv_path = os.path.join(output_dir, csv_name)
        args.models_dir = models_dir
        args.out_csv_path = os.path.join(output_dir, f"scores_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}.csv")
        args.feature_cols = f"ft_Name: ,ft_Patient: ,qi_Name: ,qi_Patient: "
        args.split_col = "split_x"
        args.name_col = "value"
        args.k = 5
        args.seed = 0

        feature_cols = _parse_feature_cols(args.feature_cols)
        train_crossfit_and_save(
            csv_path=args.csv_path,
            out_csv_path=args.out_csv_path,
            models_dir=args.models_dir,
            feature_cols=feature_cols,
            split_col=args.split_col,
            name_col=args.name_col,
            k=args.k,
            seed=args.seed,
        )

    elif args.cmd == "score_unseen":
        df_unseen = pd.read_csv(args.features_csv)
        feature_cols = None
        if args.feature_cols:
            feature_cols = _parse_feature_cols(args.feature_cols)
        df_scored, _ = score_unseen_df(args.models_dir, df_unseen, feature_cols=feature_cols)
        df_scored.to_csv(args.out_csv_path, index=False)
        print(f"Saved unseen scored CSV -> {args.out_csv_path}")

    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()

"""
python /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/evaluation/pipeline/experimental/mia/train_mia_verifier_cv.py train \
    --csv_path /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack/df_combined_1B_10_pii_rate_1.0_n_epochs_3.csv \
    --out_csv_path /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/scores_1B_10_pii_rate_1.0_n_epochs_3.csv \
    --models_dir /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/models_1B_10_pii_rate_1.0_n_epochs_3 \
    --feature_col "ft_Name: ,ft_Patient: ,qi_Name: ,qi_Patient: " \
    --split_col split_x \
    --name_col value \
    --k 5

python /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/evaluation/pipeline/experimental/mia/train_mia_verifier_cv.py train \
    --csv_path /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack/df_combined_1B_10_pii_rate_0.1_n_epochs_3.csv \
    --out_csv_path /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/scores_1B_10_pii_rate_0.1_n_epochs_3.csv \
    --models_dir /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/models_1B_10_pii_rate_0.1_n_epochs_3 \
    --feature_col "ft_Name: ,ft_Patient: ,qi_Name: ,qi_Patient: " \
    --split_col split_x \
    --name_col value \
    --k 5


  python  /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/evaluation/pipeline/experimental/mia/train_mia_verifier_cv.py score_unseen \
    --models_dir /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/models_1B_10_pii_rate_1.0_n_epochs_3 \
    --features_csv /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/evaluation/pipeline/experimental/mia/test.csv \
    --out_csv_path /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/models_1B_10_pii_rate_1.0_n_epochs_3/extracted_test.csv
"""