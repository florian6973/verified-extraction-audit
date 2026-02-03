import pandas as pd
import subprocess
import tempfile
import os
import argparse

from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir, get_src_ll_file_base, get_src_ll_file
from src.evaluation.pipeline.experimental.config_loader import load_config

from src._repo import REPO_ROOT
parser = argparse.ArgumentParser(description='Compute scores for MIA')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

data_path = os.path.join(get_output_dir(config), "all_names_ll_computed.csv")
model = config['filters']['model']
dataset_size = config['filters']['dataset_size']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']
# data_path = " + REPO_ROOT + "/outputs/pii_leakage/experimental-recall-0.1/all_names_ll_computed.csv"

src_pred = os.path.join(get_output_dir(config), f"scores_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}.csv")
src_other = os.path.join(get_output_dir(config), f"models_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}")
# src_pred = " + REPO_ROOT + "/outputs/pii_leakage/pipeline/plots/mia-verifier/scores_1B_10_pii_rate_0.1_n_epochs_3.csv"
# src_other = " + REPO_ROOT + "/outputs/pii_leakage/pipeline/plots/mia-verifier/models_1B_10_pii_rate_0.1_n_epochs_3"


def compute_scores(data_path, src_pred, src_other, output_path=None, name_col="value", score_col="score_oof_member_proba"):
    """
    Compute scores for names in data_path.
    If name is in src_pred, use the score from src_pred.
    Otherwise, put all names in a single CSV file and call train_mia_verifier_cv.py score_unseen.
    
    Args:
        data_path: Path to CSV file with names and features
        src_pred: Path to CSV file with pre-computed scores (has name_col and score_col)
        src_other: Path to models directory for train_mia_verifier_cv.py
        output_path: Path to save the final results CSV (if None, auto-generates from data_path)
        name_col: Column name for names (default: "value")
        score_col: Column name for scores in src_pred (default: "score_oof_member_proba")
    
    Returns:
        DataFrame with names and their scores (including all probability score columns)
    """
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Skipping.")
        return pd.DataFrame()
    # Read data
    df_data = pd.read_csv(data_path)
    # Rename 'name' column to name_col if it exists
    if 'name' in df_data.columns and name_col not in df_data.columns:
        df_data = df_data.rename(columns={'name': name_col})
    
    # Ensure name_col exists
    if name_col not in df_data.columns:
        raise ValueError(f"name_col '{name_col}' not found in data. Available columns: {df_data.columns.tolist()}")
    
    print(f"Loaded {len(df_data)} rows from {data_path}")
    print(f"Columns in data: {df_data.columns.tolist()}")
    
    # Read src_pred if it exists
    if os.path.exists(src_pred):
        df_pred = pd.read_csv(src_pred)
        # Determine the actual score column name in src_pred
        # Try score_col first, then score_member_proba_mean, then score_oof_member_proba
        actual_score_col = None
        for col in [score_col, "score_member_proba_mean", "score_oof_member_proba"]:
            if col in df_pred.columns:
                actual_score_col = col
                break
        
        if actual_score_col is None:
            print(f"Warning: No score column found in {src_pred}. Available columns: {df_pred.columns.tolist()}")
            df_pred = pd.DataFrame()
            names_with_scores = set()
        else:
            # Get names that already have scores
            names_with_scores = set(df_pred[name_col].dropna().unique())
            print(f"Found {len(names_with_scores)} names with pre-computed scores in {src_pred} (using column: {actual_score_col})")
    else:
        df_pred = pd.DataFrame()
        names_with_scores = set()
        actual_score_col = None
        print(f"Warning: {src_pred} not found. All names will be scored using train_mia.")
    
    # Split data into names with scores and names without scores
    df_with_scores = df_data[df_data[name_col].isin(names_with_scores)].copy()
    df_without_scores = df_data[~df_data[name_col].isin(names_with_scores)].copy()
    
    print(f"Names with pre-computed scores: {len(df_with_scores)}")
    print(f"Names to score with train_mia: {len(df_without_scores)}")
    
    # Merge scores for names that already have them
    if len(df_with_scores) > 0 and len(df_pred) > 0 and actual_score_col is not None:
        print(f"\nMerging pre-computed scores:")
        print(f"  df_with_scores shape: {df_with_scores.shape}")
        print(f"  df_pred shape: {df_pred.shape}")
        print(f"  actual_score_col: {actual_score_col}")
        
        # Get all probability-related columns from src_pred (keep all score columns)
        proba_cols = [col for col in df_pred.columns if 'proba' in col.lower() or 'score' in col.lower()]
        if name_col not in proba_cols:
            proba_cols = [name_col] + proba_cols
        
        print(f"  Probability columns from src_pred: {proba_cols}")
        
        df_pred_unique = df_pred[proba_cols].drop_duplicates(subset=[name_col])
        print(f"  df_pred_unique shape after deduplication: {df_pred_unique.shape}")
        
        df_with_scores = df_with_scores.merge(
            df_pred_unique, 
            on=name_col, 
            how='left',
            suffixes=('', '_from_pred')
        )
        
        print(f"  After merge shape: {df_with_scores.shape}")
        print(f"  Columns after merge: {df_with_scores.columns.tolist()}")
        
        # Handle column name conflicts - keep the _from_pred version and rename
        for col in proba_cols:
            if col != name_col:
                if f"{col}_from_pred" in df_with_scores.columns:
                    # Remove the original if it exists, keep the _from_pred version
                    if col in df_with_scores.columns:
                        df_with_scores = df_with_scores.drop(columns=[col])
                    df_with_scores = df_with_scores.rename(columns={f"{col}_from_pred": col})
        
        # Ensure score_col exists (use actual_score_col if score_col doesn't exist)
        if score_col not in df_with_scores.columns and actual_score_col in df_with_scores.columns:
            df_with_scores[score_col] = df_with_scores[actual_score_col]
        
        # Check if scores were merged successfully
        if score_col in df_with_scores.columns:
            non_null_scores = df_with_scores[score_col].notna().sum()
            print(f"  {non_null_scores}/{len(df_with_scores)} rows have non-null {score_col} after merge")
        else:
            print(f"  Warning: {score_col} not found in df_with_scores after merge!")
    
    # Score names without pre-computed scores using train_mia
    if len(df_without_scores) > 0:
        # Get unique names (in case there are duplicates)
        df_unseen = df_without_scores.drop_duplicates(subset=[name_col])
        
        # Detect feature columns (columns that start with 'ft_' or 'qi_' or contain feature-like patterns)
        # Exclude name_col and score/proba columns
        exclude_patterns = ['proba', 'score', name_col.lower()]
        feature_cols = [col for col in df_unseen.columns 
                       if not any(pattern in col.lower() for pattern in exclude_patterns)
                       and (col.startswith('ft_') or col.startswith('qi_') or col.startswith('ll_'))]
        
        # If no feature columns detected with prefixes, try to get from manifest or use all numeric columns
        if not feature_cols:
            # Try to load manifest to get feature columns
            try:
                import json
                manifest_path = os.path.join(src_other, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        feature_cols = manifest.get("feature_cols", [])
                        # Filter to only columns that exist in df_unseen
                        feature_cols = [col for col in feature_cols if col in df_unseen.columns]
            except:
                pass
        
        # If still no feature columns, use all numeric columns except name_col
        if not feature_cols:
            numeric_cols = df_unseen.select_dtypes(include=[float, int]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != name_col]
        
        if not feature_cols:
            raise ValueError(f"Could not detect feature columns. Available columns: {df_unseen.columns.tolist()}")
        
        print(f"Using feature columns: {feature_cols}")
        
        # Create temporary CSV file with features
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            df_unseen.to_csv(temp_csv, index=False)
        
        # Create temporary output CSV path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_out_csv = f.name
        
        try:
            # Call train_mia_verifier_cv.py score_unseen
            script_path = " + REPO_ROOT + "/src/evaluation/pipeline/experimental/mia/train_mia_verifier_cv.py"
            cmd = [
                "python", script_path, "score_unseen",
                "--models_dir", src_other,
                "--features_csv", temp_csv,
                "--out_csv_path", temp_out_csv,
                "--feature_cols", ",".join(feature_cols)
            ]
            
            print(f"Calling: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running train_mia_verifier_cv.py:")
                print(result.stderr)
                raise RuntimeError(f"train_mia_verifier_cv.py failed with return code {result.returncode}")
            
            # Read the scored results
            df_scored = pd.read_csv(temp_out_csv)
            print(f"Scored results shape: {df_scored.shape}")
            print(f"Scored results columns: {df_scored.columns.tolist()}")
            print(f"Sample scored data:\n{df_scored.head()}")
            
            # Get all probability-related columns from train_mia output (keep all score columns)
            proba_cols_scored = [col for col in df_scored.columns if 'proba' in col.lower() or 'score' in col.lower()]
            if name_col not in proba_cols_scored:
                proba_cols_scored = [name_col] + proba_cols_scored
            
            # train_mia outputs score_member_proba_mean, also create score_col for consistency
            if "score_member_proba_mean" in df_scored.columns:
                df_scored[score_col] = df_scored["score_member_proba_mean"]
                print(f"Created {score_col} from score_member_proba_mean")
            elif score_col not in df_scored.columns:
                raise ValueError(f"Score column not found in scored results. Available columns: {df_scored.columns.tolist()}")
            
            # Check for matching names before merge
            print(f"\nBefore merge:")
            print(f"  df_without_scores shape: {df_without_scores.shape}")
            print(f"  df_without_scores {name_col} unique count: {df_without_scores[name_col].nunique()}")
            print(f"  df_scored shape: {df_scored.shape}")
            print(f"  df_scored {name_col} unique count: {df_scored[name_col].nunique()}")
            
            # Check for name matching issues
            names_in_unseen = set(df_without_scores[name_col].dropna().unique())
            names_in_scored = set(df_scored[name_col].dropna().unique())
            missing_in_scored = names_in_unseen - names_in_scored
            if missing_in_scored:
                print(f"Warning: {len(missing_in_scored)} names in df_without_scores not found in df_scored")
                print(f"  Sample missing names: {list(missing_in_scored)[:5]}")
            
            # Merge back with original data to preserve all rows and all probability columns
            cols_to_merge = list(set(proba_cols_scored + [score_col]))  # Remove duplicates
            print(f"Merging columns: {cols_to_merge}")
            
            df_without_scores = df_without_scores.merge(
                df_scored[cols_to_merge], 
                on=name_col, 
                how='left'
            )
            
            # Check if scores were merged successfully
            if score_col in df_without_scores.columns:
                non_null_scores = df_without_scores[score_col].notna().sum()
                print(f"After merge: {non_null_scores}/{len(df_without_scores)} rows have non-null {score_col}")
            else:
                print(f"Warning: {score_col} not found in df_without_scores after merge!")
            
            print(f"Successfully scored {len(df_unseen)} names using train_mia")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            if os.path.exists(temp_out_csv):
                os.remove(temp_out_csv)
    
    # Combine results
    if len(df_with_scores) > 0 and len(df_without_scores) > 0:
        df_final = pd.concat([df_with_scores, df_without_scores], ignore_index=True)
    elif len(df_with_scores) > 0:
        df_final = df_with_scores
    elif len(df_without_scores) > 0:
        df_final = df_without_scores
    else:
        df_final = pd.DataFrame()
    
    # Debug: Check score columns before saving
    print(f"\nBefore saving:")
    print(f"  df_final shape: {df_final.shape}")
    print(f"  df_final columns: {df_final.columns.tolist()}")
    
    # List all probability score columns
    proba_cols_final = [col for col in df_final.columns if 'proba' in col.lower() or 'score' in col.lower()]
    print(f"  Probability score columns found: {proba_cols_final}")
    
    if proba_cols_final:
        for col in proba_cols_final:
            if col in df_final.columns:
                non_null = df_final[col].notna().sum()
                print(f"  {col}: {non_null}/{len(df_final)} non-null values")
                if non_null > 0:
                    print(f"    Min: {df_final[col].min():.4f}, Max: {df_final[col].max():.4f}, Mean: {df_final[col].mean():.4f}")
    
    # Save results to CSV
    if output_path is None:
        # Auto-generate output path from data_path
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        output_dir = os.path.dirname(data_path)
        output_path = os.path.join(output_dir, f"{base_name}_with_scores.csv")
    
    if len(df_final) > 0:
        df_final.to_csv(output_path, index=False)
        print(f"\nSaved results with all probability scores to: {output_path}")
        print(f"Total rows: {len(df_final)}")
    else:
        print("Warning: No results to save.")
    
    return df_final


if __name__ == "__main__":
    # Optionally specify output path, otherwise auto-generates from data_path
    output_path = None  # Set to a specific path if desired
    
    df_result = compute_scores(data_path, src_pred, src_other, output_path=output_path)
    print(f"\nFinal result: {len(df_result)} rows")
    print(f"All columns: {df_result.columns.tolist()}")
    
    # Show statistics for all probability score columns
    proba_cols = [col for col in df_result.columns if 'proba' in col.lower() or 'score' in col.lower()]
    if proba_cols:
        print(f"\nProbability score statistics:")
        for col in proba_cols:
            if col in df_result.columns:
                print(f"\n{col}:")
                print(df_result[col].describe())

# filter with if needed: outputs/pii_leakage/probability_universe_distribution_names.csv