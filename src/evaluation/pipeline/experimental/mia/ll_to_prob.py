from src._repo import REPO_ROOT
"""
Convert log-likelihood (LL) columns to probability columns in a CSV file.
Replaces LL columns with probability columns (exp(LL) normalized).
"""

import pandas as pd
import numpy as np
import os

import argparse
from src.evaluation.pipeline.experimental.config_loader import load_config
from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir

parser = argparse.ArgumentParser(description='Convert LL to probability')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

model = config['filters']['model']
dataset_size = config['filters']['dataset_size']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']

# Input CSV file
input_csv = os.path.join(get_output_dir(config), f"scores_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}.csv")
# input_csv = " + REPO_ROOT + "/outputs/pii_leakage/pipeline/plots/mia-verifier/scores_1B_10_pii_rate_0.1_n_epochs_3.csv"


def convert_ll_to_prob(df, ll_col):
    """
    Convert a log-likelihood column to probability.
    Uses exp(LL) for each value individually.
    For numerical stability, uses exp(LL - max(LL)) then scales back.
    
    Args:
        df: DataFrame
        ll_col: Column name with log-likelihood values
    
    Returns:
        Array with probability values
    """
    ll_values = df[ll_col].values
    
    # Handle NaN and inf values
    valid_mask = np.isfinite(ll_values)
    prob_values = np.full_like(ll_values, np.nan, dtype=float)
    
    if valid_mask.sum() > 0:
        ll_valid = ll_values[valid_mask]
        
        # For numerical stability, subtract max before exp
        # # This prevents overflow for large negative LL values
        # ll_max = ll_valid.max()
        # ll_shifted = ll_valid - ll_max
        
        # # Clip very negative values to prevent underflow
        # ll_shifted = np.clip(ll_shifted, -700, None)  # exp(-700) is very close to 0
        
        # Convert to probability: exp(LL)
        ll_shifted = ll_valid
        exp_ll = np.exp(ll_shifted)
        
        # Store probabilities
        prob_values[valid_mask] = exp_ll
    
    return prob_values


def convert_all_ll_to_prob(input_path, output_path=None):
    """
    Convert all LL columns in a CSV to probability columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, overwrites input)
    """
    print(f"Loading CSV: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Identify LL columns
    # Look for columns starting with 'ft_', 'qi_', or containing 'll' (case insensitive)
    ll_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if (col.startswith('ft_') or 
            col.startswith('qi_') or 
            'll' in col_lower or
            col_lower.startswith('ll_')):
            # Check if values look like log-likelihoods (typically negative or small values)
            if df[col].dtype in [float, int]:
                sample_values = df[col].dropna()
                if len(sample_values) > 0:
                    # Log-likelihoods are typically negative or small positive values
                    # Probabilities are typically between 0 and 1
                    min_val = sample_values.min()
                    max_val = sample_values.max()
                    # If values are mostly negative or very small, likely LL
                    # If values are between 0 and 1, likely already probabilities
                    if min_val < -1 or (min_val < 0 and max_val < 10):
                        ll_columns.append(col)
                        print(f"  Found LL column: {col} (range: [{min_val:.2f}, {max_val:.2f}])")
    
    if not ll_columns:
        print("No LL columns found. Available columns:")
        for col in df.columns:
            if df[col].dtype in [float, int]:
                sample = df[col].dropna()
                if len(sample) > 0:
                    print(f"  {col}: range [{sample.min():.2f}, {sample.max():.2f}]")
        return
    
    print(f"\nConverting {len(ll_columns)} LL columns to probabilities...")
    
    # Convert each LL column to probability
    prob_columns = {}
    for ll_col in ll_columns:
        # Create probability column name
        # ft_ columns -> p_ft_, qi_ columns -> p_base_
        if ll_col.startswith('ft_'):
            prob_col = 'p_ft_' + ll_col[3:]  # 'ft_Name: ' -> 'p_ft_Name: '
        elif ll_col.startswith('qi_'):
            prob_col = 'p_base_' + ll_col[3:]  # 'qi_Name: ' -> 'p_base_Name: '
        elif 'll' in ll_col.lower():
            # Replace 'll' with 'prob'
            prob_col = ll_col.replace('ll', 'prob').replace('LL', 'prob')
        else:
            # Add 'prob_' prefix
            prob_col = 'prob_' + ll_col
        
        # Convert LL to probability
        prob_values = convert_ll_to_prob(df, ll_col)
        prob_columns[prob_col] = prob_values
        print(f"  {ll_col} -> {prob_col}")
    
    # Add probability columns to dataframe
    for prob_col, prob_values in prob_columns.items():
        df[prob_col] = prob_values
    
    # Drop original LL columns
    df = df.drop(columns=ll_columns)
    
    print(f"\nAfter conversion:")
    print(f"  Shape: {df.shape}")
    print(f"  Removed {len(ll_columns)} LL columns")
    print(f"  Added {len(prob_columns)} probability columns")
    print(f"  New columns: {df.columns.tolist()}")
    
    # Save to output file
    if output_path is None:
        # Add "_p" suffix before the .csv extension
        base_path = os.path.splitext(input_path)[0]
        extension = os.path.splitext(input_path)[1]
        output_path = base_path + "_p" + extension
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved converted CSV to: {output_path}")
    
    # Show statistics
    print(f"\nProbability column statistics:")
    for prob_col in prob_columns.keys():
        if prob_col in df.columns:
            prob_data = df[prob_col].dropna()
            if len(prob_data) > 0:
                print(f"  {prob_col}:")
                print(f"    Count: {len(prob_data)}")
                print(f"    Min: {prob_data.min():.6e}")
                print(f"    Max: {prob_data.max():.6e}")
                print(f"    Mean: {prob_data.mean():.6e}")
    
    return df


if __name__ == "__main__":
    # Output path will be automatically generated with "_p" suffix
    # Or specify a custom output path if desired
    output_csv = None  # None = auto-generate with "_p" suffix
    
    df_result = convert_all_ll_to_prob(input_csv, output_csv)
    print("\nConversion complete!")
