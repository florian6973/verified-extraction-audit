# Convert parquet files to CSV for compatibility with the rest of the pipeline

import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path


def convert_parquet_to_csv(parquet_path, csv_path=None, overwrite=False):
    """
    Convert a parquet file to CSV format.
    
    Args:
        parquet_path: Path to the input parquet file
        csv_path: Path to the output CSV file (default: same as parquet but with .csv extension)
        overwrite: If True, overwrite existing CSV files
    """
    if csv_path is None:
        csv_path = parquet_path.replace('.parquet', '.csv')
    
    if os.path.exists(csv_path) and not overwrite:
        print(f"Skipping existing file: {csv_path}")
        return
    
    print(f"Converting {parquet_path} to {csv_path}")

    # if "1000000" in parquet_path:
        # print("Skipping 1000000")
        # return
    
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    
    # Ensure tokens column is properly formatted (convert list to string representation for CSV compatibility)
    if 'tokens' in df.columns:
        # Convert list column to string representation for CSV compatibility
        # This matches the format used in experimental_recall.py where lists are stored as strings
        df['tokens'] = df['tokens'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Ensure ll column is properly formatted
    # In experimental_recall_2.py, ll is a float (total log-likelihood)
    # In experimental_recall.py, ll is a list (log-likelihood per token)
    # We'll keep it as-is since the parquet format uses float
    if 'll' in df.columns and df['ll'].dtype == 'object':
        # If somehow ll is a list, convert it to string representation
        df['ll'] = df['ll'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Write to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    print(f"Successfully converted to {csv_path}")


def convert_all_parquet_files(base_dir, pattern="generation_*_all_*.parquet", overwrite=False):
    """
    Convert all parquet files matching the pattern in the base directory.
    
    Args:
        base_dir: Base directory to search for parquet files
        pattern: Glob pattern to match parquet files (default: "generation_*_all_*.parquet")
        overwrite: If True, overwrite existing CSV files
    """
    base_path = Path(base_dir)
    parquet_files = list(base_path.rglob(pattern))
    
    if not parquet_files:
        print(f"No parquet files found matching pattern '{pattern}' in {base_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet file(s) to convert")
    
    for parquet_path in tqdm(parquet_files, desc="Converting parquet to CSV"):
        convert_parquet_to_csv(str(parquet_path), overwrite=overwrite)


if __name__ == "__main__":
    # Default output directory from experimental_recall_2.py
    from src._repo import REPO_ROOT
    base_dir = os.path.join(REPO_ROOT, "outputs", "pii_leakage", "experimental-recall-all-test")
    
    # Convert all parquet files in the directory
    convert_all_parquet_files(base_dir, overwrite=False)
