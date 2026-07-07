#!/usr/bin/env python3
from src._repo import REPO_ROOT
"""
Count notes and patients per split (train/val/test) from splits_filtered_v12,
and count distinct names per PII rate in the ll_all_output CSV.
"""

import argparse
import os

import pandas as pd


SPLITS_DIR = REPO_ROOT + "/data/processed/splits_filtered_v12"
PII_CSV_PATH = (
    REPO_ROOT + "/outputs/pii_leakage/pipeline/"
    "ll_all_output_False_1B_100_batch.csv"
)


def count_splits(splits_dir: str) -> pd.DataFrame:
    """Load train, val, test parquet and return n_notes, n_patients per split."""
    results = []
    for split_name in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{split_name}.parquet")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        df = pd.read_parquet(path)
        n_notes = df["note_id"].nunique() if "note_id" in df.columns else len(df)
        n_patients = df["subject_id"].nunique() if "subject_id" in df.columns else pd.NA
        results.append(
            {
                "split": split_name,
                "n_notes": n_notes,
                "n_patients": n_patients,
            }
        )
    return pd.DataFrame(results)


def count_names_per_pii_rate(csv_path: str) -> pd.DataFrame:
    """Load ll batch CSV and count distinct names (value) per (pii_rate, split)."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "value" not in df.columns:
        raise ValueError(f"CSV must have a 'value' column (names). Columns: {list(df.columns)}")
    if "pii_rate" not in df.columns:
        raise ValueError(f"CSV must have a 'pii_rate' column. Columns: {list(df.columns)}")
    if "split" not in df.columns:
        raise ValueError(f"CSV must have a 'split' column. Columns: {list(df.columns)}")
    out = (
        df.groupby(["pii_rate", "split"], as_index=False)["value"]
        .nunique()
        .rename(columns={"value": "n_distinct_names"})
    )
    return out


def _fmt_int(x) -> str:
    """Format integer with commas."""
    if pd.isna(x):
        return "---"
    return f"{int(x):,}"


def to_latex_splits(df: pd.DataFrame, caption: str = "Dataset splits for MIMIC-IV-Note.", label: str = "tab:mimic_splits") -> str:
    """Format splits DataFrame as a LaTeX table (booktabs style)."""
    split_display = {"train": "Train", "val": "Val", "test": "Test"}
    rows = []
    for _, row in df.iterrows():
        split_name = split_display.get(row["split"], row["split"].capitalize())
        rows.append(f"{split_name} & {_fmt_int(row['n_notes'])} & {_fmt_int(row['n_patients'])} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lrr}}
\\toprule
\\textbf{{Split}} & \\textbf{{\\# Notes}} & \\textbf{{\\# Patients}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _pii_rate_to_pct(pii) -> str:
    """Format PII rate as percentage (e.g. 0.01 -> 1%)."""
    if isinstance(pii, float):
        return f"{int(round(pii * 100))}\\%"
    return str(pii)


def to_latex_pii_names(df: pd.DataFrame, caption: str = "Distinct names per PII rate and split.", label: str = "tab:pii_names") -> str:
    """Format PII names DataFrame as a LaTeX table (booktabs style).
    PII rate shown as percentage (e.g. 1%). If all val rows have the same n_distinct_names, collapse to one row with PII rate 'all'.
    """
    split_display = {"train": "Train", "val": "Val", "test": "Test"}
    rows = []

    for split_key in ["train", "val", "test"]:
        subset = df[df["split"].astype(str).str.lower() == split_key]
        if subset.empty:
            continue
        split_name = split_display.get(split_key, split_key.capitalize())

        if split_key == "val" and subset["n_distinct_names"].nunique() == 1:
            # All val rows have the same value: one row with 'all'
            n = subset["n_distinct_names"].iloc[0]
            rows.append(f"all & {split_name} & {_fmt_int(n)} \\\\")
        else:
            for _, row in subset.iterrows():
                pii_str = _pii_rate_to_pct(row["pii_rate"])
                rows.append(f"{pii_str} & {split_name} & {_fmt_int(row['n_distinct_names'])} \\\\")

    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{llr}}
\\toprule
\\textbf{{PII rate}} & \\textbf{{Split}} & \\textbf{{\\# Distinct names}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def main():
    parser = argparse.ArgumentParser(description="Count notes/patients per split and names per PII rate.")
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=SPLITS_DIR,
        help="Directory containing train.parquet, val.parquet, test.parquet",
    )
    parser.add_argument(
        "--pii_csv",
        type=str,
        default=PII_CSV_PATH,
        help="Path to ll_all_output_True_1B_1_batch.csv",
    )
    parser.add_argument(
        "--out_splits_tex",
        type=str,
        default=None,
        help="Path to save splits LaTeX table (e.g. tab_mimic_splits.tex).",
    )
    parser.add_argument(
        "--out_pii_tex",
        type=str,
        default=None,
        help="Path to save PII names LaTeX table (e.g. tab_pii_names.tex).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Splits (notes & patients)")
    print("=" * 60)
    df_splits = count_splits(args.splits_dir)
    if df_splits.empty:
        print("No split files found.")
    else:
        print(df_splits.to_string(index=False))
        if args.out_splits_tex:
            latex = to_latex_splits(df_splits)
            with open(args.out_splits_tex, "w") as f:
                f.write(latex)
            print(f"Saved LaTeX table to {args.out_splits_tex}")
        print()

    print("=" * 60)
    print("Distinct names per PII rate and split (ll_all_output CSV)")
    print("=" * 60)
    if not os.path.exists(args.pii_csv):
        print(f"Warning: {args.pii_csv} not found.")
    else:
        df_pii = count_names_per_pii_rate(args.pii_csv)
        print(df_pii.to_string(index=False))
        if args.out_pii_tex:
            latex = to_latex_pii_names(df_pii)
            with open(args.out_pii_tex, "w") as f:
                f.write(latex)
            print(f"Saved LaTeX table to {args.out_pii_tex}")


if __name__ == "__main__":
    main()
