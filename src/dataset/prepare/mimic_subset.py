"""Subsample MIMIC-IV discharge notes into the minimal ``(subject_id, note)`` Parquet.

MIMIC-IV discharge notes already contain ``___`` where identifiers were removed,
so they feed straight into the pipeline via :mod:`src.dataset.prepare.ingest` —
no ``admissions.csv`` / ``patients.csv`` and no full ``fake_persona`` build. This
keeps a subject-level fraction (e.g. 1% for a quick test) so all notes of a
subject stay together, and preserves the real ``note_id``.

Example
-------
    python -m src.dataset.prepare.mimic_subset \
        --discharge data/raw/discharge.csv --out data/mimic_1pct.parquet --frac 0.01
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discharge", default="data/raw/discharge.csv",
                        help="MIMIC-IV discharge.csv")
    parser.add_argument("--out", default="data/mimic_subset.parquet")
    parser.add_argument("--frac", type=float, default=0.01, help="Fraction of SUBJECTS to keep")
    parser.add_argument("--text-col", default="text", help="Column holding the note text")
    parser.add_argument("--max-notes", type=int, default=None, help="Optional hard cap on # notes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    keep_cols = {"subject_id", "note_id", args.text_col}
    df = pd.read_csv(args.discharge, usecols=lambda c: c in keep_cols)
    if args.text_col not in df.columns or "subject_id" not in df.columns:
        raise ValueError(f"{args.discharge} must have 'subject_id' and '{args.text_col}'; "
                         f"got {list(df.columns)}")
    df = df.rename(columns={args.text_col: "note"})

    subjects = df["subject_id"].drop_duplicates()
    keep = subjects.sample(frac=args.frac, random_state=args.seed)
    out = df[df["subject_id"].isin(keep)].reset_index(drop=True)
    if args.max_notes:
        out = out.head(args.max_notes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_blanks = int(out["note"].astype(str).str.count("___").sum())
    print(f"Wrote {len(out)} notes from {out['subject_id'].nunique()} subjects "
          f"({args.frac:.2%} of {subjects.size}), {n_blanks} ___ blanks -> {args.out}")


if __name__ == "__main__":
    main()
