"""Ingest a minimal ``(subject_id, note)`` dataset into the pipeline layout.

The rest of the pipeline (PII injection, sampling, evaluation) consumes a
MIMIC-shaped layout: one ``splits_filtered_v<V>/<split>.parquet`` per split with
a ``text`` column, and a row-aligned ``splits_personas_v<V>/<split>.parquet``
with synthetic-persona identifiers. This module builds exactly that layout from
the single minimal Parquet described in the paper — a table of
``(subject_id, note)`` rows where removed direct-identifier spans are marked
``___`` — so **no MIMIC data, demographics, or hand layout are required**.

Splitting is done on ``subject_id`` (all notes for a subject stay on one side),
and personas are generated with :class:`FakePersonas` using demographics
synthesized deterministically from the subject id (so results are reproducible
without MIMIC's ``patients``/``admissions`` tables).

Example
-------
    python -m src.dataset.prepare.ingest \
        --input data/synthetic/notes.parquet \
        --name synthetic --out-root data/processed --version 8
"""

import argparse
import hashlib
import json
import os
import random

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.dataset.pii_insertion.fake_persona import FakePersonas

# Persona columns that FakePersonas produces and downstream code reads.
PERSONA_COLUMNS = [
    "subject_id", "name", "unit_no", "physician_name", "race", "language",
    "dob", "email", "phone", "ssn", "address", "random_name",
]

_RACES = ["white", "black", "hispanic", "asian"]
_GENDERS = ["M", "F"]


def _stable_int(value, salt=""):
    """A deterministic non-negative int from any value (reproducible across runs)."""
    h = hashlib.sha256((salt + "::" + str(value)).encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _synthesize_demographics(subject_id):
    """Deterministic age/gender/race/language/admittime for one subject.

    These only feed persona generation (faker locale + date of birth); their
    exact values do not matter for the audit, only that they are reproducible.
    """
    r = _stable_int(subject_id, "demo")
    age = 18 + (r % 73)                      # 18..90
    gender = _GENDERS[(r >> 3) % 2]
    race = _RACES[(r >> 5) % len(_RACES)]
    language = "English"
    anchor_year = 2110 + (r % 30)            # MIMIC uses shifted years
    admittime = f"{anchor_year}-{1 + (r >> 7) % 12:02d}-{1 + (r >> 11) % 28:02d} 12:00:00"
    return dict(anchor_age=age, gender=gender, race=race, language=language,
                anchor_year=anchor_year, admittime=admittime)


def split_on_subject(df, val_frac, seed):
    """Split notes into train/val so all notes of a subject stay on one side."""
    # .tolist() avoids passing a (possibly pyarrow-backed) extension array to
    # train_test_split, which can only integer-index plain arrays/lists.
    subjects = df["subject_id"].astype(str).unique().tolist()
    if len(subjects) < 2 or val_frac <= 0:
        return df.copy(), df.iloc[0:0].copy()
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=val_frac, random_state=seed
    )
    train_subjects, val_subjects = set(train_subjects), set(val_subjects)
    train_df = df[df["subject_id"].astype(str).isin(train_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].astype(str).isin(val_subjects)].reset_index(drop=True)
    return train_df, val_df


def build_personas_for_split(fp, notes_df, subject_common_id=None):
    """Return a per-note persona frame row-aligned with ``notes_df``.

    Personas are generated once per unique subject, then broadcast to every note
    of that subject (positional alignment with the filtered/text parquet is what
    the injection step relies on).
    """
    subjects = notes_df["subject_id"].astype(str).drop_duplicates().tolist()
    patients = pd.DataFrame(
        [dict(subject_id=s, **_synthesize_demographics(s)) for s in subjects]
    )
    # One physician per ~110 patients, matching the original ratio, min 1.
    fp.create_physicians(max(1, round(len(patients) / 110)))
    personas = fp.generate_personas(patients, subject_common_id)
    personas["subject_id"] = personas["subject_id"].astype(str)

    per_note = notes_df.copy()
    per_note["subject_id"] = per_note["subject_id"].astype(str)
    merged = per_note.merge(personas, on="subject_id", how="left", suffixes=("", "_persona"))
    return merged


def ingest(input_path, name, out_root, version, val_frac, seed, shared_canary):
    logger.info(f"Reading minimal dataset from {input_path}")
    df = pd.read_parquet(input_path)

    # Accept `note` (paper's column) or `text` (internal column).
    if "note" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"note": "text"})
    if "text" not in df.columns or "subject_id" not in df.columns:
        raise ValueError(
            f"Input {input_path} must have columns 'subject_id' and 'note' (or 'text'); "
            f"got {list(df.columns)}"
        )
    df["subject_id"] = df["subject_id"].astype(str)
    if "note_id" not in df.columns:
        df["note_id"] = [f"{name}_{i}" for i in range(len(df))]

    train_df, val_df = split_on_subject(df[["subject_id", "note_id", "text"]], val_frac, seed)
    logger.info(f"Split: {len(train_df)} train notes / {len(val_df)} val notes")

    filtered_dir = os.path.join(out_root, f"splits_filtered_v{version}")
    personas_dir = os.path.join(out_root, f"splits_personas_v{version}")
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(personas_dir, exist_ok=True)

    # A single FakePersonas instance across splits keeps names globally unique
    # (train/val disjoint), as persona_check.py expects.
    random.seed(seed)
    fp = FakePersonas(seed=seed)

    # Optionally seed a shared "member" canary (John Doe) present in the train
    # split — mirrors the paper's shared cross-dataset member.
    subject_common_id = None
    if shared_canary and len(train_df) > 0:
        subject_common_id = train_df["subject_id"].iloc[0]

    manifest = {"name": name, "version": version, "splits": {}}
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        if len(split_df) == 0:
            continue
        filtered = split_df.reset_index(drop=True)[["note_id", "subject_id", "text"]]
        personas = build_personas_for_split(fp, filtered, subject_common_id)
        # Keep persona parquet row-aligned with filtered parquet.
        persona_cols = [c for c in PERSONA_COLUMNS if c in personas.columns]
        personas_out = personas.reset_index(drop=True)[["note_id"] + persona_cols]

        filtered_path = os.path.join(filtered_dir, f"{split_name}.parquet")
        personas_path = os.path.join(personas_dir, f"{split_name}.parquet")
        filtered.to_parquet(filtered_path, index=False)
        personas_out.to_parquet(personas_path, index=False)
        logger.info(f"Wrote {filtered_path} and {personas_path} ({len(filtered)} rows)")
        manifest["splits"][split_name] = {
            "n_notes": int(len(filtered)),
            "n_subjects": int(filtered["subject_id"].nunique()),
            "filtered_path": filtered_path,
            "personas_path": personas_path,
        }

    manifest_path = os.path.join(out_root, f"{name}_ingest_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet with columns subject_id, note (or text)")
    parser.add_argument("--name", default="dataset", help="Dataset name (used in note ids and manifest)")
    parser.add_argument("--out-root", default=os.environ.get("DATA_ROOT", "data/processed"),
                        help="Root for splits_filtered_v*/splits_personas_v* (env DATA_ROOT)")
    parser.add_argument("--version", type=int, default=8, help="Layout version suffix (v<version>)")
    parser.add_argument("--val-frac", type=float, default=0.5, help="Fraction of subjects held out as val (non-members)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-canary", action="store_true",
                        help="Seed a shared 'John Doe' member canary in the train split")
    args = parser.parse_args()
    ingest(args.input, args.name, args.out_root, args.version, args.val_frac, args.seed, args.shared_canary)


if __name__ == "__main__":
    main()
