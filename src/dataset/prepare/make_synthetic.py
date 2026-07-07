"""Generate a fully synthetic ``(subject_id, note)`` dataset to smoke-test the pipeline.

The notes are entirely fabricated clinical-style discharge snippets. Every
direct-identifier span is masked with ``___`` (the paper's input convention),
each preceded by a label (``Name:``, ``MRN:``, ``Attending:``, ``Address:``,
``Phone:``) so the offline injector can tell which direct identifier each blank
holds. No real patient data is involved.

Output: one Parquet with columns ``subject_id`` and ``note`` — exactly the
minimal interface described in the README. Feed it to
:mod:`src.dataset.prepare.ingest`.

Example
-------
    python -m src.dataset.prepare.make_synthetic \
        --out data/synthetic/notes.parquet --n-subjects 60 --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd

_COMPLAINTS = [
    "chest pain", "shortness of breath", "abdominal pain", "fever and chills",
    "blurred vision", "persistent cough", "dizziness", "lower back pain",
    "palpitations", "swelling in the legs", "headache", "nausea and vomiting",
]
_SERVICES = ["Emergency Department", "Cardiology", "General Medicine", "Neurology",
             "Pulmonology", "Gastroenterology", "Orthopedics"]
_SEXES = ["male", "female"]
_DISPOSITIONS = [
    "discharged home in stable condition",
    "admitted for further observation",
    "transferred to the step-down unit",
    "discharged with home health services",
]

# Each template embeds labeled ``___`` blanks. The label before each blank is
# what the offline injector uses to assign the direct-identifier type.
_TEMPLATES = [
    (
        "Name: ___\n"
        "MRN: ___\n"
        "Attending: ___\n\n"
        "{age}yo {sex} presenting with {complaint}, evaluated in the {service}. "
        "The patient was {disposition}. Address on file: ___. Contact phone: ___."
    ),
    (
        "Patient Name: ___  MRN: ___\n"
        "History: {age}-year-old {sex} with {complaint}. Seen by Attending: ___. "
        "Plan: {disposition}. Callback number Phone: ___."
    ),
    (
        "Discharge Summary\n"
        "Name: ___\nMRN: ___\n"
        "The {age}yo {sex} was admitted to {service} for {complaint} and {disposition}. "
        "Attending: ___. Mailing Address: ___."
    ),
]


def _fill_context(rng):
    return dict(
        age=rng.randint(19, 89),
        sex=rng.choice(_SEXES),
        complaint=rng.choice(_COMPLAINTS),
        service=rng.choice(_SERVICES),
        disposition=rng.choice(_DISPOSITIONS),
    )


def make_synthetic(n_subjects, max_notes_per_subject, seed):
    rng = random.Random(seed)
    rows = []
    for s in range(n_subjects):
        subject_id = f"S{s:05d}"
        n_notes = rng.randint(1, max_notes_per_subject)
        for _ in range(n_notes):
            template = rng.choice(_TEMPLATES)
            note = template.format(**_fill_context(rng))
            rows.append({"subject_id": subject_id, "note": note})
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["subject_id", "note"])


# --------------------------------------------------------------------------- #
# Scenario 2: notes that ALREADY contain the identifiers (no ___, no injection).
# Mimics a real, imperfectly de-identified corpus the user audits with a
# manually-built labeled set. Reuses di_types.detect_di_type to fill each blank.
# --------------------------------------------------------------------------- #
def _persona_values(fake, rng):
    return {
        "name-patient": f"{fake.first_name()} {fake.last_name()}",
        "name-attending": f"{fake.first_name()} {fake.last_name()}",
        "id": str(rng.randint(10000000, 99999999)),
        "address": fake.address().replace("\n", ", "),
        "phone": fake.numerify("(###) ###-####"),
        "email": fake.email(),
    }


def _fill_masked(masked_note, values, default_di):
    """Fill each ``___`` with the persona value for its detected category."""
    import re
    from src.dataset.prepare.di_types import detect_di_type
    offsets = [m.start() for m in re.finditer("___", masked_note)]
    segments = masked_note.split("___")
    out = segments[0]
    for k in range(1, len(segments)):
        cat = detect_di_type(masked_note[:offsets[k - 1]], default_di).category
        out += str(values.get(cat, "___"))
        out += segments[k]
    return out


def make_synthetic_scenario2(n_subjects, max_notes_per_subject, seed, val_frac=0.5):
    """Return (notes_df, labeled_df, train_notes, val_notes) with identifiers embedded."""
    from faker import Faker
    from src.dataset.prepare.di_types import get_di_type
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    default_di = get_di_type("name")

    subjects = [f"S{s:05d}" for s in range(n_subjects)]
    n_val = max(1, int(round(val_frac * n_subjects)))
    val_subjects = set(rng.sample(subjects, n_val))

    rows, train_names, val_names = [], set(), set()
    for subject_id in subjects:
        values = _persona_values(fake, rng)
        for _ in range(rng.randint(1, max_notes_per_subject)):
            masked = rng.choice(_TEMPLATES).format(**_fill_context(rng))
            note = _fill_masked(masked, values, default_di)
            rows.append({"subject_id": subject_id, "note": note})
        (val_names if subject_id in val_subjects else train_names).add(values["name-patient"])
    rng.shuffle(rows)
    notes_df = pd.DataFrame(rows, columns=["subject_id", "note"])

    # Labeled set: train patient names = members; val names + fresh distractors = non-members.
    distractors = {f"{fake.first_name()} {fake.last_name()}" for _ in range(len(train_names))}
    non_members = sorted((val_names | distractors) - train_names)
    labeled_df = pd.DataFrame(
        [{"entry": n, "label": 1} for n in sorted(train_names)]
        + [{"entry": n, "label": 0} for n in non_members])

    train_notes = [r for r in rows if r["subject_id"] not in val_subjects]
    val_notes = [r for r in rows if r["subject_id"] in val_subjects]
    return notes_df, labeled_df, train_notes, val_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic/notes.parquet",
                        help="Output Parquet path (columns subject_id, note)")
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=1,
                        help="1: masked notes with ___ (for injection); "
                             "2: notes with identifiers already embedded + labeled set + SFT")
    parser.add_argument("--n-subjects", type=int, default=60)
    parser.add_argument("--max-notes-per-subject", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.scenario == 1:
        df = make_synthetic(args.n_subjects, args.max_notes_per_subject, args.seed)
        df.to_parquet(args.out, index=False)
        n_blanks = int(df["note"].str.count("___").sum())
        print(f"[scenario 1] Wrote {len(df)} masked notes ({df['subject_id'].nunique()} subjects, "
              f"{n_blanks} ___ blanks) to {args.out}")
    else:
        notes_df, labeled_df, train_notes, val_notes = make_synthetic_scenario2(
            args.n_subjects, args.max_notes_per_subject, args.seed)
        notes_df.to_parquet(args.out, index=False)
        labeled_path = os.path.join(out_dir, "labeled.parquet")
        labeled_df.to_parquet(labeled_path, index=False)
        # SFT directly from the (already-identified) notes — no injection step.
        sft_dir = os.path.join(out_dir, "sft")
        os.makedirs(sft_dir, exist_ok=True)
        for split, notes in [("train", train_notes), ("val", val_notes)]:
            recs = [{"instruction": "Generate a clinical note", "output": r["note"],
                     "subject_id": r["subject_id"]} for r in notes]
            with open(os.path.join(sft_dir, f"{split}.json"), "w", encoding="utf-8") as f:
                json.dump(recs, f, indent=2, ensure_ascii=False)
        print(f"[scenario 2] Wrote {len(notes_df)} notes (identifiers embedded) to {args.out}; "
              f"labeled ({int(labeled_df['label'].sum())} members / "
              f"{int((labeled_df['label'] == 0).sum())} non-members) to {labeled_path}; "
              f"SFT to {sft_dir}/{{train,val}}.json")


if __name__ == "__main__":
    main()
