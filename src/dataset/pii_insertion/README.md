# Synthetic-persona utilities

Helpers for generating and checking the synthetic personas used to fill the
`___` blanks. **Direct-identifier injection and sampling now live in one place**,
`src/dataset/prepare/inject.py` (see the repo README, Step 2) — this folder only
provides the persona generation it consumes.

- `fake_persona.py` — the `FakePersonas` class (patient/physician names, MRN,
  address, phone, email, …). Run as a script to build the MIMIC personas
  (`splits_filtered_v* + splits_personas_v*` from the MIMIC splits):

  ```bash
  python src/dataset/pii_insertion/fake_persona.py
  ```

  For a non-MIMIC dataset, `src/dataset/prepare/ingest.py` generates personas
  directly from a `(subject_id, note)` Parquet — no demographics tables needed.

- `build_name_filter_list.py` — build first/last-name gazetteers from `faker`
  locales (used to filter non-name hallucinations during evaluation).
- `persona_check.py` — QA on the generated personas (train/val disjointness,
  name duplication, canary flags).
