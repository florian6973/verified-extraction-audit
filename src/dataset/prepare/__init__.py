"""Minimal-interface tooling for applying the audit to a new dataset.

These modules let you run the pipeline from a single Parquet of ``(subject_id,
note)`` rows (with removed direct-identifier spans marked ``___``), instead of
the MIMIC-specific splits/personas layout. See the repository README, section
"Applying our framework to a new dataset".
"""
