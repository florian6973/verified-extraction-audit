# Paper-reproduction analysis scripts (legacy)

These scripts produced the figures, tables and bootstrap confidence intervals in
the paper. They are **not part of the live audit pipeline** — nothing under
`src/evaluation/audit/` or the smoke/MIMIC jobs imports them. They are kept here
for reference / reproducibility.

They were written for the original config-driven orchestration: each was invoked
as `python <script>.py --config <config-1B-*.yaml>` and reads/writes the
old-style output tree via `experimental/config_loader.py` +
`experimental/config_helper.py` (which remain in `../experimental/` because the
live verifier still uses them). The orchestrator itself (`run_pipeline_all.py`,
`run_pipeline.sh`) and its example configs (`config-1B-*.yaml`) were removed; to
run these again you would supply your own config and output paths.

Contents:

- `create_summary_plot.py`, `create_summary_table.py`, `summary_theo_plot.py`,
  `plot_fpr_tpr_budgets.py`, `plot_ll_distribution.py`, `fpr_tpr_theo.py` —
  paper figures/tables.
- `count_splits_and_pii_names.py`, `merge_all_names.py`, `evaluate_classifier.py`,
  `ner_ll_remaining.py` — dataset accounting, name merging, NER-based extraction.
- `mia/bootstrap_metrics.py`, `mia/compare_exp_theory.py`, `mia/evaluate_scores.py`,
  `mia/prep_data.py`, `mia/name_filter.py`, `mia/filter_names.py`,
  `mia/filter_last_names.py` (+ `.txt` gazetteers) — bootstrap CIs, experiment-vs-
  theory comparison, and name-filtering used by the paper's evaluation.

The current, dataset-agnostic equivalent of this whole flow is
`python -m src.evaluation.audit.from_labels` (see the top-level `README.md`),
which reuses the verifier pieces in `../experimental/` directly.
