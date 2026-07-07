"""Generic, dataset-agnostic extraction audit (scenario 2).

Composes the evaluation-pipeline pieces (verifier, extracted-stream FPR, fold
ensemble, theory curves) to audit a new dataset from a labeled ``(entry, label)``
set plus a base and a fine-tuned model. See :mod:`.from_labels`.
"""
