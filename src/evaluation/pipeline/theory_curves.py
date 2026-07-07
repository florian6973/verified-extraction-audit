"""Closed-form extraction curves vs attacker query budget N.

Single source of the recall / precision / population-FPR / extracted-stream
FPR / extracted-stream TPR curves. Both ``attack_curves.py`` (the authors'
plotting script) and the generic ``from_labels.py`` import these, so the theory
lives in exactly one place.

Each row is one candidate identifier with:
  - ``pi`` (per-draw extraction probability, e.g. exp of the finetuned LL of the
    name under the 'Name: ' prompt), read from ``pi_col``;
  - ``q`` (verification decision in {0,1}), read from ``q_col``.

Members are ``split == 'train'``, non-members ``split == 'val'``.
"""

import numpy as np


def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    """P(name i appears at least once) after N i.i.d. draws = 1 - (1 - pi)^N."""
    return 1.0 - np.power((1.0 - pi), N)


def compute_recall_precision_curves(df_all, df_mem, budgets, pi_col, q_col):
    """Recall(N) = (1/|M|) sum_{i in M} P(E_i;N) q_i; Precision over all rows in df_all."""
    pi_all = df_all[pi_col].to_numpy(dtype=float)
    q_all = df_all[q_col].to_numpy(dtype=float)
    pi_mem = df_mem[pi_col].to_numpy(dtype=float)
    q_mem = df_mem[q_col].to_numpy(dtype=float)

    recalls, precisions = [], []
    for N in budgets:
        pE_all = prob_extracted_at_least_once(pi_all, N)
        pE_mem = prob_extracted_at_least_once(pi_mem, N)
        tp = np.sum(pE_mem * q_mem)
        accepted = np.sum(pE_all * q_all)
        recalls.append(tp / len(df_mem))
        precisions.append((tp / accepted) if accepted > 0 else np.nan)

    mem_reachable = (pi_mem > 0).astype(float)
    all_reachable = (pi_all > 0).astype(float)
    recall_inf = np.sum(mem_reachable * q_mem) / len(df_mem)
    denom_inf = np.sum(all_reachable * q_all)
    precision_inf = (np.sum(mem_reachable * q_mem) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(recalls), np.array(precisions), float(recall_inf), float(precision_inf)


def compute_fpr_curve_population(df_nonmem, budgets, pi_col, q_col):
    """Population-average FPR: (1/|V|) sum_{i in V} P(E_i;N) q_i."""
    pi_nm = df_nonmem[pi_col].to_numpy(dtype=float)
    q_nm = df_nonmem[q_col].to_numpy(dtype=float)
    fprs = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        fprs.append(np.sum(pE_nm * q_nm) / len(df_nonmem))
    nm_reachable = (pi_nm > 0).astype(float)
    fpr_inf = np.sum(nm_reachable * q_nm) / len(df_nonmem)
    return np.array(fprs), float(fpr_inf)


def compute_fpr_curve_extracted(df_nonmem, budgets, pi_col, q_col):
    """Extracted-stream (selection-aware) FPR: sum_V P(E_i;N) q_i / sum_V P(E_i;N)."""
    pi_nm = df_nonmem[pi_col].to_numpy(dtype=float)
    q_nm = df_nonmem[q_col].to_numpy(dtype=float)
    fpr_ext = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        denom = np.sum(pE_nm)
        fpr_ext.append(np.sum(pE_nm * q_nm) / denom if denom > 0 else np.nan)
    nm_reachable = (pi_nm > 0).astype(float)
    denom_inf = np.sum(nm_reachable)
    fpr_ext_inf = (np.sum(nm_reachable * q_nm) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(fpr_ext), float(fpr_ext_inf)


def compute_tpr_curve_extracted(df_mem, budgets, pi_col, q_col):
    """Extracted-stream (selection-aware) TPR: sum_M P(E_i;N) q_i / sum_M P(E_i;N)."""
    pi_m = df_mem[pi_col].to_numpy(dtype=float)
    q_m = df_mem[q_col].to_numpy(dtype=float)
    tpr_ext = []
    for N in budgets:
        pE_m = prob_extracted_at_least_once(pi_m, N)
        denom = np.sum(pE_m)
        tpr_ext.append(np.sum(pE_m * q_m) / denom if denom > 0 else np.nan)
    m_reachable = (pi_m > 0).astype(float)
    denom_inf = np.sum(m_reachable)
    tpr_ext_inf = (np.sum(m_reachable * q_m) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(tpr_ext), float(tpr_ext_inf)
