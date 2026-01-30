import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# User settings
# -----------------------
# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_0.1_3.csv"
CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_0.1_3.csv"
# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_1.0_3.csv"
TAU = 0.9
PI_COL = "p_ft_Name: "
SPLIT_COL = "split"         # "train" = member, "val" = non-member
SCORE_COL = "y_pred_proba"  # verifier score
BUDGETS = np.array([
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000,
    50000000, 100000000, 200000000, 500000000, 1000000000,
    2000000000, 5000000000, 10000000000, 20000000000, 50000000000,
    100000000000, 200000000000, 500000000000, 1000000000000, 2000000000000,
    5000000000000, 10000000000000,
], dtype=float)

plots_dir = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack"
os.makedirs(plots_dir, exist_ok=True)

# -----------------------
# Load + clean
# -----------------------
df = pd.read_csv(CSV_PATH).copy()
df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df[PI_COL] = pd.to_numeric(df[PI_COL], errors="coerce")

# Fill NA scores as requested: y_pred_proba := 1 - group
# (Assumes df['group'] exists and is 0/1)
df[SCORE_COL] = df[SCORE_COL].fillna(1 - df["group"])

# Add q columns FIRST
df["q"] = (df[SCORE_COL] >= TAU).astype(int)  # verifier decision
df["q_nover"] = 1                             # no verification baseline

print("Total pi mass in CSV:", df[PI_COL].sum())

split_lower = df[SPLIT_COL].astype(str).str.lower()
members = df[split_lower == "train"].copy()
nonmembers = df[split_lower == "val"].copy()

if len(members) == 0:
    raise ValueError("No member rows found: expected split=='train' for members.")
if len(nonmembers) == 0:
    raise ValueError("No non-member rows found: expected split=='val' for non-members.")

# -----------------------
# Helpers
# -----------------------
def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    return 1.0 - np.power((1.0 - pi), N)

def compute_recall_precision_curves(df_all: pd.DataFrame,
                                   df_mem: pd.DataFrame,
                                   budgets: np.ndarray,
                                   q_col: str):
    pi_all = df_all[PI_COL].to_numpy(dtype=float)
    q_all  = df_all[q_col].to_numpy(dtype=float)

    pi_mem = df_mem[PI_COL].to_numpy(dtype=float)
    q_mem  = df_mem[q_col].to_numpy(dtype=float)

    recalls = []
    precisions = []

    for N in budgets:
        pE_all = prob_extracted_at_least_once(pi_all, N)
        pE_mem = prob_extracted_at_least_once(pi_mem, N)

        tp = np.sum(pE_mem * q_mem)
        accepted = np.sum(pE_all * q_all)

        recall = tp / len(df_mem)
        precision = (tp / accepted) if accepted > 0 else np.nan

        recalls.append(recall)
        precisions.append(precision)

    # Asymptotic N->inf
    mem_reachable = (pi_mem > 0).astype(float)
    all_reachable = (pi_all > 0).astype(float)

    recall_inf = np.sum(mem_reachable * q_mem) / len(df_mem)
    denom_inf = np.sum(all_reachable * q_all)
    precision_inf = (np.sum(mem_reachable * q_mem) / denom_inf) if denom_inf > 0 else np.nan

    return np.array(recalls), np.array(precisions), float(recall_inf), float(precision_inf)

def compute_fpr_curve(df_nonmem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """Population-average pipeline FPR (divide by |V|)."""
    pi_nm = df_nonmem[PI_COL].to_numpy(dtype=float)
    q_nm  = df_nonmem[q_col].to_numpy(dtype=float)

    fprs = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        fp = np.sum(pE_nm * q_nm)
        fprs.append(fp / len(df_nonmem))

    nm_reachable = (pi_nm > 0).astype(float)
    fpr_inf = np.sum(nm_reachable * q_nm) / len(df_nonmem)
    return np.array(fprs), float(fpr_inf)

def compute_fpr_extracted_curve(df_nonmem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Selection-aware FPR among extracted non-members:
      sum_i P(E_i;N) q_i / sum_i P(E_i;N)
    """
    pi_nm = df_nonmem[PI_COL].to_numpy(dtype=float)
    q_nm  = df_nonmem[q_col].to_numpy(dtype=float)

    fpr_ext = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        denom = np.sum(pE_nm)
        num = np.sum(pE_nm * q_nm)
        fpr_ext.append(num / denom if denom > 0 else np.nan)

    # As N->inf and if pi>0 for all, this tends to mean(q_nm)
    nm_reachable = (pi_nm > 0).astype(float)
    denom_inf = np.sum(nm_reachable)  # ~ len(nonmembers) if all pi>0
    fpr_ext_inf = (np.sum(nm_reachable * q_nm) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(fpr_ext), float(fpr_ext_inf)

# -----------------------
# Compute curves
# -----------------------
rec_v, prec_v, recinf_v, precinf_v = compute_recall_precision_curves(df, members, BUDGETS, q_col="q")
rec_nv, prec_nv, recinf_nv, precinf_nv = compute_recall_precision_curves(df, members, BUDGETS, q_col="q_nover")

fpr_v, fprinf_v = compute_fpr_curve(nonmembers, BUDGETS, q_col="q")
fpr_nv, fprinf_nv = compute_fpr_curve(nonmembers, BUDGETS, q_col="q_nover")

fprx_v, fprxinf_v = compute_fpr_extracted_curve(nonmembers, BUDGETS, q_col="q")
fprx_nv, fprxinf_nv = compute_fpr_extracted_curve(nonmembers, BUDGETS, q_col="q_nover")

print(f"[Verifier] asymptotic recall: {recinf_v:.6f}   asymptotic precision: {precinf_v:.6f}")
print(f"[Verifier] asymptotic FPR(pop): {fprinf_v:.6f}   asymptotic FPR(extracted): {fprxinf_v:.6f}")
print(f"[No ver ] asymptotic recall: {recinf_nv:.6f}   asymptotic precision: {precinf_nv:.6f}")
print(f"[No ver ] asymptotic FPR(pop): {fprinf_nv:.6f}   asymptotic FPR(extracted): {fprxinf_nv:.6f}")

# -----------------------
# Plot: Recall + FPR(pop) and FPR(extracted)
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Recall
line_rec_v, = axes[0].plot(BUDGETS, rec_v, marker="o", label=f"Recall w/ verification (tau={TAU})")
axes[0].axhline(recinf_v, linestyle="--", color=line_rec_v.get_color(), label="Asymptote w/ verification")

line_rec_nv, = axes[0].plot(BUDGETS, rec_nv, marker="o", label="Recall w/o verification (q=1)")
axes[0].axhline(recinf_nv, linestyle="--", color=line_rec_nv.get_color(), label="Asymptote w/o verification")

axes[0].set_xscale("log")
axes[0].set_xlabel("Budget N (number of 'Name:' draws)")
axes[0].set_ylabel("Expected recall over members (split=train)")
axes[0].set_title("Recall vs Budget")
axes[0].legend()

# Right: FPR curves
# line_fpr_v, = axes[1].plot(BUDGETS, fpr_v, marker="o", label=f"FPR(pop) w/ verification (tau={TAU})")
# axes[1].axhline(fprinf_v, linestyle="--", color=line_fpr_v.get_color(), label="Asymptote FPR(pop) w/ verification")

line_fprx_v, = axes[1].plot(BUDGETS, fprx_v, marker="o", label=f"FPR(extracted) w/ verification (tau={TAU})")
axes[1].axhline(fprxinf_v, linestyle="--", color=line_fprx_v.get_color(), label="Asymptote FPR(extracted) w/ verification")

# line_fpr_nv, = axes[1].plot(BUDGETS, fpr_nv, marker="o", label="FPR(pop) w/o verification (q=1)")
# axes[1].axhline(fprinf_nv, linestyle="--", color=line_fpr_nv.get_color(), label="Asymptote FPR(pop) w/o verification")

line_fprx_nv, = axes[1].plot(BUDGETS, fprx_nv, marker="o", label="FPR(extracted) w/o verification (q=1)")
axes[1].axhline(fprxinf_nv, linestyle="--", color=line_fprx_nv.get_color(), label="Asymptote FPR(extracted) w/o verification")

# axes[0].set_ylim(0.0, 1.05)
axes[1].set_ylim(0.0, 1.05)

axes[1].set_xscale("log")
axes[1].set_xlabel("Budget N (number of 'Name:' draws)")
axes[1].set_ylabel("False positive rate on non-members (split=val)")
axes[1].set_title("FPR vs Budget (population vs extracted)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"recall_fpr_fprExtracted_vs_budget_tau_{TAU}.png"), dpi=200)
plt.show()

# -----------------------
# Optional: precision plot
# -----------------------
plt.figure(figsize=(7, 4))
plt.plot(BUDGETS, prec_v, marker="o", label=f"Precision w/ verification (tau={TAU})")
if not np.isnan(precinf_v):
    plt.axhline(precinf_v, linestyle="--", label="Asymptote precision w/ verification")
plt.plot(BUDGETS, prec_nv, marker="o", label="Precision w/o verification (q=1)")
if not np.isnan(precinf_nv):
    plt.axhline(precinf_nv, linestyle="--", label="Asymptote precision w/o verification")
plt.xscale("log")
plt.xlabel("Budget N (number of 'Name:' draws)")
plt.ylabel("Expected precision (over names in CSV)")
plt.title("Precision vs Budget")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"precision_vs_budget_with_without_ver_tau_{TAU}.png"), dpi=200)
plt.show()
