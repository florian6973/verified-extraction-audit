import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------
# User settings
# -----------------------
# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_0.1_3.csv"
CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_1.0_3.csv"
PI_COL = "p_ft_Name: "
SPLIT_COL = "split"         # "train" = member, "val" = non-member
SCORE_COL = "y_pred_proba"  # verifier score
BUDGETS = np.array([
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000,
    50000000, 100000000, 200000000, 500000000, 1000000000
], dtype=float)

plots_dir = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack-10-1.0-3"
os.makedirs(plots_dir, exist_ok=True)

# Targets for asymptotic FPR on val set
TARGET_A_FPRS = [0.01, 0.05]  # 1% and 5%

# -----------------------
# Load + clean
# -----------------------
df = pd.read_csv(CSV_PATH).copy()
df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df[PI_COL] = pd.to_numeric(df[PI_COL], errors="coerce")
df = df.dropna(subset=[SCORE_COL, PI_COL])

split_lower = df[SPLIT_COL].astype(str).str.lower()
members = df[split_lower == "train"].copy()
nonmembers = df[split_lower == "val"].copy()
if len(members) == 0:
    raise ValueError("No member rows found: expected split=='train' for members.")
if len(nonmembers) == 0:
    raise ValueError("No non-member rows found: expected split=='val' for non-members.")

print("Total pi mass in CSV:", df[PI_COL].sum())

# -----------------------
# Helpers
# -----------------------
def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    return 1.0 - np.power((1.0 - pi), N)

def pick_tau_for_asymptotic_fpr(nonmembers_df: pd.DataFrame, target_afpr: float) -> float:
    """
    Choose threshold tau so that asymptotic FPR on val is approximately target_afpr.

    As N->inf, for (almost all) pi>0, P(E_i)->1, so:
      aFPR ~= mean( 1[s_i >= tau] ) over val items.

    Therefore, tau is chosen as the (1 - target_afpr) quantile of non-member scores.
    """
    scores = nonmembers_df[SCORE_COL].to_numpy(dtype=float)

    # Edge handling
    if target_afpr <= 0:
        return np.inf  # accept none
    if target_afpr >= 1:
        return -np.inf # accept all

    # We want P(score >= tau) = target_afpr  -> tau is (1 - target_afpr) quantile
    tau = np.quantile(scores, 1.0 - target_afpr)
    return float(tau)

def compute_recall_fpr_curves(df_all: pd.DataFrame,
                              df_mem: pd.DataFrame,
                              df_nonmem: pd.DataFrame,
                              budgets: np.ndarray,
                              tau: float):
    # Deterministic accept rule
    q_all = (df_all[SCORE_COL].to_numpy(dtype=float) >= tau).astype(float)
    q_mem = (df_mem[SCORE_COL].to_numpy(dtype=float) >= tau).astype(float)
    q_nm  = (df_nonmem[SCORE_COL].to_numpy(dtype=float) >= tau).astype(float)

    pi_all = df_all[PI_COL].to_numpy(dtype=float)
    pi_mem = df_mem[PI_COL].to_numpy(dtype=float)
    pi_nm  = df_nonmem[PI_COL].to_numpy(dtype=float)

    recall = []
    fpr = []

    for N in budgets:
        pE_mem = prob_extracted_at_least_once(pi_mem, N)
        pE_nm  = prob_extracted_at_least_once(pi_nm, N)

        tp = np.sum(pE_mem * q_mem)
        fp = np.sum(pE_nm  * q_nm)

        recall.append(tp / len(df_mem))
        fpr.append(fp / len(df_nonmem))

    # Asymptotes
    mem_reachable = (pi_mem > 0).astype(float)
    nm_reachable  = (pi_nm  > 0).astype(float)
    recall_inf = np.sum(mem_reachable * q_mem) / len(df_mem)
    fpr_inf = np.sum(nm_reachable * q_nm) / len(df_nonmem)

    return np.array(recall), np.array(fpr), float(recall_inf), float(fpr_inf)

# -----------------------
# Compute curves for each target aFPR
# -----------------------
results = []
for target in TARGET_A_FPRS:
    tau = pick_tau_for_asymptotic_fpr(nonmembers, target)
    rec, fpr, rec_inf, fpr_inf = compute_recall_fpr_curves(df, members, nonmembers, BUDGETS, tau)
    results.append((target, tau, rec, fpr, rec_inf, fpr_inf))
    print(f"Target aFPR={target:.3%} -> tau={tau:.6f} -> achieved aFPR={fpr_inf:.3%}, asymptotic recall={rec_inf:.3%}")

# -----------------------
# Plot: Recall and FPR vs budget for the chosen taus
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for target, tau, rec, fpr, rec_inf, fpr_inf in results:
    # Recall
    line, = axes[0].plot(BUDGETS, rec, marker="o", label=f"aFPR≈{target:.0%} (tau={tau:.3f})")
    axes[0].axhline(rec_inf, linestyle="--", color=line.get_color())

    # FPR
    line2, = axes[1].plot(BUDGETS, fpr, marker="o", label=f"aFPR≈{target:.0%} (tau={tau:.3f})")
    axes[1].axhline(fpr_inf, linestyle="--", color=line2.get_color())

axes[0].set_xscale("log")
axes[0].set_xlabel("Budget N (number of 'Name:' draws)")
axes[0].set_ylabel("Expected recall over members (split=train)")
axes[0].set_title("Recall vs Budget at fixed asymptotic FPR")
axes[0].legend()

axes[1].set_xscale("log")
axes[1].set_xlabel("Budget N (number of 'Name:' draws)")
axes[1].set_ylabel("Expected FPR over non-members (split=val)")
axes[1].set_title("FPR vs Budget (should approach targets)")
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(plots_dir, "recall_fpr_vs_budget_tau_by_target_asymptotic_fpr.png")
plt.savefig(out_path, dpi=200)
plt.show()