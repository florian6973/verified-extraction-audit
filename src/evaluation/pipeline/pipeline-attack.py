import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# -----------------------
# User settings
# -----------------------
CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_0.1_3.csv"   # change to your path
TAU = 0.5                   # verifier threshold
PI_COL = "p_ft_Name: "      # per-attempt generation prob for prompt "Name: "
SPLIT_COL = "split"         # "train" = member, "val" = non-member
SCORE_COL = "y_pred_proba"  # classifier score
BUDGETS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000], dtype=float)
plots_dir = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack"
os.makedirs(plots_dir, exist_ok=True)
# -----------------------
# Load + clean
# -----------------------
df = pd.read_csv(CSV_PATH).copy()

# Drop NA scores (as requested)
df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df = df.dropna(subset=[SCORE_COL])

# Ensure pi is numeric + drop rows where missing
df[PI_COL] = pd.to_numeric(df[PI_COL], errors="coerce")
df = df.dropna(subset=[PI_COL])

# Deterministic verifier decision q_i at fixed tau
df["q"] = (df[SCORE_COL] >= TAU).astype(int)

print("Total mass", df[PI_COL].sum())

# Members/non-members by split
members = df[df[SPLIT_COL].astype(str).str.lower() == "train"].copy()
nonmembers = df[df[SPLIT_COL].astype(str).str.lower() == "val"].copy()

if len(members) == 0:
    raise ValueError("No member rows found: expected split=='train' for members.")
if len(nonmembers) == 0:
    raise ValueError("No non-member rows found: expected split=='val' for non-members.")

# -----------------------
# Helpers
# -----------------------
def prob_extracted_at_least_once(pi: np.ndarray, N: float) -> np.ndarray:
    """Given per-attempt probability pi, return P(extracted at least once) after N attempts."""
    # works fine for small pi; for extreme values you could use expm1/log1p tricks
    return 1.0 - np.power((1.0 - pi), N)

def compute_curves(df_all: pd.DataFrame, df_mem: pd.DataFrame, budgets: np.ndarray):
    pi_all = df_all[PI_COL].to_numpy(dtype=float)
    q_all  = df_all["q"].to_numpy(dtype=float)

    pi_mem = df_mem[PI_COL].to_numpy(dtype=float)
    q_mem  = df_mem["q"].to_numpy(dtype=float)

    recalls = []
    precisions = []

    for N in budgets:
        pE_all = prob_extracted_at_least_once(pi_all, N)
        pE_mem = prob_extracted_at_least_once(pi_mem, N)

        # Expected TP = sum over member items of P(extracted)*q
        tp = np.sum(pE_mem * q_mem)

        # Expected accepted outputs (TP + FP) = sum over all items of P(extracted)*q
        accepted = np.sum(pE_all * q_all)

        # End-to-end recall = expected TP / number of true members
        recall = tp / len(df_mem)

        # End-to-end precision = expected TP / expected accepted
        precision = (tp / accepted) if accepted > 0 else np.nan

        recalls.append(recall)
        precisions.append(precision)

    # Asymptotic N->inf values: P(extracted) -> 1 if pi>0 else 0
    mem_reachable = (pi_mem > 0).astype(float)
    all_reachable = (pi_all > 0).astype(float)

    recall_inf = np.sum(mem_reachable * q_mem) / len(df_mem)
    denom_inf = np.sum(all_reachable * q_all)
    precision_inf = (np.sum(mem_reachable * q_mem) / denom_inf) if denom_inf > 0 else np.nan

    return np.array(recalls), np.array(precisions), recall_inf, precision_inf

# -----------------------
# Compute curves
# -----------------------
rec_curve, prec_curve, rec_inf, prec_inf = compute_curves(df, members, BUDGETS)

print(f"Asymptotic recall (N->inf) at tau={TAU}: {rec_inf:.6f}")
print(f"Asymptotic precision (N->inf) at tau={TAU}: {prec_inf:.6f}")

# -----------------------
# Plot recall vs budget
# -----------------------
plt.figure()
plt.plot(BUDGETS, rec_curve, marker="o")
plt.xscale("log")
plt.xlabel("Budget N (number of 'Name:' draws)")
plt.ylabel(f"End-to-end recall (tau={TAU})")
plt.axhline(rec_inf, linestyle="--", label="Asymptote")
plt.title("Recall vs Budget (with asymptote)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "recall_vs_budget.png"), dpi=200)
plt.show()

# -----------------------
# Plot precision vs budget
# -----------------------
plt.figure()
plt.plot(BUDGETS, prec_curve, marker="o")
plt.xscale("log")
plt.xlabel("Budget N (number of 'Name:' draws)")
plt.ylabel(f"End-to-end precision (tau={TAU})")
if not np.isnan(prec_inf):
    plt.axhline(prec_inf, linestyle="--", label="Asymptote")
plt.title("Precision vs Budget (with asymptote)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "precision_vs_budget.png"), dpi=200)
plt.show()