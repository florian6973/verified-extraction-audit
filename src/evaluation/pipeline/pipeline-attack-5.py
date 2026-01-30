import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# -----------------------
# User settings
# -----------------------
# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-calibration/df_temp_sub_name-patient_1B_10_0.1_3.csv"
# TAU = 0.7
PI_COL = "p_ft_Name: "
SPLIT_COL = "split"         # "train" = member, "val" = non-member
SCORE_COL = "y_pred_proba"  # verifier score

import argparse
from src.evaluation.pipeline.experimental.config_loader import load_config
from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir

parser = argparse.ArgumentParser(description='Evaluate scores for MIA')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--tau', type=float, default=0.7, help='Verifier threshold')
args = parser.parse_args()

config = load_config(args.config)

TAU = args.tau

model = config['filters']['model']
dataset_size = config['filters']['dataset_size']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']

# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/mia-verifier/scores_1B_10_pii_rate_0.1_n_epochs_3_p.csv"
CSV_PATH = os.path.join(get_output_dir(config), f"scores_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}_p.csv")
PI_COL = "p_ft_Name: "
SPLIT_COL = "split_x"         # "train" = member, "val" = non-member
SCORE_COL = "score_oof_member_proba"  # verifier score

# CSV_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-0.1/all_names_ll_computed_with_scores.csv"
# TAU = 0.5
# column_name = "score_oof_member_proba"
# groundtruth_column = "groundtruth"

# import pandas as pd
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# df = pd.read_csv(data_path)
# df['y_true'] = (df[groundtruth_column] == 'train').astype(int)




BUDGETS = np.array([
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000,
    50000000, 100000000, 200000000, 500000000, 1000000000,
    2000000000, 5000000000, 10000000000, 20000000000, 50000000000,
    100000000000, 200000000000, 500000000000, 1000000000000, 2000000000000,
    5000000000000, 10000000000000,
], dtype=float)

# plots_dir = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/plots/pipeline-attack-5"
plots_dir = os.path.join(get_output_dir(config), "plots_theory")
os.makedirs(plots_dir, exist_ok=True)

# -----------------------
# Load + clean
# -----------------------
df = pd.read_csv(CSV_PATH).copy()

# Coerce types
df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df[PI_COL] = pd.to_numeric(df[PI_COL], errors="coerce")

# If y_pred_proba is missing: set it to 1 - group (as you requested)
# (Assumes df['group'] exists and is 0/1.)
if "group" in df.columns:
    if "group" not in df.columns:
        raise ValueError("Column 'group' not found but required to fill missing scores with 1 - group.")
    df[SCORE_COL] = df[SCORE_COL].fillna(1 - df["group"])
else:
    warnings.warn("Column 'group' not found. Will not fill missing scores with 1 - group.")

# Optional: drop rows with missing pi (usually safest)
df = df.dropna(subset=[PI_COL])

# Add decisions first so slices contain them
df["q"] = (df[SCORE_COL] >= TAU).astype(int)  # verification at TAU
df["q_nover"] = 1                             # no verification baseline

print("Total pi mass in CSV:", df[PI_COL].sum())

# Split members/nonmembers after q columns exist
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
    """P(name i appears at least once) after N i.i.d. draws."""
    return 1.0 - np.power((1.0 - pi), N)

def compute_recall_precision_curves(df_all: pd.DataFrame,
                                   df_mem: pd.DataFrame,
                                   budgets: np.ndarray,
                                   q_col: str):
    """
    Recall(N) = (1/|M|) sum_{i in M} P(E_i;N) q_i
    Precision(N) = TP / Accepted where Accepted sums over ALL rows in df_all.
    """
    pi_all = df_all[PI_COL].to_numpy(dtype=float)
    q_all  = df_all[q_col].to_numpy(dtype=float)

    pi_mem = df_mem[PI_COL].to_numpy(dtype=float)
    q_mem  = df_mem[q_col].to_numpy(dtype=float)

    recalls, precisions = [], []

    for N in budgets:
        pE_all = prob_extracted_at_least_once(pi_all, N)
        pE_mem = prob_extracted_at_least_once(pi_mem, N)

        tp = np.sum(pE_mem * q_mem)
        accepted = np.sum(pE_all * q_all)

        recall = tp / len(df_mem)
        precision = (tp / accepted) if accepted > 0 else np.nan

        recalls.append(recall)
        precisions.append(precision)

    # Asymptotic N->inf (if pi>0, pE -> 1)
    mem_reachable = (pi_mem > 0).astype(float)
    all_reachable = (pi_all > 0).astype(float)

    recall_inf = np.sum(mem_reachable * q_mem) / len(df_mem)
    denom_inf = np.sum(all_reachable * q_all)
    precision_inf = (np.sum(mem_reachable * q_mem) / denom_inf) if denom_inf > 0 else np.nan

    return np.array(recalls), np.array(precisions), float(recall_inf), float(precision_inf)

def compute_fpr_curve_population(df_nonmem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Population-average pipeline FPR:
      (1/|V|) sum_{i in V} P(E_i;N) q_i
    """
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

def compute_fpr_curve_extracted(df_nonmem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Extracted-stream FPR (selection-aware):
      sum_{i in V} P(E_i;N) q_i  /  sum_{i in V} P(E_i;N)
    """
    pi_nm = df_nonmem[PI_COL].to_numpy(dtype=float)
    q_nm  = df_nonmem[q_col].to_numpy(dtype=float)

    fpr_ext = []
    for N in budgets:
        pE_nm = prob_extracted_at_least_once(pi_nm, N)
        denom = np.sum(pE_nm)
        num = np.sum(pE_nm * q_nm)
        fpr_ext.append(num / denom if denom > 0 else np.nan)

    # Asymptote: if pi>0 for all, tends to mean(q_nm)
    nm_reachable = (pi_nm > 0).astype(float)
    denom_inf = np.sum(nm_reachable)
    fpr_ext_inf = (np.sum(nm_reachable * q_nm) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(fpr_ext), float(fpr_ext_inf)

def compute_tpr_curve_extracted(df_mem: pd.DataFrame, budgets: np.ndarray, q_col: str):
    """
    Extracted-stream TPR (selection-aware):
      sum_{i in M} P(E_i;N) q_i  /  sum_{i in M} P(E_i;N)
    """
    pi_m = df_mem[PI_COL].to_numpy(dtype=float)
    q_m  = df_mem[q_col].to_numpy(dtype=float)

    tpr_ext = []
    for N in budgets:
        pE_m = prob_extracted_at_least_once(pi_m, N)
        denom = np.sum(pE_m)
        num = np.sum(pE_m * q_m)
        tpr_ext.append(num / denom if denom > 0 else np.nan)

    m_reachable = (pi_m > 0).astype(float)
    denom_inf = np.sum(m_reachable)
    tpr_ext_inf = (np.sum(m_reachable * q_m) / denom_inf) if denom_inf > 0 else np.nan
    return np.array(tpr_ext), float(tpr_ext_inf)

# -----------------------
# Compute curves
# -----------------------
rec_v, prec_v, recinf_v, precinf_v = compute_recall_precision_curves(df, members, BUDGETS, q_col="q")
rec_nv, prec_nv, recinf_nv, precinf_nv = compute_recall_precision_curves(df, members, BUDGETS, q_col="q_nover")

fpr_pop_v, fpr_pop_inf_v = compute_fpr_curve_population(nonmembers, BUDGETS, q_col="q")
fpr_pop_nv, fpr_pop_inf_nv = compute_fpr_curve_population(nonmembers, BUDGETS, q_col="q_nover")

fpr_ext_v, fpr_ext_inf_v = compute_fpr_curve_extracted(nonmembers, BUDGETS, q_col="q")
fpr_ext_nv, fpr_ext_inf_nv = compute_fpr_curve_extracted(nonmembers, BUDGETS, q_col="q_nover")

tpr_ext_v, tpr_ext_inf_v = compute_tpr_curve_extracted(members, BUDGETS, q_col="q")
tpr_ext_nv, tpr_ext_inf_nv = compute_tpr_curve_extracted(members, BUDGETS, q_col="q_nover")

print(f"[Verifier] asymptotic recall: {recinf_v:.6f}   asymptotic precision: {precinf_v:.6f}")
print(f"[Verifier] asymptotic FPR(pop): {fpr_pop_inf_v:.6f}   asymptotic FPR(extracted): {fpr_ext_inf_v:.6f}")
print(f"[Verifier] asymptotic TPR(extracted): {tpr_ext_inf_v:.6f}")
print(f"[No ver ] asymptotic recall: {recinf_nv:.6f}   asymptotic precision: {precinf_nv:.6f}")
print(f"[No ver ] asymptotic FPR(pop): {fpr_pop_inf_nv:.6f}   asymptotic FPR(extracted): {fpr_ext_inf_nv:.6f}")
print(f"[No ver ] asymptotic TPR(extracted): {tpr_ext_inf_nv:.6f}")

# -----------------------
# Display theoretical values for budget = 10^4
# -----------------------
target_budget = 1e4  # 10^4
budget_idx = np.argmin(np.abs(BUDGETS - target_budget))
actual_budget = BUDGETS[budget_idx]

print("\n" + "="*80)
print(f"THEORETICAL VALUES FOR BUDGET = {target_budget:.0f} (actual: {actual_budget:.0f})")
print("="*80)
print(f"\nWith Verification (tau={TAU}):")
print(f"  Recall:              {rec_v[budget_idx]:.6f}")
print(f"  Precision:           {prec_v[budget_idx]:.6f}")
print(f"  FPR (population):    {fpr_pop_v[budget_idx]:.6f}")
print(f"  FPR (extracted):     {fpr_ext_v[budget_idx]:.6f}")
print(f"  TPR (extracted):     {tpr_ext_v[budget_idx]:.6f}")

print(f"\nWithout Verification (q=1):")
print(f"  Recall:              {rec_nv[budget_idx]:.6f}")
print(f"  Precision:           {prec_nv[budget_idx]:.6f}")
print(f"  FPR (population):    {fpr_pop_nv[budget_idx]:.6f}")
print(f"  FPR (extracted):     {fpr_ext_nv[budget_idx]:.6f}")
print(f"  TPR (extracted):     {tpr_ext_nv[budget_idx]:.6f}")
print("="*80 + "\n")

# -----------------------
# Save theoretical curves to CSV
# -----------------------
df_curves = pd.DataFrame({
    'budget': BUDGETS,
    'recall_with_verification': rec_v,
    'recall_without_verification': rec_nv,
    'precision_with_verification': prec_v,
    'precision_without_verification': prec_nv,
    'fpr_population_with_verification': fpr_pop_v,
    'fpr_population_without_verification': fpr_pop_nv,
    'fpr_extracted_with_verification': fpr_ext_v,
    'fpr_extracted_without_verification': fpr_ext_nv,
    'tpr_extracted_with_verification': tpr_ext_v,
    'tpr_extracted_without_verification': tpr_ext_nv,
})

# Add asymptotic values as a row with budget = inf (or very large number)
asymptotic_row = pd.DataFrame({
    'budget': [np.inf],
    'recall_with_verification': [recinf_v],
    'recall_without_verification': [recinf_nv],
    'precision_with_verification': [precinf_v],
    'precision_without_verification': [precinf_nv],
    'fpr_population_with_verification': [fpr_pop_inf_v],
    'fpr_population_without_verification': [fpr_pop_inf_nv],
    'fpr_extracted_with_verification': [fpr_ext_inf_v],
    'fpr_extracted_without_verification': [fpr_ext_inf_nv],
    'tpr_extracted_with_verification': [tpr_ext_inf_v],
    'tpr_extracted_without_verification': [tpr_ext_inf_nv],
})

df_curves = pd.concat([df_curves, asymptotic_row], ignore_index=True)

# Save to CSV
csv_output_path = os.path.join(plots_dir, f"theoretical_curves_tau_{TAU}.csv")
df_curves.to_csv(csv_output_path, index=False)
print(f"Saved theoretical curves to: {csv_output_path}")
print(f"  Shape: {df_curves.shape} (including asymptotic row)")
print(f"  Columns: {df_curves.columns.tolist()}\n")

# -----------------------
# Plot: Recall, FPR(extracted), TPR(extracted)
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- (1) Recall
line_rec_v, = axes[0].plot(BUDGETS, rec_v, marker="o", label=f"Recall w/ verification (tau={TAU})")
axes[0].axhline(recinf_v, linestyle="--", color=line_rec_v.get_color(), label="Asymptote w/ verification")

line_rec_nv, = axes[0].plot(BUDGETS, rec_nv, marker="o", label="Recall w/o verification (q=1)")
axes[0].axhline(recinf_nv, linestyle="--", color=line_rec_nv.get_color(), label="Asymptote w/o verification")

axes[0].set_xscale("log")
axes[0].set_xlabel("Budget N (number of 'Name:' draws)")
axes[0].set_ylabel("Expected recall over members (split=train)")
axes[0].set_title("Recall vs Budget")
axes[0].legend()

# ---- (2) FPR extracted
line_fprx_v, = axes[1].plot(BUDGETS, fpr_ext_v, marker="o", label=f"FPR_extracted w/ verification (tau={TAU})")
axes[1].axhline(fpr_ext_inf_v, linestyle="--", color=line_fprx_v.get_color(), label="Asymptote FPR_extracted (ver)")

line_fprx_nv, = axes[1].plot(BUDGETS, fpr_ext_nv, marker="o", label="FPR_extracted w/o verification (q=1)")
axes[1].axhline(fpr_ext_inf_nv, linestyle="--", color=line_fprx_nv.get_color(), label="Asymptote FPR_extracted (no-ver)")

axes[1].set_xscale("log")
axes[1].set_xlabel("Budget N (number of 'Name:' draws)")
axes[1].set_ylabel("FPR among extracted non-members (val | extracted)")
axes[1].set_title("Extracted-stream FPR vs Budget")
axes[1].legend()

# ---- (3) TPR extracted
line_tprx_v, = axes[2].plot(BUDGETS, tpr_ext_v, marker="o", label=f"TPR_extracted w/ verification (tau={TAU})")
axes[2].axhline(tpr_ext_inf_v, linestyle="--", color=line_tprx_v.get_color(), label="Asymptote TPR_extracted (ver)")

line_tprx_nv, = axes[2].plot(BUDGETS, tpr_ext_nv, marker="o", label="TPR_extracted w/o verification (q=1)")
axes[2].axhline(tpr_ext_inf_nv, linestyle="--", color=line_tprx_nv.get_color(), label="Asymptote TPR_extracted (no-ver)")

axes[2].set_xscale("log")
axes[2].set_xlabel("Budget N (number of 'Name:' draws)")
axes[2].set_ylabel("TPR among extracted members (train | extracted)")
axes[2].set_title("Extracted-stream TPR vs Budget")
axes[2].legend()

axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)

fig.suptitle(f"Theory for {model} {dataset_size} {pii_rate} {n_epochs} {TAU}")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"recall_fprExtracted_tprExtracted_vs_budget_tau_{TAU}.png"), dpi=200)
plt.show()
print("Saved to", os.path.join(plots_dir, f'recall_fprExtracted_tprExtracted_vs_budget_tau_{TAU}.png'))

# -----------------------
# Optional: also plot population FPR for reference
# -----------------------
plt.figure(figsize=(7, 4))
plt.plot(BUDGETS, fpr_pop_v, marker="o", label=f"FPR(pop) w/ verification (tau={TAU})")
plt.axhline(fpr_pop_inf_v, linestyle="--", label="Asymptote FPR(pop) ver")
plt.plot(BUDGETS, fpr_pop_nv, marker="o", label="FPR(pop) w/o verification (q=1)")
plt.axhline(fpr_pop_inf_nv, linestyle="--", label="Asymptote FPR(pop) no-ver")
plt.xscale("log")
plt.xlabel("Budget N (number of 'Name:' draws)")
plt.ylabel("Population FPR over non-members (val)")
plt.title("Population FPR vs Budget")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"fpr_population_vs_budget_tau_{TAU}.png"), dpi=200)
plt.show()

# -----------------------
# Optional: precision plot (interpret carefully if CSV doesn't cover full universe)
# -----------------------
plt.figure(figsize=(7, 4))
plt.plot(BUDGETS, prec_v, marker="o", label=f"Precision w/ verification (tau={TAU})")
if not np.isnan(precinf_v):
    plt.axhline(precinf_v, linestyle="--", label="Asymptote precision ver")
plt.plot(BUDGETS, prec_nv, marker="o", label="Precision w/o verification (q=1)")
if not np.isnan(precinf_nv):
    plt.axhline(precinf_nv, linestyle="--", label="Asymptote precision no-ver")
plt.xscale("log")
plt.xlabel("Budget N (number of 'Name:' draws)")
plt.ylabel("Expected precision (over names in CSV)")
plt.title("Precision vs Budget")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"precision_vs_budget_with_without_ver_tau_{TAU}.png"), dpi=200)
plt.show()

# -----------------------
# Plot TPR, FPR, and Recall for different tau values (WITH verification only)
# -----------------------
tau_values = [0.3, 0.5, 0.7]
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Compute curves for each tau value
for tau_idx, tau_val in enumerate(tau_values):
    # Compute q for this tau value on the full dataframe
    q_tau_col = f"q_tau_{tau_val}"
    df[q_tau_col] = (df[SCORE_COL] >= tau_val).astype(int)
    
    # Re-split members/nonmembers with the new q column
    split_lower = df[SPLIT_COL].astype(str).str.lower()
    members_tau = df[split_lower == "train"].copy()
    nonmembers_tau = df[split_lower == "val"].copy()
    
    # Compute FPR, TPR, and Recall curves
    fpr_ext_tau, fpr_ext_inf_tau = compute_fpr_curve_extracted(nonmembers_tau, BUDGETS, q_col=q_tau_col)
    tpr_ext_tau, tpr_ext_inf_tau = compute_tpr_curve_extracted(members_tau, BUDGETS, q_col=q_tau_col)
    rec_tau, _, rec_inf_tau, _ = compute_recall_precision_curves(df, members_tau, BUDGETS, q_col=q_tau_col)
    
    # Calculate number of unique samples for asymptote recall
    # Asymptote recall = (1/|M|) sum_{i in M} reachable_i * q_i
    # where reachable_i = 1 if pi > 0, else 0
    pi_mem = members_tau[PI_COL].to_numpy(dtype=float)
    q_mem = members_tau[q_tau_col].to_numpy(dtype=float)
    mem_reachable = (pi_mem > 0).astype(float)
    num_unique_samples = np.sum(mem_reachable * q_mem).astype(int)
    total_members = len(members_tau)
    
    # Plot FPR
    line_fpr, = axes[0].plot(BUDGETS, fpr_ext_tau, marker=markers[tau_idx], 
                             color=colors[tau_idx], label=f"FPR (tau={tau_val})", linewidth=2)
    axes[0].axhline(fpr_ext_inf_tau, linestyle="--", color=colors[tau_idx], 
                    alpha=0.5, label=f"Asymptote FPR (tau={tau_val})")
    
    # Plot TPR
    line_tpr, = axes[1].plot(BUDGETS, tpr_ext_tau, marker=markers[tau_idx], 
                             color=colors[tau_idx], label=f"TPR (tau={tau_val})", linewidth=2)
    axes[1].axhline(tpr_ext_inf_tau, linestyle="--", color=colors[tau_idx], 
                    alpha=0.5, label=f"Asymptote TPR (tau={tau_val})")
    
    # Plot Recall with unique samples annotation
    line_rec, = axes[2].plot(BUDGETS, rec_tau, marker=markers[tau_idx], 
                             color=colors[tau_idx], label=f"Recall (tau={tau_val})", linewidth=2)
    axes[2].axhline(rec_inf_tau, linestyle="--", color=colors[tau_idx], 
                    alpha=0.5, label=f"Asymptote Recall (tau={tau_val}): {num_unique_samples}/{total_members} samples")

# Configure FPR plot
axes[0].set_xscale("log")
axes[0].set_xlabel("Budget N (number of 'Name:' draws)", fontsize=12)
axes[0].set_ylabel("FPR among extracted non-members (val | extracted)", fontsize=12)
axes[0].set_title("Extracted-stream FPR vs Budget (with verification)", fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# Configure TPR plot
axes[1].set_xscale("log")
axes[1].set_xlabel("Budget N (number of 'Name:' draws)", fontsize=12)
axes[1].set_ylabel("TPR among extracted members (train | extracted)", fontsize=12)
axes[1].set_title("Extracted-stream TPR vs Budget (with verification)", fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# Configure Recall plot
axes[2].set_xscale("log")
axes[2].set_xlabel("Budget N (number of 'Name:' draws)", fontsize=12)
axes[2].set_ylabel("Expected recall over members (split=train)", fontsize=12)
axes[2].set_title("Total Recall vs Budget (with verification)", fontsize=14, fontweight='bold')
axes[2].legend(fontsize=9)  # Slightly smaller font to accommodate sample counts
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

fig.suptitle(f"TPR, FPR, and Recall for different tau values - {model} {dataset_size} {pii_rate} {n_epochs}", 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"tpr_fpr_recall_vs_budget_different_tau.png"), dpi=200, bbox_inches='tight')
print(f"Saved TPR/FPR/Recall plot for different tau values to: {os.path.join(plots_dir, 'tpr_fpr_recall_vs_budget_different_tau.png')}")
plt.show()
