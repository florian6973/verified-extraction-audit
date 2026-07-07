"""Closed-form attack curves for one (dataset, model, tau): compute + plot.

Reads the verifier scores CSV (``scores_*_p.csv`` with per-draw extraction
probability ``pi`` and verifier score) and produces, as a function of attacker
query budget N, the recall / precision / population-FPR / extracted-stream FPR /
extracted-stream TPR curves. The curve math lives in
:mod:`src.evaluation.pipeline.theory_curves` (single source); this module is the
config-driven driver that assembles the curves into a CSV and plots them.

    python -m src.evaluation.pipeline.attack_curves --config <cfg.yaml> --tau 0.5

Importing this module has no side effects (compute/plot only run from ``main``),
and plotting writes PNGs — it never blocks on ``plt.show()``.
"""

import os
import warnings

import numpy as np
import pandas as pd

from src.evaluation.pipeline import theory_curves as tc

# Column conventions in the scores CSV.
PI_COL = "p_ft_Name: "               # per-draw extraction probability
SPLIT_COL = "split_x"                # "train" = member, "val" = non-member
SCORE_COL = "score_oof_member_proba"  # verifier score

DEFAULT_BUDGETS = np.array([
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000,
    50000000, 100000000, 200000000, 500000000, 1000000000,
    2000000000, 5000000000, 10000000000, 20000000000, 50000000000,
    100000000000, 200000000000, 500000000000, 1000000000000, 2000000000000,
    5000000000000, 10000000000000,
], dtype=float)


def load_scores(csv_path, pi_col=PI_COL, split_col=SPLIT_COL, score_col=SCORE_COL):
    """Load + clean the scores CSV; add the q (verify) / q_nover columns."""
    df = pd.read_csv(csv_path).copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[pi_col] = pd.to_numeric(df[pi_col], errors="coerce")
    if "group" in df.columns:
        df[score_col] = df[score_col].fillna(1 - df["group"])
    else:
        warnings.warn("Column 'group' not found; not filling missing scores with 1 - group.")
    return df.dropna(subset=[pi_col])


def compute_curves(df, tau, budgets=DEFAULT_BUDGETS, pi_col=PI_COL,
                   split_col=SPLIT_COL, score_col=SCORE_COL):
    """Compute all attack curves; return a DataFrame (with an inf-budget asymptote row).

    Reuses :mod:`theory_curves` for every quantity.
    """
    df = df.copy()
    df["q"] = (df[score_col] >= tau).astype(int)  # verification at tau
    df["q_nover"] = 1                              # no-verification baseline
    split = df[split_col].astype(str).str.lower()
    members, nonmembers = df[split == "train"].copy(), df[split == "val"].copy()
    if len(members) == 0 or len(nonmembers) == 0:
        raise ValueError("Expected split=='train' (members) and split=='val' (non-members).")

    budgets = np.asarray(budgets, dtype=float)
    rec_v, prec_v, rec_inf_v, prec_inf_v = tc.compute_recall_precision_curves(df, members, budgets, pi_col, "q")
    rec_nv, prec_nv, rec_inf_nv, prec_inf_nv = tc.compute_recall_precision_curves(df, members, budgets, pi_col, "q_nover")
    fprp_v, fprp_inf_v = tc.compute_fpr_curve_population(nonmembers, budgets, pi_col, "q")
    fprp_nv, fprp_inf_nv = tc.compute_fpr_curve_population(nonmembers, budgets, pi_col, "q_nover")
    fprx_v, fprx_inf_v = tc.compute_fpr_curve_extracted(nonmembers, budgets, pi_col, "q")
    fprx_nv, fprx_inf_nv = tc.compute_fpr_curve_extracted(nonmembers, budgets, pi_col, "q_nover")
    tprx_v, tprx_inf_v = tc.compute_tpr_curve_extracted(members, budgets, pi_col, "q")
    tprx_nv, tprx_inf_nv = tc.compute_tpr_curve_extracted(members, budgets, pi_col, "q_nover")

    cols = {
        "budget": budgets,
        "recall_with_verification": rec_v, "recall_without_verification": rec_nv,
        "precision_with_verification": prec_v, "precision_without_verification": prec_nv,
        "fpr_population_with_verification": fprp_v, "fpr_population_without_verification": fprp_nv,
        "fpr_extracted_with_verification": fprx_v, "fpr_extracted_without_verification": fprx_nv,
        "tpr_extracted_with_verification": tprx_v, "tpr_extracted_without_verification": tprx_nv,
    }
    inf_row = {
        "budget": np.inf,
        "recall_with_verification": rec_inf_v, "recall_without_verification": rec_inf_nv,
        "precision_with_verification": prec_inf_v, "precision_without_verification": prec_inf_nv,
        "fpr_population_with_verification": fprp_inf_v, "fpr_population_without_verification": fprp_inf_nv,
        "fpr_extracted_with_verification": fprx_inf_v, "fpr_extracted_without_verification": fprx_inf_nv,
        "tpr_extracted_with_verification": tprx_inf_v, "tpr_extracted_without_verification": tprx_inf_nv,
    }
    return pd.concat([pd.DataFrame(cols), pd.DataFrame([inf_row])], ignore_index=True)


def plot_curves(df_curves, tau, out_dir, title=""):
    """Plot recall / extracted-stream FPR / extracted-stream TPR vs budget (no plt.show)."""
    import matplotlib
    matplotlib.use("Agg")  # headless: never block on a display
    import matplotlib.pyplot as plt

    finite = df_curves[np.isfinite(df_curves["budget"])]
    b = finite["budget"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("recall", "recall_with_verification", "recall_without_verification", "Recall vs Budget"),
        ("fpr", "fpr_extracted_with_verification", "fpr_extracted_without_verification", "Extracted-stream FPR vs Budget"),
        ("tpr", "tpr_extracted_with_verification", "tpr_extracted_without_verification", "Extracted-stream TPR vs Budget"),
    ]
    for ax, (_, cv, cnv, ttl) in zip(axes, panels):
        ax.plot(b, finite[cv], marker="o", label=f"with verification (tau={tau})")
        ax.plot(b, finite[cnv], marker="o", label="without verification (q=1)")
        ax.set_xscale("log")
        ax.set_xlabel("Budget N ('Name:' draws)")
        ax.set_title(ttl)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    out = os.path.join(out_dir, f"attack_curves_tau_{tau}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    import argparse
    from src.evaluation.pipeline.experimental.config_loader import load_config
    from src.evaluation.pipeline.experimental.config_helper import get_output_dir

    parser = argparse.ArgumentParser(description="Compute + plot theoretical attack curves.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--tau", type=float, default=0.7, help="Verifier threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    f = config["filters"]
    out_dir = os.path.join(get_output_dir(config), "plots_theory")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        get_output_dir(config),
        f"scores_{f['model']}_{f['dataset_size']}_pii_rate_{f['pii_rate']}_n_epochs_{f['n_epochs']}_p.csv")

    df = load_scores(csv_path)
    df_curves = compute_curves(df, args.tau)
    out_csv = os.path.join(out_dir, f"theoretical_curves_tau_{args.tau}.csv")
    df_curves.to_csv(out_csv, index=False)
    print(f"Saved theoretical curves -> {out_csv}  {df_curves.shape}")
    png = plot_curves(df_curves, args.tau, out_dir,
                      title=f"Theory {f['model']} {f['dataset_size']} {f['pii_rate']} {f['n_epochs']} tau={args.tau}")
    print(f"Saved plot -> {png}")


if __name__ == "__main__":
    main()
