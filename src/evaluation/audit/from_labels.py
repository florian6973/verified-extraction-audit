"""Generic extraction audit from a labeled ``(entry, label)`` set + two models.

This is the dataset-agnostic entry point for scenario 2. It does NOT reimplement
any of the audit logic — it composes the existing pipeline pieces:

  * features         -> compute_ll_names.compute_name_ll  (finetuned/base LL)
  * verifier         -> train_mia_verifier_cv.train_crossfit_and_save (5-fold cross-fit)
  * pi               -> mia.ll_to_prob.convert_all_ll_to_prob (p = exp(LL))
  * operating tau    -> compute_threshold_fpr5_extr.compute_threshold_per_fold
                        (EXTRACTED-STREAM FPR <= target, weighted by pi)
  * theory curves    -> theory_curves (recall / extracted-stream FPR & TPR vs budget)
  * ensemble scoring -> mia.compute_scores.compute_scores (score_unseen fold ensemble)

The only glue specific to the minimal interface is (a) turning the labeled parquet
into the CSV the verifier expects, and (b) ``check_names`` — matching generated
completions to the labeled members (the ground-truth step that is otherwise
dataset-specific).

Report contains the THEORETICAL curves always, and the EXPERIMENTAL curves when
``--generations`` (from ``generate_completions.py``) is supplied.

Example
-------
    python -m src.evaluation.audit.from_labels \
        --labeled data/labeled.parquet \
        --base-model models/base/Llama_3.2-1B \
        --finetuned-model outputs/mydata/finetuned \
        --di-type name --budgets 1e5 1e6 \
        --generations outputs/mydata/completions.parquet \
        --output-dir outputs/mydata/audit
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from src.dataset.prepare.di_types import get_di_type, parse_candidate
from src.evaluation.pipeline.experimental.compute_ll_names import compute_name_ll_batch, load_model
from src.evaluation.pipeline.experimental.mia.train_mia_verifier_cv import train_crossfit_and_save
from src.evaluation.pipeline.experimental.mia.ll_to_prob import convert_all_ll_to_prob
from src.evaluation.pipeline.experimental.mia.compute_scores import compute_scores
from src.evaluation.pipeline.experimental.compute_threshold_fpr5_extr import compute_threshold_per_fold
from src.evaluation.pipeline import theory_curves as tc

DEFAULT_BUDGETS = [1e5, 1e6]


# --------------------------------------------------------------------------- #
# Feature extraction (reuses compute_ll_names.compute_name_ll / load_model)
# --------------------------------------------------------------------------- #
def _features_for(values, model_path, prompts, prefix, label="", batch_size=64):
    """{f'{prefix}{prompt}': [LL per value]} for one model, via compute_name_ll_batch.

    LLs are computed in batches (right-padded), which is 1-2 orders of magnitude
    faster than one-name-at-a-time on the ~K generated candidates; a tqdm bar per
    prompt keeps the progress visible.
    """
    tok, model, device = load_model(model_path)
    cols = {}
    for prompt in prompts:
        cols[f"{prefix}{prompt}"] = compute_name_ll_batch(
            prompt, values, tok, model, device, batch_size=batch_size,
            desc=f"LL {label}{prefix}{prompt}")
    del tok, model
    return cols


def build_feature_frame(values, base_model, finetuned_model, prompts, label="", batch_size=64):
    """DataFrame with value + ft_<prompt> (finetuned) + qi_<prompt> (base).

    Runs 4 LL passes over ``values`` (finetuned/base x each prompt); ``label``
    tags the progress bars so labeled-set vs generation passes are distinguishable.
    """
    ft = _features_for(values, finetuned_model, prompts, "ft_", label, batch_size)
    qi = _features_for(values, base_model, prompts, "qi_", label, batch_size)
    return pd.DataFrame({"value": list(values), **ft, **qi})


def feature_columns(prompts):
    return [f"ft_{p}" for p in prompts] + [f"qi_{p}" for p in prompts]


# --------------------------------------------------------------------------- #
# check_names: ground-truth membership for generated candidates (generic).
# --------------------------------------------------------------------------- #
def check_names(gen_candidates, members, nonmembers):
    """Label each parsed candidate 'train' (member) / 'val' (non-member) / 'other'."""
    def gt(c):
        if c in members:
            return "train"
        if c in nonmembers:
            return "val"
        return "other"
    return [gt(c) for c in gen_candidates]


# --------------------------------------------------------------------------- #
# Theoretical curves (reuses theory_curves) at a fixed tau.
# --------------------------------------------------------------------------- #
def theoretical_report(scores_p_csv, pi_col, budgets, tau):
    df = pd.read_csv(scores_p_csv)
    df[pi_col] = pd.to_numeric(df[pi_col], errors="coerce")
    df = df.dropna(subset=[pi_col])
    df["q"] = (pd.to_numeric(df["score_oof_member_proba"], errors="coerce") >= tau).astype(int)
    df["q_nover"] = 1
    split = df["split"].astype(str).str.lower()
    members, nonmembers = df[split == "train"], df[split == "val"]
    budgets = np.asarray(budgets, dtype=float)

    rec_v, _, rec_inf_v, _ = tc.compute_recall_precision_curves(df, members, budgets, pi_col, "q")
    rec_nv, _, rec_inf_nv, _ = tc.compute_recall_precision_curves(df, members, budgets, pi_col, "q_nover")
    fpr_ext_v, _ = tc.compute_fpr_curve_extracted(nonmembers, budgets, pi_col, "q")
    tpr_ext_v, _ = tc.compute_tpr_curve_extracted(members, budgets, pi_col, "q")

    return [
        {
            "Q": float(Q),
            "recall_with_verification": round(float(rec_v[i]), 6),
            "recall_without_verification": round(float(rec_nv[i]), 6),
            "fpr_extracted": round(float(fpr_ext_v[i]), 6),
            "tpr_extracted": round(float(tpr_ext_v[i]), 6),
        }
        for i, Q in enumerate(budgets)
    ], {"recall_with_verification": rec_inf_v, "recall_without_verification": rec_inf_nv}


# --------------------------------------------------------------------------- #
# Lightweight name filter (the structural part of paper/mia/name_mask, without the
# first-name gazetteer): drop de-identification placeholders and obvious non-names
# so they don't inflate the false-positive count. The paper's full filter also
# requires a known first name — see paper/mia/name_filter.py for exact parity.
# --------------------------------------------------------------------------- #
_NAME_JUNK = set('_,"\'()#/:$0123456789\n\t')


def _looks_like_name(s):
    """True if ``s`` looks like a 'First Last' name (>=2 alphabetic tokens, no junk)."""
    if not isinstance(s, str) or any(ch in _NAME_JUNK for ch in s):
        return False
    toks = s.split()
    if len(toks) < 2:
        return False
    return bool(re.sub(r'[^A-Za-z]', '', toks[0])) and bool(re.sub(r'[^A-Za-z]', '', toks[1]))


# --------------------------------------------------------------------------- #
# Extracted-stream confusion matrix (positive = injected member; negative = every
# other candidate). Matches paper/mia/bootstrap_metrics (y_true = groundtruth ==
# 'train'). ``is_member`` / ``passed`` are boolean arrays over the same candidates.
# --------------------------------------------------------------------------- #
def _confusion(is_member, passed, total_members):
    tp = int((is_member & passed).sum())
    fp = int((~is_member & passed).sum())
    fn = int((is_member & ~passed).sum())
    tn = int((~is_member & ~passed).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / (tp + fn) if (tp + fn) else None,      # recall on generated members
        "fpr": fp / (fp + tn) if (fp + tn) else None,      # non-members flagged / all non-members
        "ppv": tp / (tp + fp) if (tp + fp) else None,      # precision of the verified stream
        "recall_with_verification": tp / total_members if total_members else None,
    }


# --------------------------------------------------------------------------- #
# Experimental measurement (reuses compute_scores fold ensemble).
# --------------------------------------------------------------------------- #
def experimental_report(gens_df, labeled_df, di, prompts, base_model, finetuned_model,
                        scores_csv, models_dir, out_dir, tau, budgets, value_col="value",
                        seed=42, n_bootstrap=1000, filter_names=False):
    members = {parse_candidate(di, e) for e, l in zip(labeled_df["entry"], labeled_df["label"]) if l == 1}
    nonmembers = {parse_candidate(di, e) for e, l in zip(labeled_df["entry"], labeled_df["label"]) if l == 0}
    total = len(members)
    if total == 0 or value_col not in gens_df.columns:
        return None

    gen_cands = [parse_candidate(di, str(v)) for v in gens_df[value_col].tolist()]
    uniq = list(dict.fromkeys(gen_cands))  # order-preserving unique

    # Features for candidates, then ensemble-score via compute_scores (score_unseen).
    print(f"[audit] experimental extraction: {len(gens_df)} generations -> {len(uniq)} unique "
          f"candidates; computing LL features (4 passes: finetuned/base x {len(prompts)} prompts)...",
          flush=True)
    feats = build_feature_frame(uniq, base_model, finetuned_model, prompts, label="gens ")
    feats["groundtruth"] = check_names(uniq, members, nonmembers)
    ll_csv = os.path.join(out_dir, "all_names_ll_computed.csv")
    feats.to_csv(ll_csv, index=False)

    scored = compute_scores(ll_csv, src_pred=scores_csv, src_other=models_dir,
                            output_path=os.path.join(out_dir, "all_names_ll_computed_with_scores.csv"),
                            name_col="value", score_col="score_oof_member_proba")
    score_of = {}
    if len(scored) and "score_oof_member_proba" in scored.columns:
        for v, s in zip(scored["value"], scored["score_oof_member_proba"]):
            score_of[v] = float(s) if pd.notna(s) else 0.0

    # First-seen query index for EVERY unique candidate, so we can build the full
    # extracted-stream confusion matrix at any budget Q.
    first_seen = {}
    for i, c in enumerate(gen_cands):
        if c not in first_seen:
            first_seen[c] = i
    n_gen = len(gens_df)

    # Vectors over all unique candidates (aligned by index). Positive class = an
    # injected identifier (a member); NEGATIVE = every other candidate — labeled
    # non-members AND unlabeled 'other' — matching the paper's bootstrap_metrics
    # (y_true = groundtruth == 'train'; negatives = 'val' + 'other'). The verifier
    # was run on every completion, so a non-member passing tau is a false positive.
    uniq_all = list(first_seen.keys())
    n_before_filter = len(uniq_all)
    if filter_names:
        # Drop de-id placeholders / non-name junk so they don't inflate FP (as the
        # paper does with name_mask). Members are real names, so they survive.
        uniq_all = [c for c in uniq_all if _looks_like_name(c)]
    idx_arr = np.array([first_seen[c] for c in uniq_all])
    is_member = np.array([c in members for c in uniq_all], dtype=bool)
    passed = np.array([score_of.get(c, 0.0) >= tau for c in uniq_all], dtype=bool)
    rng = np.random.default_rng(seed)

    def _ci(a):
        return [round(float(np.percentile(a, 2.5)), 6), round(float(np.percentile(a, 97.5)), 6)]

    def _safe(num, den):
        return round(num / den, 6) if den else None

    # Experimental budgets can't exceed the queries actually run, so cap each Q at
    # n_gen and drop duplicates (a raw 1e6 row was identical to 1e5 with only 1e5
    # completions). Theory extrapolates; experiment can't.
    curves, seen_budgets = [], set()
    for Q in budgets:
        Qi = min(int(Q), n_gen)
        if Qi in seen_budgets:
            continue
        seen_budgets.add(Qi)
        sel = idx_arr < Qi
        yt, yp = is_member[sel], passed[sel]
        n = int(sel.sum())
        m = _confusion(yt, yp, total)
        # 95% bootstrap CIs by resampling the in-stream candidates with replacement.
        tprs, fprs, ppvs, recs = [], [], [], []
        for _ in range(n_bootstrap if n else 0):
            s = rng.integers(0, n, n)
            bm = _confusion(yt[s], yp[s], total)
            tprs.append(bm["tpr"] or 0.0)
            fprs.append(bm["fpr"] or 0.0)
            ppvs.append(bm["ppv"] or 0.0)
            recs.append(bm["recall_with_verification"] or 0.0)
        curves.append({
            "Q": float(Qi),
            "n_candidates": n,
            "n_members_generated": int(yt.sum()),
            # Full confusion matrix (positive = injected member, negative = all others).
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "tpr": round(m["tpr"], 6) if m["tpr"] is not None else None,   # recall on generated members
            "fpr": round(m["fpr"], 6) if m["fpr"] is not None else None,   # non-members flagged / all non-members
            "ppv": round(m["ppv"], 6) if m["ppv"] is not None else None,   # precision of the verified stream
            # Extraction recall over ALL injected members (fixed denominator).
            "recall_without_verification": _safe(int(yt.sum()), total),
            "recall_with_verification": (round(m["recall_with_verification"], 6)
                                         if m["recall_with_verification"] is not None else None),
            "tpr_ci95": _ci(tprs) if tprs else None,
            "fpr_ci95": _ci(fprs) if fprs else None,
            "ppv_ci95": _ci(ppvs) if ppvs else None,
            "recall_with_verification_ci95": _ci(recs) if recs else None,
        })
    return {"n_generations": int(n_gen),
            "unique_candidates": len(uniq_all),
            "name_filter_applied": bool(filter_names),
            "unique_candidates_before_filter": int(n_before_filter),
            "unique_members_extracted": int(is_member.sum()),
            "total_members": total,
            "n_bootstrap": int(n_bootstrap),
            "note": ("Confusion matrix over ALL unique generated candidates: positive = injected "
                     "member ('train'), negative = every other candidate (labeled non-members + "
                     "unlabeled 'other'), matching paper/mia/bootstrap_metrics. fpr = FP/(FP+TN) "
                     "over all non-members; tpr = TP/(TP+FN) over generated members; "
                     "recall_with_verification = TP/total_members. CIs are 95% bootstrap. To match "
                     "the paper's language-filtered numbers, drop non-name junk first (--filter-names)."),
            "curves": curves}


# --------------------------------------------------------------------------- #
def run(args):
    di = get_di_type(args.di_type)
    prompts = args.prompts or di.query_prompts
    os.makedirs(args.output_dir, exist_ok=True)

    labeled = pd.read_parquet(args.labeled)
    labeled["label"] = labeled["label"].astype(int)
    # Normalize entries through the same parser used on completions.
    labeled["value"] = [parse_candidate(di, e) for e in labeled["entry"]]

    # 1) Features -> df_combined.csv (columns the verifier expects)
    print(f"[audit] 1/5 verifier features for {len(labeled)} labeled entries...", flush=True)
    feats = build_feature_frame(labeled["value"].tolist(), args.base_model, args.finetuned_model,
                                prompts, label="labeled ")
    feats["value"] = labeled["value"].values
    feats["split"] = np.where(labeled["label"].values == 1, "train", "val")
    df_combined_csv = os.path.join(args.output_dir, "df_combined.csv")
    feats.to_csv(df_combined_csv, index=False)

    # 2) Cross-fit verifier (reused) -> OOF scores + fold models
    print("[audit] 2/5 training cross-fit verifier...", flush=True)
    feat_cols = [c for c in feature_columns(prompts) if c in feats.columns]
    k = min(5, int(labeled["label"].value_counts().min()))
    models_dir = os.path.join(args.output_dir, "verifier_models")
    scores_csv = os.path.join(args.output_dir, "scores.csv")
    train_crossfit_and_save(csv_path=df_combined_csv, out_csv_path=scores_csv,
                            models_dir=models_dir, feature_cols=feat_cols,
                            split_col="split", name_col="value", k=max(2, k), seed=args.seed)

    # 3) pi = exp(LL) (reused) -> scores_p.csv with p_ft_/p_base_
    scores_p_csv = os.path.join(args.output_dir, "scores_p.csv")
    convert_all_ll_to_prob(scores_csv, scores_p_csv)
    pi_col = f"p_ft_{di.primary_prompt}"

    # 4) tau at extracted-stream FPR <= target (reused)
    print("[audit] 3/5 operating threshold (extracted-stream FPR)...", flush=True)
    df_scores = pd.read_csv(scores_p_csv)
    fpr_budget = args.fpr_budget or max(args.budgets)
    fold_res = compute_threshold_per_fold(df_scores, fold_id_col="fold_id", y_true_col="y_true",
                                          score_col="score_oof_member_proba", target_fpr=args.target_fpr,
                                          budget_N=float(fpr_budget), pi_col=pi_col)
    tau = float(np.mean([r["threshold"] for r in fold_res.values()])) if fold_res else 0.5

    # 5) Theoretical curves (reused)
    print("[audit] 4/5 theoretical curves...", flush=True)
    theo, theo_inf = theoretical_report(scores_p_csv, pi_col, args.budgets, tau)

    # Verifier discrimination on the labeled set (OOF) — the audit's headline number.
    try:
        from sklearn.metrics import roc_auc_score
        verifier_auc = round(float(roc_auc_score(df_scores["y_true"], df_scores["score_oof_member_proba"])), 6)
    except Exception:
        verifier_auc = None

    report = {
        "di_type": di.name,
        "prompts": prompts,
        "n_members": int((labeled["label"] == 1).sum()),
        "n_non_members": int((labeled["label"] == 0).sum()),
        "feature_columns": feat_cols,
        "verifier_auc": verifier_auc,
        "operating_point": {"target_fpr": args.target_fpr, "fpr_budget": float(fpr_budget),
                            "tau_extracted_fpr": tau},
        "extraction_theoretical": theo,
        "extraction_theoretical_asymptote": theo_inf,
    }

    # 6) Experimental (optional)
    if args.generations:
        print("[audit] 5/5 experimental extraction from generations...", flush=True)
        gens_df = (pd.read_parquet(args.generations) if args.generations.endswith(".parquet")
                   else pd.read_csv(args.generations))
        report["extraction_experimental"] = experimental_report(
            gens_df, labeled, di, prompts, args.base_model, args.finetuned_model,
            scores_csv, models_dir, args.output_dir, tau, args.budgets,
            seed=args.seed, n_bootstrap=args.bootstrap, filter_names=args.filter_names)

    out = os.path.join(args.output_dir, "audit_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote report -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", required=True, help="Parquet of (entry, label); 1=member, 0=non-member")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--di-type", default="name")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Query prompts (default: the DI type's query_prompts)")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--fpr-budget", type=float, default=None,
                        help="Budget N at which the extracted-stream FPR<=target tau is chosen "
                             "(default: max(--budgets))")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--generations", default=None,
                        help="Completions parquet/csv (from generate_completions.py) for the experimental curves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for experimental metric 95%% CIs (0 to disable)")
    parser.add_argument("--filter-names", action="store_true",
                        help="Drop non-name junk (de-id ___ placeholders, digits, punctuation) from "
                             "generated candidates before the confusion matrix, so it doesn't inflate FP")
    parser.add_argument("--output-dir", default="outputs/audit")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
