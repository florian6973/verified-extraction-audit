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


def _newest_mtime(path):
    """Most recent mtime under a model dir (or the file itself); 0 if missing."""
    if os.path.isfile(path):
        return os.path.getmtime(path)
    newest = 0.0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return newest


def cached_feature_frame(values, base_model, finetuned_model, prompts, out_dir, label="", batch_size=64):
    """Like build_feature_frame but caches LLs by candidate string in
    ``out_dir/ll_cache.parquet``. LLs depend only on (string, models, prompts), so
    re-audits (e.g. different match flags) reuse them; the cache is ignored when
    either model is newer than it, so a retrain recomputes. Only missing candidates
    hit the GPU.
    """
    cols = feature_columns(prompts)
    cache_path = os.path.join(out_dir, "ll_cache.parquet")
    cache = None
    if os.path.exists(cache_path):
        try:
            if os.path.getmtime(cache_path) >= max(_newest_mtime(finetuned_model), _newest_mtime(base_model)):
                c = pd.read_parquet(cache_path)
                if "value" in c.columns and set(cols).issubset(c.columns):
                    cache = c.drop_duplicates("value").set_index("value")
        except Exception:
            cache = None
    values = list(values)
    have = [v for v in values if cache is not None and v in cache.index]
    missing = [v for v in values if cache is None or v not in cache.index]
    print(f"[audit] LL cache: {len(have)} hit / {len(missing)} to compute (of {len(values)})", flush=True)
    frames = []
    if have:
        frames.append(cache.loc[have, cols].reset_index())
    if missing:
        frames.append(build_feature_frame(missing, base_model, finetuned_model, prompts, label, batch_size))
    feats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["value"] + cols)
    try:  # refresh the cache with the union (best-effort)
        union = feats[["value"] + cols]
        if cache is not None:
            union = pd.concat([cache.reset_index()[["value"] + cols], union], ignore_index=True)
        union.drop_duplicates("value").to_parquet(cache_path, index=False)
    except Exception:
        pass
    # return rows in the requested order
    return feats.drop_duplicates("value").set_index("value").reindex(values).reset_index()


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


def _parse(di, s, strict=False):
    """Parse a raw string to its identifier value.

    Default is the DI type's parser (for names: first two words, dots stripped,
    ``.title()``-cased). With ``strict`` and a name-type DI, the ``.title()``
    case-fold is skipped, so a generated candidate only matches a member when the
    casing is identical — i.e. matching uses the exact string that pi = exp(LL) is
    computed for. This isolates whether case normalization is inflating coverage.
    """
    if strict and di.parse_strategy == "first_two_words":
        return " ".join(str(s).strip().split()[:2]).replace(".", "").strip()
    return parse_candidate(di, s)


def _extract(di, s, strict=False, position_match=False):
    """Parse a completion into a candidate value, or ``None`` if it shouldn't count.

    With ``position_match`` and a name-type DI, require the extracted name to sit at
    index 1 of the raw completion — a clean `` First Last`` right after the prompt —
    reproducing the paper's ``ner_ll_remaining`` ``idx == 1`` filter. This makes
    "extracted" mean "the completion STARTS with the name", which is exactly what
    ``pi = exp(LL)`` models (``P(completion starts with the name tokens)``). Without
    it, a member counts even when it appears in a misaligned completion, so
    experimental coverage runs ~2x above the theory.
    """
    val = _parse(di, s, strict)
    if position_match and di.parse_strategy == "first_two_words":
        if str(s).lower().find(val.lower()) != 1:
            return None
    return val


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
                        seed=42, n_bootstrap=1000, filter_names=False, strict_match=False,
                        position_match=False):
    members = {_parse(di, e, strict_match) for e, l in zip(labeled_df["entry"], labeled_df["label"]) if l == 1}
    nonmembers = {_parse(di, e, strict_match) for e, l in zip(labeled_df["entry"], labeled_df["label"]) if l == 0}
    total = len(members)
    if total == 0 or value_col not in gens_df.columns:
        return None

    # One candidate per completion; None when position_match rejects a misaligned one.
    gen_cands = [_extract(di, str(v), strict_match, position_match) for v in gens_df[value_col].tolist()]
    uniq = list(dict.fromkeys(c for c in gen_cands if c is not None))  # order-preserving unique

    # Features for candidates, then ensemble-score via compute_scores (score_unseen).
    print(f"[audit] experimental extraction: {len(gens_df)} generations -> {len(uniq)} unique "
          f"candidates; computing LL features (4 passes: finetuned/base x {len(prompts)} prompts)...",
          flush=True)
    feats = cached_feature_frame(uniq, base_model, finetuned_model, prompts, out_dir, label="gens ")
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
        if c is not None and c not in first_seen:
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

    # Coverage calibration: is pi = exp(LL) actually the per-query generation rate?
    # For each member, compare its empirical hit count in the generations to N*pi
    # (its expected count under the theory). ratio >> 1 => pi under-predicts.
    coverage_calibration = None
    try:
        import math
        from collections import Counter
        counts = Counter(c for c in gen_cands if c is not None)   # multiplicities over valid extractions
        sdf = pd.read_csv(scores_csv)
        ll_col = f"ft_{di.primary_prompt}"
        rows = []
        for v, yt, ll in zip(sdf["value"], sdf["y_true"], sdf[ll_col]):
            if int(yt) != 1:
                continue
            pi = math.exp(float(ll)) if pd.notna(ll) else 0.0
            rows.append((v, pi, counts.get(v, 0), n_gen * pi))
        ddf = pd.DataFrame(rows, columns=["value", "pi", "empirical_hits", "expected_hits_N_pi"])
        ddf.to_csv(os.path.join(out_dir, "coverage_diag.csv"), index=False)
        g = ddf[(ddf["empirical_hits"] > 0) & (ddf["pi"] > 0)]
        ratio = (g["empirical_hits"] / g["expected_hits_N_pi"]) if len(g) else pd.Series(dtype=float)
        coverage_calibration = {
            "members_generated": int((ddf["empirical_hits"] > 0).sum()),
            "median_empirical_over_expected": round(float(ratio.median()), 4) if len(ratio) else None,
            "iqr_empirical_over_expected": ([round(float(ratio.quantile(.25)), 4),
                                             round(float(ratio.quantile(.75)), 4)] if len(ratio) else None),
            "generated_but_expected_lt_0p5": int(((ddf["empirical_hits"] > 0)
                                                  & (ddf["expected_hits_N_pi"] < 0.5)).sum()),
            "note": ("empirical_hits vs N*pi per member (coverage_diag.csv). ratio>1 => pi=exp(LL) "
                     "under-predicts the true generation rate; that gap (not casing) drives "
                     "experimental recall_without > theoretical."),
        }
    except Exception as e:  # diagnostic must never break the audit
        coverage_calibration = {"error": repr(e)}

    return {"n_generations": int(n_gen),
            "unique_candidates": len(uniq_all),
            "valid_extractions": int(sum(c is not None for c in gen_cands)),
            "strict_match": bool(strict_match),
            "position_match": bool(position_match),
            "coverage_calibration": coverage_calibration,
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
    # Normalize entries through the same parser used on completions (so pi and the
    # extraction match use the identical string; --strict-match keeps exact case).
    labeled["value"] = [_parse(di, e, args.strict_match) for e in labeled["entry"]]

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
            seed=args.seed, n_bootstrap=args.bootstrap, filter_names=args.filter_names,
            strict_match=args.strict_match, position_match=args.position_match)

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
    parser.add_argument("--strict-match", action="store_true",
                        help="Match generated candidates to members with EXACT case (skip .title()), "
                             "so extraction uses the same string that pi=exp(LL) is computed for")
    parser.add_argument("--position-match", action="store_true",
                        help="Count a completion as an extraction only when the name is at index 1 "
                             "(a clean ' First Last' right after the prompt), as the paper's "
                             "ner_ll_remaining does; aligns experimental coverage with pi")
    parser.add_argument("--output-dir", default="outputs/audit")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
