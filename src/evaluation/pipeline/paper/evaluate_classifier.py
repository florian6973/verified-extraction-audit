# Evaluate pre-trained classifier on generated names
# Load classifier, exclude training samples, compute confusion matrix

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve, roc_curve

from config_loader import load_config

# Parse arguments first
parser = argparse.ArgumentParser(description='Evaluate classifier on generated names')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--classifier', type=str, default=None, help='Path to classifier pkl file (overrides config)')
parser.add_argument('--df_temp_sub', type=str, default=None, help='Path to df_temp_sub CSV (overrides config)')
parser.add_argument('--computed_ll', type=str, default=None, help='Path to computed ll CSV (overrides config)')
parser.add_argument('--output', type=str, default=None, help='Output CSV path (overrides config)')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Get paths from config (can be overridden by args)
OUTPUT_DIR = config['output_dir']
CLASSIFIER_FILE = args.classifier or config['classifier']['clf_file']
DF_TEMP_SUB_FILE = args.df_temp_sub or config['classifier']['df_temp_sub_file']
COMPUTED_LL_FILE = args.computed_ll or os.path.join(OUTPUT_DIR, 'all_names_ll_computed.csv')
OUTPUT_RESULTS_FILE = args.output or os.path.join(OUTPUT_DIR, 'classifier_evaluation_results.csv')

# Feature column names (must match what classifier was trained on)
# models = ['p_pre', 'p_ft']
# prompts = ['Name: ', 'Patient: ']
FEATURE_COLUMNS = ['p_pre_Name: ', 'p_pre_Patient: ', 'p_ft_Name: ', 'p_ft_Patient: ']


def load_classifier(clf_path):
    """Load the pre-trained classifier."""
    print(f"Loading classifier from: {clf_path}")
    clf = joblib.load(clf_path)
    print(f"Classifier type: {type(clf).__name__}")
    return clf


def clean_name(name):
    """Clean name: lowercase and remove dots (keep spaces for consistency)."""
    if isinstance(name, str):
        name = name.lower().replace('.', '')
    return name


def load_training_samples(df_temp_sub_path):
    """Load df_temp_sub to get names used for training the classifier.
    
    Training samples are those where y_pred_proba is NA (not in test set).
    """
    print(f"Loading training samples from: {df_temp_sub_path}")
    df = pd.read_csv(df_temp_sub_path)
    print(f"Total samples in df_temp_sub: {len(df)}")
    
    # Training samples are those where y_pred_proba is NA
    # (the classifier was trained on these, so we exclude them from evaluation)
    df_train = df[df['y_pred_proba'].isna()]
    print(f"Training samples (y_pred_proba is NA): {len(df_train)}")
    print(f"Test samples (y_pred_proba is not NA): {len(df) - len(df_train)}")
    
    # Get unique names used for training (cleaned: lowercase, no dots)
    training_names = set(df_train['value'].dropna().apply(clean_name).unique())
    print(f"Unique names in training data: {len(training_names)}")
    
    return df, training_names


def load_and_prepare_features(ll_file, training_names):
    """
    Load computed ll data and prepare features for classifier.
    Exclude names that were used for training the classifier.
    """
    print(f"\nLoading computed ll from: {ll_file}")
    df = pd.read_csv(ll_file)
    print(f"Total names in computed ll: {len(df)}")
    
    # Check available columns
    print(f"Available columns: {df.columns.tolist()}")
    
    # Map column names from computed file to classifier feature names
    # Computed file has: ll_base_name:, ll_base_patient:, ll_finetuned_name:, ll_finetuned_patient:
    # Classifier expects: p_pre_Name: , p_pre_Patient: , p_ft_Name: , p_ft_Patient: 
    
    column_mapping = {
        'll_base_name:': 'p_pre_Name: ',
        'll_base_patient:': 'p_pre_Patient: ',
        'll_finetuned_name:': 'p_ft_Name: ',
        'll_finetuned_patient:': 'p_ft_Patient: ',
    }
    
    # Check if we have the expected columns
    available_ll_cols = [c for c in df.columns if c.startswith('ll_')]
    print(f"Available ll columns: {available_ll_cols}")
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Check for missing features
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        print("Available columns after mapping:", df.columns.tolist())
        return None, None
    
    # Exclude names used for training (use same cleaning as training_names)
    df['name_clean'] = df['name'].apply(clean_name)
    df_excluded = df[df['name_clean'].isin(training_names)]
    df_eval = df[~df['name_clean'].isin(training_names)]
    
    print(f"\nExcluded {len(df_excluded)} names that were in training data")
    print(f"Remaining names for evaluation: {len(df_eval)}")
    
    # Also show breakdown by groundtruth
    print(f"\nGroundtruth distribution (excluded):")
    if 'groundtruth' in df_excluded.columns:
        print(df_excluded['groundtruth'].value_counts())
    
    print(f"\nGroundtruth distribution (evaluation):")
    if 'groundtruth' in df_eval.columns:
        print(df_eval['groundtruth'].value_counts())
    
    return df_eval, df_excluded


def compute_metrics_at_threshold(y_true, y_pred_proba, threshold, threshold_name=""):
    """
    Compute confusion matrix and metrics at a specific threshold.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    F1 = 2 * PPV * TPR / (PPV + TPR) if (PPV + TPR) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"\n--- {threshold_name} (threshold = {threshold:.4f}) ---")
    print(f"Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0 (Val)  1 (Train)")
    print(f"Actual 0 (Val)   {TN:6d}   {FP:6d}")
    print(f"Actual 1 (Train) {FN:6d}   {TP:6d}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {F1:.4f}")
    print(f"  FPR:       {FPR:.4f}  (Val predicted as Train)")
    print(f"  TPR:       {TPR:.4f}  (Train correctly identified, Recall)")
    print(f"  Precision: {PPV:.4f}")
    print(f"  TNR:       {TNR:.4f}  (Specificity)")
    
    return {
        'threshold': threshold,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'FPR': FPR, 'TPR': TPR, 'TNR': TNR, 'FNR': FNR,
        'PPV': PPV, 'NPV': NPV, 'F1': F1, 'accuracy': accuracy
    }


def evaluate_classifier(clf, df_eval, feature_columns):
    """
    Run classifier on evaluation data and compute metrics.
    """
    print(f"\n{'='*60}")
    print("Evaluating classifier")
    print(f"{'='*60}")
    
    # Check for NaN in features
    df_valid = df_eval.dropna(subset=feature_columns)
    print(f"Valid samples (no NaN in features): {len(df_valid)} / {len(df_eval)}")
    
    if len(df_valid) == 0:
        print("No valid samples to evaluate!")
        return None
    
    # Prepare features (convert to log probabilities like in training)
    X = df_valid[feature_columns].values
    X = np.log(np.exp(X))  # Already log probs, but ensure consistency
    
    # Get ground truth labels
    # train = 1, val/other = 0
    y_true = (df_valid['groundtruth'] == 'train').astype(int).values
    
    print(f"\nGround truth distribution:")
    print(f"  Train (1): {(y_true == 1).sum()}")
    print(f"  Val/Other (0): {(y_true == 0).sum()}")
    
    # Predict
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    df_valid = df_valid.copy()
    df_valid['y_pred'] = y_pred
    df_valid['y_pred_proba'] = y_pred_proba
    df_valid['y_true'] = y_true
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # AUC
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: Cannot compute (only one class present)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0 (Val)  1 (Train)")
    print(f"Actual 0 (Val)   {TN:6d}   {FP:6d}")
    print(f"Actual 1 (Train) {FN:6d}   {TP:6d}")
    
    # FPR, TPR, and other rates
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall/Sensitivity)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate (Specificity)
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value (Precision)
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative Predictive Value
    
    print(f"\nRate Metrics:")
    print(f"  FPR (False Positive Rate): {FPR:.4f}  (Val predicted as Train)")
    print(f"  TPR (True Positive Rate):  {TPR:.4f}  (Train correctly identified, Recall)")
    print(f"  TNR (True Negative Rate):  {TNR:.4f}  (Val correctly identified, Specificity)")
    print(f"  FNR (False Negative Rate): {FNR:.4f}  (Train predicted as Val)")
    print(f"  PPV (Precision):           {PPV:.4f}  (Of predicted Train, how many are Train)")
    print(f"  NPV:                       {NPV:.4f}  (Of predicted Val, how many are Val)")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Val/Other', 'Train']))
    
    # ============================================================
    # THRESHOLD ANALYSIS
    # ============================================================
    print(f"\n{'='*60}")
    print("THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    
    # Find optimal F1 threshold
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_f1_idx = np.argmax(f1_scores[:-1])  # Last element is for recall=0
    optimal_f1_threshold = thresholds_pr[best_f1_idx]
    
    # Find optimal FPR threshold (minimize FPR while TPR >= 0.5, or lowest FPR)
    fprs, tprs, thresholds_roc = roc_curve(y_true, y_pred_proba)
    
    # Option 1: Threshold for FPR <= 0.05 with maximum TPR
    target_fpr = 0.05
    valid_idx = np.where(fprs <= target_fpr)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(tprs[valid_idx])]
        optimal_fpr_threshold = thresholds_roc[best_idx] if best_idx < len(thresholds_roc) else 0.5
        achieved_fpr = fprs[best_idx]
    else:
        # If no threshold achieves FPR <= 0.05, find the one with minimum FPR
        best_idx = np.argmin(fprs[1:]) + 1  # Skip first point (FPR=0, threshold=inf)
        optimal_fpr_threshold = thresholds_roc[best_idx] if best_idx < len(thresholds_roc) else 0.5
        achieved_fpr = fprs[best_idx]
    
    # Option 2: Threshold for minimum FPR while TPR >= 0.5
    valid_tpr_idx = np.where(tprs >= 0.5)[0]
    if len(valid_tpr_idx) > 0:
        min_fpr_idx = valid_tpr_idx[np.argmin(fprs[valid_tpr_idx])]
        optimal_fpr_tpr50_threshold = thresholds_roc[min_fpr_idx] if min_fpr_idx < len(thresholds_roc) else 0.5
    else:
        optimal_fpr_tpr50_threshold = optimal_fpr_threshold
    
    # Compute metrics at default threshold (0.5)
    print(f"\n{'='*60}")
    print("1. DEFAULT THRESHOLD (0.5)")
    print(f"{'='*60}")
    metrics_default = compute_metrics_at_threshold(y_true, y_pred_proba, 0.5, "Default Threshold")
    
    # Compute metrics at optimal F1 threshold
    print(f"\n{'='*60}")
    print("2. OPTIMAL F1 THRESHOLD")
    print(f"{'='*60}")
    metrics_f1 = compute_metrics_at_threshold(y_true, y_pred_proba, optimal_f1_threshold, "Optimal F1")
    
    # Compute metrics at optimal FPR threshold (FPR <= 0.05)
    print(f"\n{'='*60}")
    print(f"3. OPTIMAL FPR THRESHOLD (target FPR <= {target_fpr})")
    print(f"{'='*60}")
    metrics_fpr = compute_metrics_at_threshold(y_true, y_pred_proba, optimal_fpr_threshold, f"Low FPR (target <= {target_fpr})")
    
    # Compute metrics at threshold for min FPR with TPR >= 0.5
    print(f"\n{'='*60}")
    print("4. MIN FPR WITH TPR >= 0.5")
    print(f"{'='*60}")
    metrics_fpr_tpr50 = compute_metrics_at_threshold(y_true, y_pred_proba, optimal_fpr_tpr50_threshold, "Min FPR (TPR >= 0.5)")
    
    # Summary table
    print(f"\n{'='*60}")
    print("THRESHOLD SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threshold Type':<30} {'Threshold':>10} {'FPR':>8} {'TPR':>8} {'F1':>8} {'Accuracy':>10}")
    print("-" * 80)
    print(f"{'Default (0.5)':<30} {0.5:>10.4f} {metrics_default['FPR']:>8.4f} {metrics_default['TPR']:>8.4f} {metrics_default['F1']:>8.4f} {metrics_default['accuracy']:>10.4f}")
    print(f"{'Optimal F1':<30} {optimal_f1_threshold:>10.4f} {metrics_f1['FPR']:>8.4f} {metrics_f1['TPR']:>8.4f} {metrics_f1['F1']:>8.4f} {metrics_f1['accuracy']:>10.4f}")
    print(f"{'Low FPR (target <= 0.05)':<30} {optimal_fpr_threshold:>10.4f} {metrics_fpr['FPR']:>8.4f} {metrics_fpr['TPR']:>8.4f} {metrics_fpr['F1']:>8.4f} {metrics_fpr['accuracy']:>10.4f}")
    print(f"{'Min FPR (TPR >= 0.5)':<30} {optimal_fpr_tpr50_threshold:>10.4f} {metrics_fpr_tpr50['FPR']:>8.4f} {metrics_fpr_tpr50['TPR']:>8.4f} {metrics_fpr_tpr50['F1']:>8.4f} {metrics_fpr_tpr50['accuracy']:>10.4f}")
    
    # Breakdown by groundtruth
    print(f"\n{'='*60}")
    print("Predictions by Ground Truth Category")
    print(f"{'='*60}")
    for gt in df_valid['groundtruth'].unique():
        df_gt = df_valid[df_valid['groundtruth'] == gt]
        pred_train = (df_gt['y_pred'] == 1).sum()
        pred_val = (df_gt['y_pred'] == 0).sum()
        avg_proba = df_gt['y_pred_proba'].mean()
        print(f"\n{gt.upper()} (n={len(df_gt)}):")
        print(f"  Predicted as Train: {pred_train} ({100*pred_train/len(df_gt):.1f}%)")
        print(f"  Predicted as Val:   {pred_val} ({100*pred_val/len(df_gt):.1f}%)")
        print(f"  Avg probability (train): {avg_proba:.4f}")
    
    return df_valid


def main():
    print("="*60)
    print("Classifier Evaluation on Generated Names")
    print("="*60)
    print(f"Classifier: {CLASSIFIER_FILE}")
    print(f"Training samples: {DF_TEMP_SUB_FILE}")
    print(f"Computed LL: {COMPUTED_LL_FILE}")
    print(f"Output: {OUTPUT_RESULTS_FILE}")
    print("="*60)
    
    # Load classifier
    clf = load_classifier(CLASSIFIER_FILE)
    
    # Load training samples
    df_train_samples, training_names = load_training_samples(DF_TEMP_SUB_FILE)
    
    # Load and prepare features
    df_eval, df_excluded = load_and_prepare_features(COMPUTED_LL_FILE, training_names)
    
    if df_eval is None:
        print("Failed to prepare features. Exiting.")
        return
    
    # Evaluate classifier
    df_results = evaluate_classifier(clf, df_eval, FEATURE_COLUMNS)
    
    if df_results is not None:
        # Save results
        df_results.to_csv(OUTPUT_RESULTS_FILE, index=False)
        print(f"\nSaved evaluation results to: {OUTPUT_RESULTS_FILE}")
        
        # Save False Positives (val/other predicted as train) - ranked by score (highest first)
        df_fp = df_results[(df_results['y_true'] == 0) & (df_results['y_pred'] == 1)].copy()
        df_fp = df_fp.sort_values('y_pred_proba', ascending=False)
        fp_file = OUTPUT_RESULTS_FILE.replace('.csv', '_false_positives.csv')
        df_fp.to_csv(fp_file, index=False)
        print(f"Saved {len(df_fp)} False Positives (val predicted as train) to: {fp_file}")
        
        # Save False Negatives (train predicted as val) - ranked by score (lowest first)
        df_fn = df_results[(df_results['y_true'] == 1) & (df_results['y_pred'] == 0)].copy()
        df_fn = df_fn.sort_values('y_pred_proba', ascending=True)
        fn_file = OUTPUT_RESULTS_FILE.replace('.csv', '_false_negatives.csv')
        df_fn.to_csv(fn_file, index=False)
        print(f"Saved {len(df_fn)} False Negatives (train predicted as val) to: {fn_file}")
        
        # Also print some examples
        print(f"\n{'='*60}")
        print("Sample Predictions")
        print(f"{'='*60}")
        
        # High confidence train predictions
        print("\nTop 10 names predicted as TRAIN (highest probability):")
        top_train = df_results.nlargest(10, 'y_pred_proba')[['name', 'groundtruth', 'y_pred_proba', 'y_pred']]
        print(top_train.to_string())
        
        # High confidence val predictions
        print("\nTop 10 names predicted as VAL/OTHER (lowest probability):")
        top_val = df_results.nsmallest(10, 'y_pred_proba')[['name', 'groundtruth', 'y_pred_proba', 'y_pred']]
        print(top_val.to_string())
        
        # Print False Positives summary
        print(f"\n{'='*60}")
        print(f"False Positives (val/other predicted as train): {len(df_fp)}")
        print(f"{'='*60}")
        if len(df_fp) > 0:
            print("\nTop 10 False Positives (highest confidence mistakes):")
            print(df_fp[['name', 'groundtruth', 'y_pred_proba']].head(10).to_string())
        
        # Print False Negatives summary
        print(f"\n{'='*60}")
        print(f"False Negatives (train predicted as val): {len(df_fn)}")
        print(f"{'='*60}")
        if len(df_fn) > 0:
            print("\nTop 10 False Negatives (lowest confidence mistakes):")
            print(df_fn[['name', 'groundtruth', 'y_pred_proba']].head(10).to_string())


if __name__ == "__main__":
    main()
