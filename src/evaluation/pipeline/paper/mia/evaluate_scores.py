import argparse
import os
import subprocess
from src.evaluation.pipeline.experimental.config_loader import load_config
from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir

parser = argparse.ArgumentParser(description='Evaluate scores for MIA')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

model = config['filters']['model']
dataset_size = config['filters']['dataset_size']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']

column_name = "score_oof_member_proba"
groundtruth_column = "groundtruth"
data_path = os.path.join(get_output_dir(config), f"all_names_ll_computed_with_scores.csv")
data_path_src = os.path.join(get_output_dir(config), f"scores_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}_p.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from src.evaluation.pipeline.paper.mia.name_filter import name_mask

# Load data
if not os.path.exists(data_path):
    print(f"Warning: {data_path} not found. Skipping.")
    # exit(1)
    exit()
df = pd.read_csv(data_path)

df_src = pd.read_csv(data_path_src)
total_train_names = df_src[df_src['split_x'] == 'train'].count()[0]
print(f"Total train names: {total_train_names}")

# Check if score column exists and has valid values
if column_name not in df.columns:
    print(f"Error: Column '{column_name}' not found in data.")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Check if folder is in threshold_fpr5_results.csv and get avg_threshold
output_dir = get_output_dir(config)
folder_name = os.path.basename(output_dir)
# threshold_fpr5_path = os.path.join(os.path.dirname(output_dir), 'threshold_fpr5_results.csv')
threshold_fpr5_path = os.path.join(os.path.dirname(output_dir), 'threshold_extracted_fpr0.05_results.csv')
avg_thr_from_fpr5 = None

if os.path.exists(threshold_fpr5_path):
    try:
        df_fpr5 = pd.read_csv(threshold_fpr5_path)
        if 'directory' in df_fpr5.columns and 'avg_threshold' in df_fpr5.columns:
            matching_rows = df_fpr5[df_fpr5['directory'] == folder_name]
            if len(matching_rows) > 0:
                # Take the first match (or average if multiple)
                avg_thr_from_fpr5 = matching_rows.iloc[0]['avg_threshold']
                print(f"\nFound folder '{folder_name}' in threshold_fpr5_results.csv")
                print(f"avg_threshold from FPR5: {avg_thr_from_fpr5:.6f}")
            else:
                print(f"\nFolder '{folder_name}' not found in threshold_fpr5_results.csv")
        else:
            print(f"\nthreshold_fpr5_results.csv missing required columns")
    except Exception as e:
        print(f"\nError reading threshold_fpr5_results.csv: {e}")
else:
    print(f"\nthreshold_fpr5_results.csv not found at: {threshold_fpr5_path}")

def evaluate_metrics(df_input, output_suffix, generate_plots=True, filter_groundtruth=False, avg_thr=None, filter_language=True):
    """
    Evaluate metrics for MIA scores.
    
    Args:
        df_input: Input dataframe
        output_suffix: Suffix for output CSV file (e.g., '_metrics_by_threshold' or '_metrics_by_threshold_without_others')
        generate_plots: Whether to generate plots (only for first pass)
        filter_groundtruth: If True, filter to only 'train' and 'val' groundtruth values
        avg_thr: Average threshold from FPR5 results (optional)
        filter_language: If True, filter to only likely names using language heuristics (default: True)
    """
    # Start with input dataframe
    df_work = df_input.copy()
    
    # Apply language filter if requested
    if filter_language:
        if 'value' in df_work.columns:
            name_mask_result = name_mask(df_work, column='value')
            rows_before_lang = len(df_work)
            df_work = df_work[name_mask_result].copy()
            rows_after_lang = len(df_work)
            print(f"\nLanguage filter applied: {rows_before_lang} -> {rows_after_lang} rows ({rows_before_lang - rows_after_lang} filtered out)")
        else:
            print(f"\nWarning: 'value' column not found, skipping language filter")
    
    # Filter if requested
    if filter_groundtruth:
        rows_before_gt = len(df_work)
        df_work = df_work[df_work['groundtruth'].isin(['train', 'val'])].copy()
        print(f"\n{'='*80}")
        print("PASS 2: Filtered to train/val only (for more accurate FPR)")
        print(f"{'='*80}")
        print(f"Original rows: {rows_before_gt}, Filtered rows: {len(df_work)}")
    else:
        print(f"\n{'='*80}")
        print("PASS 1: All data")
        print(f"{'='*80}")
    
    df_work['y_true'] = (df_work[groundtruth_column] == 'train').astype(int)
    
    # Remove rows with missing scores
    df_valid = df_work.dropna(subset=[column_name]).copy()
    print(f"Total rows: {len(df_work)}, Valid rows (with scores): {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("Error: No valid scores found in the data.")
        return
    
    y_true = df_valid['y_true'].values
    y_scores = df_valid[column_name].values
    
    # Compute ROC curve to get TPR and FPR for different thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Compute AUC
    auc_score = roc_auc_score(y_true, y_scores)
    print(f"\nAUC-ROC: {auc_score:.4f}")
    
    # Handle dimension mismatch: fpr and tpr have len(thresholds) + 1
    # The last point corresponds to threshold = inf (classify all as negative)
    # Filter out inf and NaN values from thresholds for plotting
    valid_mask = np.isfinite(thresholds)
    thresholds_clean = thresholds[valid_mask]
    tpr_for_plot = tpr[:len(thresholds)][valid_mask]
    fpr_for_plot = fpr[:len(thresholds)][valid_mask]
    
    # ============================================================
    # Plot TPR and FPR as a function of threshold (only for first pass)
    # ============================================================
    if generate_plots:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: TPR and FPR vs Threshold
        plt.subplot(1, 2, 1)
        plt.plot(thresholds_clean, tpr_for_plot, 'b-', label='TPR (True Positive Rate)', linewidth=2)
        plt.plot(thresholds_clean, fpr_for_plot, 'r-', label='FPR (False Positive Rate)', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('TPR and FPR vs Threshold', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        if len(thresholds_clean) > 0:
            plt.xlim([thresholds_clean.min(), thresholds_clean.max()])
        plt.ylim([0, 1])
        
        # Plot 2: ROC Curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(data_path.replace('.csv', '_tpr_fpr_curves.png'), dpi=150, bbox_inches='tight')
        print(f"\nSaved plots to: {data_path.replace('.csv', '_tpr_fpr_curves.png')}")
        plt.close()
    
    # ============================================================
    # Display Confusion Matrices for different thresholds
    # ============================================================
    print("\n" + "="*80)
    print("CONFUSION MATRICES AT DIFFERENT THRESHOLDS")
    print("="*80)
    
    # Select thresholds to display
    # Include: 0.3, 0.4, 0.5, 0.6, 0.7, and optimal threshold (Youden's J)
    youden_j = tpr_for_plot - fpr_for_plot
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds_clean[optimal_idx]
    
    thresholds_to_display = [0.3, 0.4, 0.5, 0.6, 0.7]
    if optimal_threshold not in thresholds_to_display:
        thresholds_to_display.append(optimal_threshold)
    thresholds_to_display = sorted(set(thresholds_to_display))
    
    # Store metrics for each threshold
    metrics_data = []
    
    for thr in thresholds_to_display:
        y_pred = (y_scores >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle edge cases where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            # If only one class predicted, adjust
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 1:
                    TP = (y_true == 1).sum()
                    FP = (y_true == 0).sum()
                    TN, FN = 0, 0
                else:
                    TN = (y_true == 0).sum()
                    FN = (y_true == 1).sum()
                    TP, FP = 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
            cm = np.array([[TN, FP], [FN, TP]])
        
        # Calculate metrics
        tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = tpr_val
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # Mark optimal threshold
        threshold_label = f"{thr:.3f}"
        if abs(thr - optimal_threshold) < 0.001:
            threshold_label += " (OPTIMAL - Youden's J)"
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold_label}")
        print(f"{'='*80}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0 (Val)    1 (Train)")
        print(f"Actual 0 (Val)   {TN:8d}   {FP:8d}")
        print(f"Actual 1 (Train) {FN:8d}   {TP:8d}")
        print(f"\nMetrics:")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f} (TPR)")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  TPR:       {tpr_val:.4f} (True Positive Rate)")
        print(f"  FPR:       {fpr_val:.4f} (False Positive Rate)")
        print(f"  Specificity: {specificity:.4f} (TNR)")
        print(f"  Total train names: {total_train_names}")
        print(f"  Total recall: {TP/total_train_names:.4f}")
        
        # Store metrics for CSV
        is_optimal = abs(thr - optimal_threshold) < 0.001
        metrics_data.append({
            'threshold': thr,
            'is_optimal': is_optimal,
            'TPR': tpr_val,
            'FPR': fpr_val,
            'recall': recall,
            'TN': TN,
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'total_train_names': total_train_names,
            'total_recall': TP/total_train_names if total_train_names > 0 else 0
        })
    
    # Also compute metrics at the default threshold (0.6)
    default_thr = 0.6
    if default_thr not in thresholds_to_display:
        y_pred = (y_scores >= default_thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 1:
                    TP = (y_true == 1).sum()
                    FP = (y_true == 0).sum()
                    TN, FN = 0, 0
                else:
                    TN = (y_true == 0).sum()
                    FN = (y_true == 1).sum()
                    TP, FP = 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
            cm = np.array([[TN, FP], [FN, TP]])
        
        tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = tpr_val
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {default_thr:.3f} (DEFAULT)")
        print(f"{'='*80}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0 (Val)    1 (Train)")
        print(f"Actual 0 (Val)   {TN:8d}   {FP:8d}")
        print(f"Actual 1 (Train) {FN:8d}   {TP:8d}")
        print(f"\nMetrics:")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f} (TPR)")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  TPR:       {tpr_val:.4f} (True Positive Rate)")
        print(f"  FPR:       {fpr_val:.4f} (False Positive Rate)")
        print(f"  Specificity: {specificity:.4f} (TNR)")
        
        # Store metrics for CSV if not already stored
        if default_thr not in thresholds_to_display:
            metrics_data.append({
                'threshold': default_thr,
                'is_optimal': False,
                'TPR': tpr_val,
                'FPR': fpr_val,
                'recall': recall,
                'TN': TN,
                'TP': TP,
                'FN': FN,
                'FP': FP,
                'precision': precision,
                'f1_score': f1,
                'accuracy': accuracy,
                'specificity': specificity,
                'total_train_names': total_train_names,
                'total_recall': TP/total_train_names if total_train_names > 0 else 0
            })
    
    # Also compute metrics at avg_thr threshold if available
    if avg_thr is not None and avg_thr not in thresholds_to_display and default_thr != avg_thr:
        y_pred = (y_scores >= avg_thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 1:
                    TP = (y_true == 1).sum()
                    FP = (y_true == 0).sum()
                    TN, FN = 0, 0
                else:
                    TN = (y_true == 0).sum()
                    FN = (y_true == 1).sum()
                    TP, FP = 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
            cm = np.array([[TN, FP], [FN, TP]])
        
        tpr_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = tpr_val
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {avg_thr:.6f} (AVG_THR from FPR5)")
        print(f"{'='*80}")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0 (Val)    1 (Train)")
        print(f"Actual 0 (Val)   {TN:8d}   {FP:8d}")
        print(f"Actual 1 (Train) {FN:8d}   {TP:8d}")
        print(f"\nMetrics:")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f} (TPR)")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  TPR:       {tpr_val:.4f} (True Positive Rate)")
        print(f"  FPR:       {fpr_val:.4f} (False Positive Rate)")
        print(f"  Specificity: {specificity:.4f} (TNR)")
        print(f"  Total train names: {total_train_names}")
        print(f"  Total recall: {TP/total_train_names:.4f}")
        
        metrics_data.append({
            'threshold': avg_thr,
            'is_optimal': False,
            'TPR': tpr_val,
            'FPR': fpr_val,
            'recall': recall,
            'TN': TN,
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'total_train_names': total_train_names,
            'total_recall': TP/total_train_names if total_train_names > 0 else 0
        })
    
    print("\n" + "="*80)
    
    # Save metrics to CSV
    df_metrics = pd.DataFrame(metrics_data)
    # Sort by threshold
    df_metrics = df_metrics.sort_values('threshold').reset_index(drop=True)
    # Save to CSV
    output_csv_path = data_path.replace('.csv', output_suffix + '.csv')
    df_metrics.to_csv(output_csv_path, index=False)
    print(f"\nSaved metrics by threshold to: {output_csv_path}")
    print(f"Metrics for {len(df_metrics)} thresholds saved.")
    print(f"\nMetrics summary:")
    print(df_metrics[['threshold', 'is_optimal', 'TPR', 'FPR', 'recall', 'total_recall', 'TP', 'FP', 'TN', 'FN']].to_string(index=False))

# Pass 1: All data (original behavior)
evaluate_metrics(df, '_metrics_by_threshold', generate_plots=True, filter_groundtruth=False, avg_thr=avg_thr_from_fpr5)

# Pass 2: Filtered to train/val only (for more accurate FPR)
evaluate_metrics(df, '_metrics_by_threshold_without_others', generate_plots=False, filter_groundtruth=True, avg_thr=avg_thr_from_fpr5)

# Run theoretical evaluation with avg_thr from FPR5 if available
if avg_thr_from_fpr5 is not None and args.config is not None:
    print(f"\n{'='*80}")
    print(f"Running theoretical evaluation with threshold from FPR5: {avg_thr_from_fpr5:.6f}")
    print(f"{'='*80}")
    
    # Find attack_curves.py
    # evaluate_scores.py is in experimental/mia/
    # attack_curves.py is in pipeline/
    script_dir = os.path.dirname(os.path.abspath(__file__))  # experimental/mia/
    experimental_dir = os.path.dirname(script_dir)  # experimental/
    pipeline_dir = os.path.dirname(experimental_dir)  # pipeline/
    pipeline_attack_script = os.path.join(pipeline_dir, 'attack_curves.py')
    
    if os.path.exists(pipeline_attack_script):
        cmd = ['python', pipeline_attack_script, '--config', args.config, '--tau', str(avg_thr_from_fpr5)]
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=pipeline_dir,
                check=True,
                capture_output=False
            )
            print(f"\n✓ Theoretical evaluation completed successfully with tau={avg_thr_from_fpr5:.6f}")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error running theoretical evaluation: {e}")
        except Exception as e:
            print(f"\n✗ Unexpected error running theoretical evaluation: {e}")
    else:
        print(f"Warning: attack_curves.py not found at: {pipeline_attack_script}")
        print(f"  Script directory: {script_dir}")
        print(f"  Experimental directory: {experimental_dir}")
        print(f"  Pipeline directory: {pipeline_dir}")
