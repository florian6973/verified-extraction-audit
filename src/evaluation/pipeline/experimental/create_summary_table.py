"""
Create summary tables for theoretical curves.
Tables show FP and TP at specific budgets (100, 1000, 10000, 100000, 10^6, 10^7).
One table per (dataset_size, model_size) group.
Columns are different model configurations at optimal tau, with FP and TP subcolumns.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from summary_theo_plot.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summary_theo_plot import (
    BASE_DIR,
    THRESHOLD_CSV,
    load_threshold_data,
    collect_theoretical_data,
    get_dataset_size_label
)


# Target budgets for the table
TARGET_BUDGETS = [100, 1000, 10000, 100000, 1e6, 1e7]


def get_value_at_budget(budgets, values, target_budget):
    """
    Get the value at a specific budget by finding the closest budget point.
    
    Args:
        budgets: Array of budget values
        values: Array of corresponding values
        target_budget: Target budget to find
    
    Returns:
        Value at the closest budget point, or NaN if not found
    """
    if len(budgets) == 0 or len(values) == 0:
        return np.nan
    
    # Find the index of the closest budget
    budget_idx = np.argmin(np.abs(budgets - target_budget))
    return values[budget_idx]


def create_summary_tables(theory_data, output_dir=None, use_filtered_exp=False):
    """
    Create summary tables with FP and TP at specific budgets.
    One table per (dataset_size, model_size) group.
    
    Args:
        theory_data: List of dictionaries with theoretical data
        output_dir: Directory to save tables (default: BASE_DIR)
        use_filtered_exp: Whether to use filtered experimental data
    """
    if output_dir is None:
        output_dir = BASE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    print(f"\nFound {len(ds_model_list)} unique (dataset_size, model_size) combinations")
    
    # Create one table per group
    for ds_size, model in ds_model_list:
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        
        if len(group_data) == 0:
            continue
        
        print(f"\nCreating table for ({ds_size}, {model}) with {len(group_data)} configurations...")
        
        # Create a list to store rows
        table_rows = []
        
        # For each target budget, create a row
        for target_budget in TARGET_BUDGETS:
            row_data = {'Budget': target_budget}
            
            # For each model configuration, extract FP and TP at this budget
            for data in group_data:
                n_ep = data['n_epochs']
                pii_rate = data['pii_rate']
                tau_val = data['tau']
                
                # Create model label
                if n_ep in [2, 3]:
                    ep_label = "no_overfit"
                elif n_ep == 10:
                    ep_label = "overfit"
                else:
                    ep_label = f"ep{n_ep}"
                
                pii_pct = int(pii_rate * 100)
                model_label = f"{ep_label}_PII{pii_pct}%_tau{tau_val:.4f}"
                
                # Get TP at this budget (from recall * total_train_names)
                tp = np.nan
                if data['total_train_names'] is not None and data['recall'] is not None:
                    recall_at_budget = get_value_at_budget(
                        data['budget'], data['recall'], target_budget
                    )
                    if not np.isnan(recall_at_budget):
                        tp = recall_at_budget * data['total_train_names']
                
                # Get FP at this budget (from cumulative_fp)
                fp = np.nan
                if data['cumulative_fp'] is not None:
                    fp = get_value_at_budget(
                        data['budget'], data['cumulative_fp'], target_budget
                    )
                
                # Add to row with subcolumns
                row_data[f"{model_label}_FP"] = fp
                row_data[f"{model_label}_TP"] = tp
            
            table_rows.append(row_data)
        
        # Create DataFrame
        df_table = pd.DataFrame(table_rows)
        
        # Format budget column for display as $10^x$
        def format_budget_col(x):
            if x == 0:
                return "$10^0$"
            exponent = int(np.log10(x))
            return f"$10^{{{exponent}}}$"
        
        df_table['Budget'] = df_table['Budget'].apply(format_budget_col)
        
        # Reorder columns: Budget first, then each model with FP/TP pairs
        model_configs = []
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            tau_val = data['tau']
            
            if n_ep in [2, 3]:
                ep_label = "no_overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep{n_ep}"
            
            pii_pct = int(pii_rate * 100)
            model_label = f"{ep_label}_PII{pii_pct}%_tau{tau_val:.4f}"
            model_configs.append(model_label)
        
        # Create column order
        col_order = ['Budget']
        for model_label in model_configs:
            col_order.extend([f"{model_label}_FP", f"{model_label}_TP"])
        
        # Reorder columns (only include columns that exist)
        existing_cols = [col for col in col_order if col in df_table.columns]
        df_table = df_table[existing_cols]
        
        # Format numeric columns
        # Use iloc to avoid issues with duplicate column names
        for col_idx, col in enumerate(df_table.columns):
            if col != 'Budget':
                col_data = df_table.iloc[:, col_idx]
                # Check if it's a numeric column
                if isinstance(col_data, pd.Series) and col_data.dtype in [np.float64, np.float32]:
                    df_table.iloc[:, col_idx] = col_data.apply(
                        lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
                    )
        
        # Save table
        ds_label = get_dataset_size_label(ds_size)
        suffix = "_filtered" if use_filtered_exp else ""
        output_file = os.path.join(
            output_dir, 
            f"summary_table_{ds_label}_{model}{suffix}.csv"
        )
        
        df_table.to_csv(output_file, index=False)
        print(f"Saved table to: {output_file}")
        
        # Also print a formatted version
        print(f"\nTable for ({ds_label}, {model}):")
        print(df_table.to_string(index=False))
        print()


def export_to_latex(df_table, group_data, ds_label, model, output_file):
    """
    Export DataFrame to LaTeX format with proper multi-column headers.
    
    Args:
        df_table: DataFrame with multi-index columns (Budget index, model configs with FP/TP)
        group_data: List of data dictionaries for this group (for model labels)
        ds_label: Dataset size label
        model: Model size
        output_file: Output file path for LaTeX
    """
    # Escape special LaTeX characters in model labels
    def escape_latex(text):
        if isinstance(text, str):
            # Replace underscores with escaped underscores for subscripts
            text = text.replace('_', r'\_')
            # Escape other special characters (but not % since we handle it separately)
            text = text.replace('&', r'\&')
            text = text.replace('#', r'\#')
            return text
        return text
    
    # Start LaTeX table
    latex_lines = []
    latex_lines.append(r"% Requires: \usepackage{booktabs}")
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")  # Use smaller font for wide tables
    latex_lines.append(r"\caption{FP and TP at different budgets for " + 
                       escape_latex(f"{ds_label}, {model}") + r"}")
    latex_lines.append(r"\label{tab:" + f"{ds_label}_{model}".replace(' ', '_').replace('-', '_') + r"}")
    
    # Count columns: 1 for Budget + 2 (FP/TP) per model config
    n_models = len(group_data)
    n_cols = 1 + 2 * n_models
    
    # Build column specification
    col_spec = "l"  # Budget column (left-aligned)
    for _ in range(n_models):
        col_spec += "|cc"  # Two centered columns per model (FP and TP)
    
    latex_lines.append(r"\begin{tabular}{" + col_spec + r"}")
    latex_lines.append(r"\toprule")
    
    # First header row: model configurations (spanning 2 columns each)
    header_row1 = r"Budget"
    for data in group_data:
        n_ep = data['n_epochs']
        pii_rate = data['pii_rate']
        tau_val = data['tau']
        
        if n_ep in [2, 3]:
            ep_label = "no overfit"
        elif n_ep == 10:
            ep_label = "overfit"
        else:
            ep_label = f"ep {n_ep}"
        
        pii_pct = int(pii_rate * 100)
        if tau_val < 0.01:
            tau_str = f"{tau_val:.4f}"
        elif tau_val < 1:
            tau_str = f"{tau_val:.3f}"
        else:
            tau_str = f"{tau_val:.2f}"
        
        # Get total_train_names
        total_names = data.get('total_train_names', None)
        if total_names is not None:
            names_str = f"$N={int(total_names)}$"
        else:
            names_str = r"$N=$N/A"
        
        # Break model label into four lines for narrower table
        # Use \shortstack to stack the four lines
        model_label = (r"\shortstack{" + 
                      escape_latex(ep_label) + r" \\ " +
                      f"PII {pii_pct}" + r"\%" + r" \\ " +
                      f"$\\tau={tau_str}$" + r" \\ " +
                      names_str + r"}")
        header_row1 += r" & \multicolumn{2}{c|}{" + model_label + r"}"
    
    header_row1 += r" \\"
    latex_lines.append(header_row1)
    latex_lines.append(r"\cmidrule{2-" + str(n_cols) + r"}")
    
    # Second header row: FP and TP for each model
    header_row2 = r""
    for _ in range(n_models):
        header_row2 += r" & FP & TP"
    header_row2 += r" \\"
    latex_lines.append(header_row2)
    latex_lines.append(r"\midrule")
    
    # Data rows - iterate through columns in DataFrame order
    for budget_idx in df_table.index:
        # Budget is already formatted as $10^x$, use it directly
        budget_str = str(budget_idx)
        # If it's already in LaTeX math format, use as-is; otherwise escape
        if budget_str.startswith('$') and budget_str.endswith('$'):
            row = budget_str
        else:
            row = escape_latex(budget_str)
        # Get values for each column pair (FP, TP) in order
        for col in df_table.columns:
            value = df_table.loc[budget_idx, col]
            # Handle case where MultiIndex returns a Series (duplicate columns)
            if isinstance(value, pd.Series):
                # If Series, take the first value (they should all be the same)
                value = value.iloc[0] if len(value) > 0 else np.nan
            # Check if value is NaN
            if pd.isna(value):
                row += r" & N/A"
            else:
                # Round to the nearest integer
                rounded_value = int(np.round(value))
                row += f" & {rounded_value}"
        row += r" \\"
        latex_lines.append(row)
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))


def create_summary_tables_multiindex(theory_data, output_dir=None, use_filtered_exp=False):
    """
    Create summary tables with multi-level columns (model configs with FP/TP subcolumns).
    One table per (dataset_size, model_size) group.
    
    Args:
        theory_data: List of dictionaries with theoretical data
        output_dir: Directory to save tables (default: BASE_DIR)
        use_filtered_exp: Whether to use filtered experimental data
    """
    if output_dir is None:
        output_dir = BASE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique (dataset_size, model_size) combinations
    ds_model_groups = set()
    for data in theory_data:
        ds_model_groups.add((data['dataset_size'], data['model_size']))
    
    ds_model_list = sorted(list(ds_model_groups))
    
    print(f"\nFound {len(ds_model_list)} unique (dataset_size, model_size) combinations")
    
    # Create one table per group
    for ds_size, model in ds_model_list:
        # Filter data for this group
        group_data = [d for d in theory_data 
                     if d['dataset_size'] == ds_size and d['model_size'] == model]
        
        if len(group_data) == 0:
            continue
        
        print(f"\nCreating multi-index table for ({ds_size}, {model}) with {len(group_data)} configurations...")
        
        # Sort group_data by (pii_rate, n_epochs) for consistent column ordering
        group_data = sorted(group_data, key=lambda x: (x['pii_rate'], x['n_epochs']))
        
        # Create a list to store rows
        table_rows = []
        
        # For each target budget, create a row
        for target_budget in TARGET_BUDGETS:
            row_data = {}
            
            # For each model configuration, extract FP and TP at this budget
            for data in group_data:
                n_ep = data['n_epochs']
                pii_rate = data['pii_rate']
                tau_val = data['tau']
                
                # Create model label (cleaner format)
                if n_ep in [2, 3]:
                    ep_label = "no_overfit"
                elif n_ep == 10:
                    ep_label = "overfit"
                else:
                    ep_label = f"ep{n_ep}"
                
                pii_pct = int(pii_rate * 100)
                # Format tau for label (shorter format)
                if tau_val < 0.01:
                    tau_str = f"{tau_val:.4f}"
                elif tau_val < 1:
                    tau_str = f"{tau_val:.3f}"
                else:
                    tau_str = f"{tau_val:.2f}"
                
                model_label = f"{ep_label}_PII{pii_pct}%_τ{tau_str}"
                
                # Get TP at this budget (from recall * total_train_names)
                tp = np.nan
                if data['total_train_names'] is not None and data['recall'] is not None:
                    recall_at_budget = get_value_at_budget(
                        data['budget'], data['recall'], target_budget
                    )
                    if not np.isnan(recall_at_budget):
                        tp = recall_at_budget * data['total_train_names']
                
                # Get FP at this budget (from cumulative_fp)
                fp = np.nan
                if data['cumulative_fp'] is not None:
                    fp = get_value_at_budget(
                        data['budget'], data['cumulative_fp'], target_budget
                    )
                
                # Add to row with tuple keys for multi-index
                row_data[(model_label, 'FP')] = fp
                row_data[(model_label, 'TP')] = tp
            
            row_data['Budget'] = target_budget
            table_rows.append(row_data)
        
        # Create DataFrame
        df_table = pd.DataFrame(table_rows)
        
        # Set Budget as index temporarily, then create multi-index columns
        df_table = df_table.set_index('Budget')
        
        # Create multi-index columns in the same order as group_data
        columns_list = []
        for data in group_data:
            n_ep = data['n_epochs']
            pii_rate = data['pii_rate']
            tau_val = data['tau']
            
            if n_ep in [2, 3]:
                ep_label = "no_overfit"
            elif n_ep == 10:
                ep_label = "overfit"
            else:
                ep_label = f"ep{n_ep}"
            
            pii_pct = int(pii_rate * 100)
            # Format tau for label
            if tau_val < 0.01:
                tau_str = f"{tau_val:.4f}"
            elif tau_val < 1:
                tau_str = f"{tau_val:.3f}"
            else:
                tau_str = f"{tau_val:.2f}"
            
            model_label = f"{ep_label}_PII{pii_pct}%_τ{tau_str}"
            columns_list.append((model_label, 'FP'))
            columns_list.append((model_label, 'TP'))
        
        # Reorder columns
        df_table = df_table.reindex(columns=pd.MultiIndex.from_tuples(columns_list))
        
        # Format budget index for display as $10^x$
        def format_budget(x):
            # Convert to power of 10 format
            if x == 0:
                return "$10^0$"
            # Calculate the exponent
            exponent = int(np.log10(x))
            return f"$10^{{{exponent}}}$"
        
        df_table.index = df_table.index.map(format_budget)
        
        # Keep numeric values for better sorting/analysis, but format for display
        # Save both formatted and unformatted versions
        
        # Save unformatted version (for analysis)
        ds_label = get_dataset_size_label(ds_size)
        suffix = "_filtered" if use_filtered_exp else ""
        output_file = os.path.join(
            output_dir, 
            f"summary_table_{ds_label}_{model}{suffix}.csv"
        )
        
        df_table.to_csv(output_file)
        print(f"Saved table to: {output_file}")
        
        # Create formatted version for display
        # Format by working with underlying values to avoid MultiIndex assignment issues
        def format_value(x):
            """Format a single scalar value."""
            if pd.isna(x):
                return "N/A"
            try:
                return f"{float(x):.2f}"
            except (ValueError, TypeError):
                return str(x)
        
        # Convert to numpy array, format, then create new DataFrame
        values = df_table.values
        formatted_values = np.array([[format_value(val) for val in row] for row in values])
        df_table_formatted = pd.DataFrame(
            formatted_values,
            index=df_table.index,
            columns=df_table.columns
        )
        
        # Also print a formatted version
        print(f"\nTable for ({ds_label}, {model}):")
        print(df_table_formatted.to_string())
        print()
        
        # Export to LaTeX
        latex_output_file = os.path.join(
            output_dir, 
            f"summary_table_{ds_label}_{model}{suffix}.tex"
        )
        export_to_latex(df_table, group_data, ds_label, model, latex_output_file)
        print(f"Saved LaTeX table to: {latex_output_file}")


def main():
    """Main function to collect data and create tables."""
    parser = argparse.ArgumentParser(description='Create summary tables for theoretical curves')
    parser.add_argument('--use-filtered-exp', action='store_true',
                        help='Use filtered experimental metrics (train/val only) instead of all values')
    parser.add_argument('--multi-index', action='store_true',
                        help='Create tables with multi-index columns (model configs with FP/TP subcolumns)')
    args = parser.parse_args()
    
    use_filtered_exp = args.use_filtered_exp
    
    print("="*80)
    print("Creating Summary Tables for Theoretical Curves")
    if use_filtered_exp:
        print("Using filtered experimental metrics (train/val only)")
    else:
        print("Using all experimental metrics")
    print("="*80)
    
    # Load threshold data
    print("\nLoading threshold data...")
    threshold_map = load_threshold_data(THRESHOLD_CSV)
    
    if len(threshold_map) == 0:
        print("ERROR: No threshold data found. Cannot proceed.")
        return 1
    
    # Load threshold DataFrame for file paths
    threshold_df = None
    if os.path.exists(THRESHOLD_CSV):
        try:
            threshold_df = pd.read_csv(THRESHOLD_CSV)
        except Exception as e:
            print(f"Warning: Could not load threshold CSV as DataFrame: {e}")
    
    # Collect theoretical data
    print("\nCollecting theoretical curve data...")
    theory_data = collect_theoretical_data(BASE_DIR, threshold_map, threshold_df=threshold_df, use_filtered_exp=use_filtered_exp)
    
    print(f"\nCollected theoretical data for {len(theory_data)} directories")
    
    # Create tables (use multi-index by default as it matches the requirement better)
    print("\nCreating summary tables...")
    create_summary_tables_multiindex(theory_data, use_filtered_exp=use_filtered_exp)
    
    # Also create simple version if requested
    if not args.multi_index:
        create_summary_tables(theory_data, use_filtered_exp=use_filtered_exp)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
