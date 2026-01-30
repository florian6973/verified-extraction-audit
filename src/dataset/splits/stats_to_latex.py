import pandas as pd
from pathlib import Path
import argparse

SPLITS = [
    'train_1', 'val_1', 'train_10', 'val_10', 'train', 'val', 'test'
]

def get_stats(df):
    num_rows = len(df)
    num_patients = df['subject_id'].nunique()
    return num_rows, num_patients

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_dir', type=Path, default=Path('data/processed/splits'))
    parser.add_argument('--output', type=Path, default=Path('outputs/splits/stats_table.tex'))
    args = parser.parse_args()

    stats = []
    for split in SPLITS:
        path = args.splits_dir / f'{split}.parquet'
        df = pd.read_parquet(path)
        num_rows, num_patients = get_stats(df)
        stats.append((split, num_rows, num_patients))

    # Create a DataFrame for the stats
    stats_df = pd.DataFrame(stats, columns=['Split', 'Rows', 'Patients'])

    # Create a multi-index for the first column
    stats_df['Percentage'] = stats_df['Split'].apply(lambda x: '10\\%' if '10' in x else '1\\%' if '1' in x else '100\\%')
    # Remove '_x' suffix from split names
    stats_df['Split'] = stats_df['Split'].str.replace('_10', '')
    stats_df['Split'] = stats_df['Split'].str.replace('_1', '')
    stats_df['Split'] = stats_df['Split'].str.capitalize()
    stats_df.set_index(['Percentage', 'Split'], inplace=True)

    # Convert DataFrame to LaTeX table with a label and visible borders
    latex_table = stats_df.to_latex(index=True, label='tab:dataset_stats', caption='Dataset Statistics', column_format='llrr')

    # Add necessary LaTeX packages and center the table
    latex_table = '\\usepackage{booktabs}\n\\usepackage{multirow}\n' + latex_table
    latex_table = latex_table.replace('\\begin{tabular}', '\\begin{center}\\begin{tabular}')
    latex_table = latex_table.replace('\\end{tabular}', '\\end{tabular}\\end{center}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table written to {args.output}")

if __name__ == '__main__':
    main()
