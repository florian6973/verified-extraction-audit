from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir, get_src_ll_file_base, get_src_ll_file
from src.evaluation.pipeline.experimental.config_loader import load_config
import argparse

import pandas as pd
import os

from src._repo import REPO_ROOT
parser = argparse.ArgumentParser(description='Prep data for MIA')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

dataset_size = config['filters']['dataset_size']
model = config['filters']['model']
pii_rate = config['filters']['pii_rate']
n_epochs = config['filters']['n_epochs']

data_path_ft = get_src_ll_file(config)
data_path_qi = get_src_ll_file_base(config)
# data_path_ft = f(REPO_ROOT + '/outputs/pii_leakage/pipeline/ll_all_output_False_{model}_{dataset_size}_batch.csv'
# data_path_qi = f(REPO_ROOT + '/outputs/pii_leakage/pipeline/ll_all_output_True_{model}_{dataset_size}_batch.csv'

output_dir = get_output_dir(config)
path_out = os.path.join(output_dir, f"df_combined_{model}_{dataset_size}_pii_rate_{pii_rate}_n_epochs_{n_epochs}.csv")


def prep_data(data_path, prefix, pii_rate=None, n_epochs=None):
    df = pd.read_csv(data_path)
    df = df[(df['pii_type'] == 'name-patient') | (df['pii_type'] == 'name-attending')]
    if pii_rate is not None:
        df = df[df['pii_rate'] == pii_rate]
    if n_epochs is not None:
        df = df[df['n_epochs'] == n_epochs]
    df_pivot = df.pivot(index=['split', 'value'], columns='prompt', values='ll').add_prefix(prefix).reset_index()
    return df_pivot

df_ft = prep_data(data_path_ft, 'ft_', pii_rate, n_epochs)
df_qi = prep_data(data_path_qi, 'qi_')

df_combined = pd.merge(df_ft, df_qi, on='value', how='inner')

print(df_combined)

# exit()
df_combined.to_csv(path_out, index=False)
print(f"Saved to {path_out}")