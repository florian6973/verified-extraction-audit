import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import json
from collections import Counter
import matplotlib.pyplot as plt

def _get_index_folder():
    """Index folder: env INDEX_FOLDER or repo root / index."""
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.environ.get("INDEX_FOLDER", os.path.join(_repo, "index"))

def select_datasets(dataset_size=1, datasets_index_path=None):
    if datasets_index_path is None:
        datasets_index_path = os.path.join(_get_index_folder(), "datasets.csv")
    datasets_index = pd.read_csv(datasets_index_path)
    datasets_index = datasets_index[datasets_index['injection_strategy'] == 'manual']
    datasets_index = datasets_index[datasets_index['dataset_size'] == dataset_size]
    return datasets_index

def read_dataset(row):
    ds_counter = Counter()
    ds_counter['address'] = 0

    values = {}
    values['address'] = []
    
    with open(row['dataset_path'], 'r') as f:
        dataset = json.load(f)
    # print(dataset[0])
    dataset_size = row['dataset_size']
    if dataset_size == 100:
        dataset_size_str = 'train'
    else:
        dataset_size_str = f'train_{dataset_size}'
    # Base path for PII tags/values: env PII_INSERTION_OUTPUTS or ./outputs/pii_insertion
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _base = os.environ.get("PII_INSERTION_OUTPUTS", os.path.join(_repo, "outputs", "pii_insertion"))
    for note in tqdm(dataset):
        note_name = note['original_note_number']
        pii_fields = note['replaced_pii_fields']
        path_tags = os.path.join(_base, "direct", "gemini-2.5-flash-preview-05-20_v8", dataset_size_str, "tags_manual", note_name)
        path_value = os.path.join(_base, "direct", "gemini-2.5-flash-preview-05-20_v8", dataset_size_str, "json_manual", note_name)

        with open(path_tags, 'r') as f:
            tags = json.load(f)
        with open(path_value, 'r') as f:
            value = json.load(f)
        for field in pii_fields:
            try:
                if tags[field] not in [
                    'name-patient',
                    'name-attending',
                    'name-other',
                    'address', 
                    'phone',
                    'email',
                    'id',
                ]:
                    continue
                if tags[field] not in values:
                    values[tags[field]] = []
                values[tags[field]].append(value[field])
                ds_counter[tags[field]] += 1
            except:
                continue
        # print(ds_counter)
        # input()

    return ds_counter, values

def prepare():
    datasets_index = select_datasets()

    results = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'pii_type', 'raw_count','raw_count_unique', 'proportion', 'proportion_unique'])
    for idx, row in datasets_index.iterrows():
        ds_counter, values = read_dataset(row)

        print(ds_counter)
        values_unique_ratio = {key: len(set(values[key])) / len(values[key]) if len(values[key]) > 0 else 1 for key in values}
        proportions = {key: ds_counter[key] / sum(ds_counter.values()) for key in ds_counter}
        ds_counter['name_total'] = ds_counter['name-patient'] + ds_counter['name-attending'] + ds_counter['name-other']
        values['name_total'] = values['name-patient'] + values['name-attending'] + values['name-other']
        values_unique_ratio['name_total'] = len(set(values['name_total'])) / len(values['name_total'])
        proportions['name_total'] = ds_counter['name_total'] / sum(ds_counter.values())
        print(proportions)
        print(values_unique_ratio)
        for key in ds_counter:
            results.loc[len(results)] = [row['dataset_size'], row['pii_rate'], key, ds_counter[key], len(set(values[key])), proportions[key], values_unique_ratio[key]]
        # input()
        # break
        # dataset_path = f'/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/index/datasets/{dataset_index}'
        # dataset_path = glob.glob(dataset_path)
        # print(dataset_path)
        # break

    results =results.round(4)
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.environ.get("OUTPUT_DIR", os.path.join(_repo, "outputs", "pii_leakage", "distributions"))
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, "denominators.csv"), index=False)

if __name__ == '__main__':
    prepare()
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _out = os.environ.get("OUTPUT_DIR", os.path.join(_repo, "outputs", "pii_leakage", "distributions"))
    results = pd.read_csv(os.path.join(_out, "denominators.csv"))
    # results.pivot(index='dataset_size', columns='pii_type', values=['proportion', 'proportion_unique'])
    # print(results)

    def plot(quantity_name):
        plt.figure(figsize=(10, 5))
        for k, (i, gb) in enumerate(results.groupby(['dataset_size', 'pii_rate'])):
            print(i)
            print(gb)
            plt.subplot(2, 2, k+1)
            gb = gb[~gb['pii_type'].isin(['name-patient', 'name-attending', 'name-other'])]
            gb.sort_values(by=quantity_name, inplace=True, ascending=False)
            plt.bar(gb['pii_type'], gb[quantity_name])
            plt.title(f'Dataset Size: {i[0]}, PII Rate: {i[1]}')
            plt.xlabel('PII Type')
            plt.ylabel(quantity_name.title())
            # if quantity_name == 'proportion_unique':
            plt.ylim(-0.1, 1.1)
            # input()
        plt.tight_layout()
        plt.savefig(os.path.join(_out, f"denominators_{quantity_name}.png"))
        plt.close()
        # input()

    plot('proportion')
    plot('proportion_unique')