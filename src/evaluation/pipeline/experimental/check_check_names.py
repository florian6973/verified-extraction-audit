src_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-all-test/generation_False_all_10_1B_0.1_3_1000000.parquet"

import pandas as pd

df = pd.read_parquet(src_file)
df = df[df['value'].str[0] == ' ']
df = df[df['value'].str[1] != ' ']
df = df[df['value'].str[1:4] != '___']
df = df[df['value'].str.split(' ').str.len() >= 2]
df_ini = df.copy()
df['value'] = df['value'].apply(lambda x: x.replace('.', '').split('\n')[0])
# good it be that some things are coller avec le name?
# no EOS token could be an issue
# now match names from theo and discard what is next simply
df['value'] = df['value'].apply(lambda x: ' '.join(x.strip().split(' ')[0:2]))

print(df['value'].unique())


# theoretical names
path_name_csv = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_10_batch.csv'
df_names = pd.read_csv(path_name_csv)
df_names = df_names[df_names['split'] == 'train']
df_names = df_names[df_names['prompt'] == 'Name: ']
df_names = df_names[df_names['pii_rate'] == 0.1]
df_names = df_names[df_names['n_epochs'] == 3]
df_names = df_names[df_names['dataset_size'] == 10]
# df_names = df_names[df_names['model'] == '1B']
print(df_names['value'].unique())
# df_names = df_names[df_names['
# name attending vs just names when filtering
# print(df.columns)
# print(df.head())

from tqdm import tqdm

possible_names = df_names['value'].dropna().astype(str).unique()

# Collect rows to keep via tqdm loop
# indices_to_keep = []
# for idx, v in tqdm(list(enumerate(df['value'])), desc="Filtering values by names"):
#     if any(str(v).startswith(str(name)) for name in possible_names):
#         indices_to_keep.append(idx)

# df = df.loc[indices_to_keep].copy()

import re

# # make sure values are strings once (cheap) instead of str() repeatedly in the loop
# s = df["value"].astype(str)

# # build a regex: ^(name1|name2|name3...)
# pat = r"^(?:" + "|".join(map(re.escape, map(str, possible_names))) + r")"

# mask = s.str.match(pat, na=False)
# indices_to_keep = df.index[mask].to_list()

# print(df['value'].unique())

import re

s = df["value"].astype(str)

# Prefer longest prefix if overlaps exist
names = sorted(map(str, possible_names), key=len, reverse=True)

pat = r"^(?P<found>" + "|".join(map(re.escape, names)) + r")"
found = s.str.extract(pat)["found"]          # Series: matched prefix or NaN

mask = found.notna()
indices_to_keep = df.index[mask].to_list()

# If you want it attached:
df_out = df.loc[mask].assign(found_name=found[mask].values)
df_out = df_out[['value', 'found_name']]
df_out.drop_duplicates(inplace=True, subset=['found_name'])
# df_out = df_out[df_out['split'] == 'train']
print(df_out.columns)
df_filtered = df_ini[mask].loc[df_out.index]

print(df_out)


print("Check")
path_check_file = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-output-test/10_1B_0.1_3_1000000/values_found_name_train.csv"
df_check = pd.read_csv(path_check_file)
df_check = df_check[df_check['idx'] == 1]
df_check.drop_duplicates(inplace=True, subset=['value'])
print(df_check)
# print(df['value'].unique())

# Display df_out values where found_name is not in df_check['value']
print("\nValues in df_out where found_name is NOT in df_check['value']:")
df_missing = df_filtered[~df_out['found_name'].isin(df_check['value'])]
print(df_missing['value'].unique())
# print(df_filtered)

# print(df['value'].value_counts())

# print(df['value'].value_counts(normalize=True))