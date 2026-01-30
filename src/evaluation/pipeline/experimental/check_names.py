# get_names from ll_all for right model
# check both if name present in output and if exactly space + name

import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter
import numpy as np
import re
from config_helper import format_path, get_generation_file, get_src_ll_file, get_output_dir
from config_loader import load_config, print_config

# Parse arguments
parser = argparse.ArgumentParser(description='Check names in generated output')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

# Load config
config = load_config(args.config)
print_config(config)

# Get paths from config
OUTPUT_DIR = get_output_dir(config)

# Get filter parameters from config
PROMPT = config['filters']['prompt']
PII_RATE = config['filters']['pii_rate']
N_EPOCHS = config['filters']['n_epochs']
DATASET_SIZE = config['filters']['dataset_size']
MODEL = config['filters']['model']
PII_TYPES = config['filters']['pii_types']

file_gen = get_generation_file(config) # config['inputs']['generation_file']
src_names = get_src_ll_file(config) # config['inputs']['src_ll_file']

print("Input files:")
print(file_gen)
print(src_names)
print("Output directory:")
print(OUTPUT_DIR)
# input()

df_gen = pd.read_csv(file_gen)
df_gen = df_gen[df_gen['prompt'] == PROMPT]
df_src = pd.read_csv(src_names)
df_src = df_src[df_src['prompt'] == PROMPT]

df_src = df_src[df_src['pii_rate'] == PII_RATE]
df_src = df_src[df_src['n_epochs'] == N_EPOCHS]

df_src_names = df_src[['pii_type', 'prompt', 'value', 'split', 'pii_rate', 'n_epochs']].drop_duplicates()
# df_src = df_src[df_src['dataset_size'] == 10]
# df_src = df_src[['pii_type', 'prompt', 'value', 'list_tokens', 'list_tokens_with_first_name']]
df_src_names = df_src[df_src['pii_type'].isin(PII_TYPES)]
# df_src_names = df_src
# df_src_names = df_src_names[['pii_type', 'prompt', 'value']]

df_src_train_names = df_src_names[df_src_names['split'] == 'train']

df_src_val_names = df_src_names[df_src_names['split'] == 'val']
print(df_gen.columns)
print(df_src.columns)
print(df_gen)
print(df_src_names)

# why NAN VALUES FOR EXTRACTION??? OK because MRN so no first name

def clean_name(name):
    """Clean name: lowercase and remove dots (keep spaces for idx check)."""
    if isinstance(name, str):
        name = name.lower().replace('.', '')
    return name

# Preprocessing: filter and clean generated values (from check_check_names.py)
print("\nPreprocessing generated values...")
df_gen_ini = df_gen.copy()
df_gen_filtered = df_gen[df_gen['value'].astype(str).str[0] == ' '].copy()
df_gen_filtered = df_gen_filtered[df_gen_filtered['value'].astype(str).str[1] != ' ']
df_gen_filtered = df_gen_filtered[df_gen_filtered['value'].astype(str).str[1:4] != '___']
df_gen_filtered = df_gen_filtered[df_gen_filtered['value'].astype(str).str.split(' ').str.len() >= 2]

# Clean values: remove dots, split by newline and take first part, then take first two words
df_gen_filtered = df_gen_filtered.copy()
df_gen_filtered['value_processed'] = df_gen_filtered['value'].apply(
    lambda x: ' '.join(x.replace('.', '').split('\n')[0].strip().split(' ')[0:2]) if isinstance(x, str) else x
)
df_gen_filtered['value_processed_clean'] = df_gen_filtered['value_processed'].apply(clean_name)

print(f"Filtered from {len(df_gen)} to {len(df_gen_filtered)} generated values after preprocessing")

def check_names(df_src_names, df_gen_filtered, split):
    """Check names using preprocessing + regex method."""
    # Get unique names from source
    possible_names = df_src_names['value'].dropna().astype(str).unique()
    possible_names_clean = [clean_name(name) for name in possible_names if isinstance(name, str)]
    possible_names_clean = [name for name in possible_names_clean if name]  # Remove empty strings
    
    if len(possible_names_clean) == 0:
        print(f"[{split}] No valid names to match")
        return pd.DataFrame(columns=['value', 'idx', 'value_found']), Counter()
    
    # Sort names by length (longest first) to prefer longest prefix if overlaps exist
    names_sorted = sorted(possible_names_clean, key=len, reverse=True)
    
    # Build regex pattern: ^(name1|name2|name3...)
    pat = r"^(?P<found>" + "|".join(map(re.escape, names_sorted)) + r")"
    
    # Apply regex matching
    s = df_gen_filtered["value_processed_clean"].astype(str)
    found = s.str.extract(pat)["found"]  # Series: matched prefix or NaN
    
    mask = found.notna()
    df_matched = df_gen_filtered[mask].copy()
    df_matched['found_name'] = found[mask].values
    
    # Get original names (not cleaned) for matching
    name_mapping = {}
    for orig_name in possible_names:
        orig_clean = clean_name(orig_name)
        if orig_clean in names_sorted:
            name_mapping[orig_clean] = orig_name
    
    # Map found_name back to original case
    df_matched['found_name_original'] = df_matched['found_name'].map(name_mapping)
    
    # Calculate idx (position where name starts in the original value)
    values_found = []
    seen = set()
    counter = Counter()
    
    for idx_row, row in df_matched.iterrows():
        found_name_clean = row['found_name']
        found_name_orig = row['found_name_original']
        value_processed_clean = row['value_processed_clean']
        value_original = row['value']
        
        if found_name_clean in seen:
            continue
        seen.add(found_name_clean)
        
        # Find position in original value (before preprocessing)
        value_original_clean = clean_name(value_original)
        if isinstance(value_original_clean, str) and found_name_clean in value_original_clean:
            idx = value_original_clean.index(found_name_clean)
            values_found.append((found_name_orig, idx, value_original))
            counter[found_name_orig] += 1
    
    print(split)
    print(f"Found {len(values_found)} matches")
    print(f"Total unique names checked: {len(seen)}")
    print(f"Total possible names: {len(possible_names_clean)}")
    
    df_found = pd.DataFrame(values_found, columns=['value', 'idx', 'value_found'])
    df_found.to_csv(os.path.join(OUTPUT_DIR, f'values_found_name_{split}.csv'), index=False)
    
    return df_found, counter

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_found_train, counter_train = check_names(df_src_train_names, df_gen_filtered, 'train')
df_found_val, counter_val = check_names(df_src_val_names, df_gen_filtered, 'val')

df_found_all = pd.concat([df_found_train, df_found_val])

# Check if all idx are equal to 1 (space + name pattern) - separately for train and val
# how can continuation be 58 if just 20 tokens? because characters.
for split_name, df_split in [('train', df_found_train), ('val', df_found_val)]:
    exceptions_split = df_split[df_split['idx'] != 1]
    if len(exceptions_split) == 0:
        print(f"\n[{split_name}] All idx values are equal to 1 (space + name pattern)")
    else:
        print(f"\n[{split_name}] Found {len(exceptions_split)} exceptions where idx != 1:")
        print(exceptions_split.to_string())
        print(f"\n[{split_name}] idx value distribution:")
        print(df_split['idx'].value_counts().sort_index())

# Combined summary
exceptions = df_found_all[df_found_all['idx'] != 1]
print(f"\n[TOTAL] {len(exceptions)} exceptions out of {len(df_found_all)} total matches")

# Keep only completions where idx == 1 (space + name pattern)
df_found_train = df_found_train[df_found_train['idx'] == 1]
df_found_val = df_found_val[df_found_val['idx'] == 1]
df_found_all = pd.concat([df_found_train, df_found_val])
print(f"\n[FILTERED] Kept {len(df_found_all)} matches with idx == 1")

df_found_values = df_found_all['value_found'].unique().tolist()

# remaining_values: generated values that were NOT matched to any name in df_src
# these still need idx == 1 check (the name detected by NER should start at position 1)
# Exclude values containing "___" (placeholders) or numbers
def has_numbers(s):
    return any(c.isdigit() for c in s)

# Get original values from filtered dataframe
values_gen_original = df_gen_filtered['value'].unique().tolist()

remaining_values = [value for value in values_gen_original 
                    if value not in df_found_values 
                    and isinstance(value, str) 
                    and '___' not in value
                    and not has_numbers(value)]
print(f"\nRemaining values (not in df_src, excluding ___ and numbers): {len(remaining_values)}")

# For names that ARE in df_src, get the ll from it
# Join df_found with df_src to get ll values
df_found_train_with_ll = df_found_train.merge(
    df_src[['value', 'll', 'pii_type', 'split']].drop_duplicates(subset=['value']), 
    on='value', 
    how='left'
)
df_found_val_with_ll = df_found_val.merge(
    df_src[['value', 'll', 'pii_type', 'split']].drop_duplicates(subset=['value']), 
    on='value', 
    how='left'
)

print(f"\n[Train] Found ll for {df_found_train_with_ll['ll'].notna().sum()} / {len(df_found_train_with_ll)} names")
print(f"[Val] Found ll for {df_found_val_with_ll['ll'].notna().sum()} / {len(df_found_val_with_ll)} names")

# Save the matched names with ll
df_found_train_with_ll.to_csv(os.path.join(OUTPUT_DIR, 'values_found_name_train_with_ll.csv'), index=False)
df_found_val_with_ll.to_csv(os.path.join(OUTPUT_DIR, 'values_found_name_val_with_ll.csv'), index=False)

# Save remaining values to be processed by NER + ll computation
df_remaining = pd.DataFrame({'value_found': remaining_values})
df_remaining.to_csv(os.path.join(OUTPUT_DIR, 'remaining_values_for_ner.csv'), index=False)

print(f"\nOutput files saved to: {OUTPUT_DIR}")

# # for each name, compute the probability of being extracted by the model
# probas_train = {}
# probas_val = {}
# for name in counter_train.keys():
#     probas_train[name] = counter_train[name] / len(df_gen)
# for name in counter_val.keys():
#     probas_val[name] = counter_val[name] / len(df_gen)

# # theoretical proba from df_src
# df_probas_train = pd.DataFrame(probas_train.items(), columns=['value', 'probability'])
# df_probas_train_joined = df_probas_train.merge(df_src, on='value', how='inner')
# df_probas_train_joined = df_probas_train_joined[['value', 'probability', 'll']]
# df_probas_train_joined['proba_theo'] = np.exp(df_probas_train_joined['ll'])
# df_probas_val = pd.DataFrame(probas_val.items(), columns=['value', 'probability'])
# df_probas_val_joined = df_probas_val.merge(df_src, on='value', how='inner')
# df_probas_val_joined = df_probas_val_joined[['value', 'probability', 'll']]
# df_probas_val_joined['proba_theo'] = np.exp(df_probas_val_joined['ll'])
# df_probas_train_joined.to_csv('probas_train_joined.csv', index=False)
# df_probas_val_joined.to_csv('probas_val_joined.csv', index=False)



# coverage is wrong, then let's check ranking
# compute the probabilities for each name first to make sure it is consistent

# to check ALL using a classifier trained on the right model (see MIA code output)


# check all names, apply mia, and then compare
# training_names
# other_names

