from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import os

# -----------------------
# Trie utilities (token-level)
# -----------------------
@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    # list because multiple originals could map to same token list (rare but safe)
    terminal_token_lists: List[List[int]] = field(default_factory=list)

def parse_list_tokens(tokens_input) -> List[int]:
    """
    Parse tokens from various input formats (string, list, array).
    Handles multiple formats:
    - List/array: [1, 2, 3] (from parquet)
    - Simple list string: "[1, 2, 3]" or "[1,2,3]" (from CSV)
    - Tensor format: "tensor(1) tensor(2) tensor(3)" or "[tensor(1, device='cuda:0'), tensor(2)]"
    - Space-separated: "1 2 3"
    - Comma-separated: "1, 2, 3" or "1,2,3"
    """
    # Handle None or NaN
    if tokens_input is None:
        return []
    try:
        if pd.isna(tokens_input):
            return []
    except (ValueError, TypeError):
        # pd.isna() can fail on arrays, continue to check type
        pass
    
    # Handle list/array inputs (from parquet)
    if isinstance(tokens_input, (list, tuple)):
        try:
            return [int(x) for x in tokens_input]
        except (ValueError, TypeError):
            return []
    
    # Handle numpy arrays
    if hasattr(tokens_input, '__iter__') and not isinstance(tokens_input, str):
        try:
            return [int(x) for x in tokens_input]
        except (ValueError, TypeError):
            return []
    
    # Handle string inputs (from CSV)
    if not isinstance(tokens_input, str):
        return []
    
    tokens_str = tokens_input
    
    # Try to parse tensor format first (most specific, handles device='cuda:0' etc.)
    import re
    # Match tensor(123) or tensor(123, device='cuda:0') etc.
    tensor_matches = re.findall(r"tensor\s*\(\s*(\d+)", tokens_str)
    if tensor_matches:
        return [int(x) for x in tensor_matches]
    
    # Try to parse as regular list string (simple integers only)
    try:
        import ast
        parsed = ast.literal_eval(tokens_str)
        if isinstance(parsed, list):
            # Handle list of integers
            result = []
            for x in parsed:
                if isinstance(x, int):
                    result.append(x)
                elif isinstance(x, (str, float)):
                    result.append(int(x))
            if result:
                return result
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # Try to parse as comma-separated integers (with or without brackets)
    try:
        cleaned = tokens_str.strip().strip('[]')
        if ',' in cleaned:
            tokens = []
            for x in cleaned.split(','):
                x_clean = x.strip()
                # Try to extract number if it's wrapped in something
                num_match = re.search(r'(\d+)', x_clean)
                if num_match:
                    tokens.append(int(num_match.group(1)))
                elif x_clean.isdigit():
                    tokens.append(int(x_clean))
            if tokens:
                return tokens
    except:
        pass
    
    # Try to parse as space-separated integers (handle multiple spaces)
    try:
        cleaned = tokens_str.strip().strip('[]')
        # Use regex to find all numbers (handles multiple spaces, commas, etc.)
        # This will extract all integers from the string
        num_matches = re.findall(r'\b(\d+)\b', cleaned)
        if num_matches:
            return [int(x) for x in num_matches]
    except:
        pass
    
    return []

def build_token_trie(token_lists: List[List[int]]) -> TrieNode:
    root = TrieNode()
    for token_list in token_lists:
        if not token_list:
            continue
        node = root
        for token in token_list:
            node = node.children.setdefault(token, TrieNode())
        node.terminal_token_lists.append(token_list)
    return root

def trie_match_prefixes(root: TrieNode, token_list: List[int]) -> List[List[int]]:
    """
    Return ALL terminal token lists encountered along the path of token_list.
    This gives "prefix-credit" matches: every source-name that is a prefix of token_list.
    """
    node = root
    hits = []
    for token in token_list:
        if token not in node.children:
            break
        node = node.children[token]
        if node.terminal_token_lists:
            hits.extend(node.terminal_token_lists)
    return hits

def trie_match_longest(root: TrieNode, token_list: List[int]) -> Optional[List[int]]:
    """
    Return the LONGEST terminal match along the path of token_list (if any).
    This gives "longest-match wins" behavior.
    """
    node = root
    last = None
    for token in token_list:
        if token not in node.children:
            break
        node = node.children[token]
        if node.terminal_token_lists:
            # if multiple, just take the first (identical token lists usually)
            last = node.terminal_token_lists[0]
    return last


# -----------------------
# Replacement check_names implementing fixes #1 and #4
# -----------------------
def check_names(
    df_src_names: pd.DataFrame,
    df_gen_filtered: pd.DataFrame,
    split: str,
    OUTPUT_DIR: str,
    match_mode: str = "all_prefixes",  # "all_prefixes" or "longest"
) -> Tuple[pd.DataFrame, Counter]:
    """
    Fix #1: A name is considered found if EXISTS any occurrence with idx==0.
    Fix #4: Optionally credit all prefix names (prefix-event) instead of only longest match.
    
    Now matches on list_tokens instead of strings.
    Since space has no separate token, matches start at idx==0 in the generated token list.

    Returns:
      df_found: unique found names (original casing) with idx==0 (one row per name)
      counter: counts of idx==0 hits across samples (not unique)
    """
    # ---- build source universe from list_tokens
    if 'list_tokens' not in df_src_names.columns:
        print(f"[{split}] Error: 'list_tokens' column not found in source dataframe")
        return pd.DataFrame(columns=["value", "idx", "value_found"]), Counter()
    
    # Get unique (value, list_tokens) pairs
    src_tokens_data = df_src_names[['value', 'list_tokens']].drop_duplicates()
    

    # Parse token lists and build mapping
    token_list_to_name = {}
    possible_token_lists = []
    failed_parses = []
    for _, row in tqdm(src_tokens_data.iterrows(), desc="Parsing token lists", total=len(src_tokens_data)):
        token_list = parse_list_tokens(row['list_tokens'])
        if token_list:  # Only add non-empty token lists
            possible_token_lists.append(token_list)
            # Use tuple as key for token list (lists are not hashable)
            token_list_key = tuple(token_list)
            if token_list_key not in token_list_to_name:
                token_list_to_name[token_list_key] = row['value']
            # If multiple names map to same token list, keep the first one
        else:
            # Track failed parses for debugging
            failed_parses.append((row['value'], row['list_tokens']))
    
    if failed_parses:
        print(f"\n[{split}] Warning: {len(failed_parses)} token lists failed to parse. Examples:")
        for name, tokens_str in failed_parses[:5]:  # Show first 5 examples
            print(f"  Name: {name}, list_tokens: {str(tokens_str)[:100]}")

    if not possible_token_lists:
        print(f"[{split}] No valid token lists to match")
        return pd.DataFrame(columns=["value", "idx", "value_found"]), Counter()

    print(f"Building trie from {len(possible_token_lists)} token lists...")
    # trie for prefix lookups on tokens
    trie_root = build_token_trie(possible_token_lists)
    
    # Debug: Check if a specific name is in the trie (e.g., "Lauren Wagner")
    debug_name = "Lauren Wagner"
    debug_found = False
    for token_list_key, name in token_list_to_name.items():
        if name == debug_name:
            debug_found = True
            print(f"DEBUG: Found '{debug_name}' with tokens: {list(token_list_key)}")
            break
    if not debug_found:
        print(f"DEBUG: '{debug_name}' not found in token_list_to_name mapping")

    # ---- scan generations
    found_unique_idx0 = set()  # Store token list tuples
    # optional examples (store one original completion that triggered it)
    found_example = {}
    # count how many idx==0 occurrences per name (not unique)
    counter = Counter()

    # IMPORTANT: do NOT dedupe before idx check (Fix #1)
    for _, row in tqdm(df_gen_filtered.iterrows(), desc="Checking generations", total=len(df_gen_filtered)):
        # Get tokens from generated dataframe (column name is 'tokens', format is [1,2,3] or list from parquet)
        gen_tokens = row.get("tokens", None)
        s_orig = row.get("value", None)

        # Check if tokens is None/NaN (handle both scalar and array cases)
        try:
            if pd.isna(gen_tokens):
                continue
        except (ValueError, TypeError):
            # pd.isna() can fail on arrays/lists, check if it's None directly
            if gen_tokens is None:
                continue
        
        # Check if value is a string
        if not isinstance(s_orig, str):
            continue

        gen_token_list = parse_list_tokens(gen_tokens)
        if not gen_token_list:
            continue

        if match_mode == "longest":
            hit_token_list = trie_match_longest(trie_root, gen_token_list)
            hits_token_lists = [hit_token_list] if hit_token_list is not None else []
        elif match_mode == "all_prefixes":
            hits_token_lists = trie_match_prefixes(trie_root, gen_token_list)
        else:
            raise ValueError("match_mode must be 'longest' or 'all_prefixes'")

        if not hits_token_lists:
            continue
        
        # Debug: Check if we're matching "Lauren Wagner" tokens
        debug_lauren_tokens = [43460, 52475]  # From the CSV example
        if gen_token_list[:2] == debug_lauren_tokens or any([43460, 52475] == hit[:2] if len(hit) >= 2 else False for hit in hits_token_lists):
            print(f"DEBUG: Found potential Lauren Wagner match!")
            print(f"  gen_token_list[:10]: {gen_token_list[:10]}")
            print(f"  hits_token_lists: {hits_token_lists}")
            print(f"  s_orig: {s_orig[:100] if s_orig else None}")

        # For each credited token list, check idx==0 on the generated token list
        # idx==0 means the name starts at position 0 (space has no separate token)
        # We need to check if the generated token list starts with the name tokens
        for name_token_list in hits_token_lists:
            if not name_token_list:
                continue
            
            # Check if name_token_list appears at position 0 in gen_token_list
            # Since space has no token, the name tokens should start at idx 0
            if len(gen_token_list) >= len(name_token_list):
                # Check if name_token_list is a prefix of gen_token_list starting at position 0
                gen_prefix = gen_token_list[:len(name_token_list)]
                # Ensure both are lists of integers for comparison
                if isinstance(gen_prefix, list) and isinstance(name_token_list, list):
                    # Convert to integers if needed (handle any type mismatches)
                    try:
                        gen_prefix_ints = [int(x) for x in gen_prefix]
                        name_token_list_ints = [int(x) for x in name_token_list]
                        if gen_prefix_ints == name_token_list_ints:
                            # Use the original name_token_list format for the key to match token_list_to_name
                            token_list_key = tuple(name_token_list)  # Use original format from trie match
                            
                            # Debug for Lauren Wagner
                            if name_token_list_ints == [43460, 52475]:
                                print(f"DEBUG: Lauren Wagner idx==0 check passed!")
                                print(f"  token_list_key: {token_list_key}")
                                print(f"  token_list_key in token_list_to_name: {token_list_key in token_list_to_name}")
                                print(f"  name_value: {token_list_to_name.get(token_list_key, 'NOT FOUND')}")
                            
                            found_unique_idx0.add(token_list_key)
                            name_value = token_list_to_name.get(token_list_key, "")
                            if name_value:
                                counter[name_value] += 1
                                if token_list_key not in found_example:
                                    found_example[token_list_key] = s_orig  # store one example completion
                            elif name_token_list_ints == [43460, 52475]:
                                print(f"DEBUG: Lauren Wagner matched but name_value is empty!")
                                print(f"  Available keys in token_list_to_name (first 5): {list(token_list_to_name.keys())[:5]}")
                    except (ValueError, TypeError) as e:
                        # Skip if conversion fails
                        if name_token_list == [43460, 52475] or gen_prefix[:2] == [43460, 52475]:
                            print(f"DEBUG: Lauren Wagner conversion error: {e}")
                            print(f"  gen_prefix: {gen_prefix}, name_token_list: {name_token_list}")
                        continue

    # ---- build output df (one row per unique found name)
    rows = []
    for token_list_key in sorted(found_unique_idx0):
        orig_name = token_list_to_name.get(token_list_key, "")
        rows.append((orig_name, 0, found_example.get(token_list_key, "")))

    df_found = pd.DataFrame(rows, columns=["value", "idx", "value_found"])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"values_found_name_{split}.csv")
    df_found.to_csv(out_path, index=False)

    print(f"\n[{split}] match_mode={match_mode}")
    print(f"  Total possible token lists: {len(possible_token_lists)}")
    print(f"  Unique names found with idx==0: {len(df_found)}")
    print(f"  Saved: {out_path}")

    return df_found, counter

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

# Try to read as parquet first, fall back to CSV
try:
    parquet_file = file_gen.replace('.csv', '.parquet')
    print(f"Trying to load from parquet: {parquet_file}")
    if os.path.exists(parquet_file):
        df_gen = pd.read_parquet(parquet_file)
        print(f"Loaded from parquet: {parquet_file}")
    else:
        df_gen = pd.read_csv(file_gen)
        print(f"Loaded from CSV: {file_gen}")
except Exception as e:
    print(f"Error loading parquet, trying CSV: {e}")
    df_gen = pd.read_csv(file_gen)
    print(f"Loaded from CSV: {file_gen}")
print(f"\nLoaded {len(df_gen)} rows from generation file")
print(f"Filtering by prompt: '{PROMPT}'")
print(f"Unique prompts in df_gen: {df_gen['prompt'].unique() if 'prompt' in df_gen.columns else 'NO PROMPT COLUMN'}")
df_gen_before = len(df_gen)
df_gen = df_gen[df_gen['prompt'] == PROMPT]
print(f"After prompt filter: {len(df_gen)} rows (dropped {df_gen_before - len(df_gen)} rows)")
# Try to read source as parquet first, fall back to CSV
try:
    src_parquet_file = src_names.replace('.csv', '.parquet')
    if os.path.exists(src_parquet_file):
        df_src = pd.read_parquet(src_parquet_file)
        print(f"Loaded source from parquet: {src_parquet_file}")
    else:
        df_src = pd.read_csv(src_names)
        print(f"Loaded source from CSV: {src_names}")
except Exception as e:
    print(f"Error loading source parquet, trying CSV: {e}")
    df_src = pd.read_csv(src_names)
    print(f"Loaded source from CSV: {src_names}")

df_src = df_src[df_src['prompt'] == PROMPT]

df_src = df_src[df_src['pii_rate'] == PII_RATE]
df_src = df_src[df_src['n_epochs'] == N_EPOCHS]

# Include list_tokens in the source dataframe for token-based matching
df_src_names = df_src[['pii_type', 'prompt', 'value', 'list_tokens', 'split', 'pii_rate', 'n_epochs']].drop_duplicates()
# df_src = df_src[df_src['dataset_size'] == 10]
# df_src = df_src[['pii_type', 'prompt', 'value', 'list_tokens', 'list_tokens_with_first_name']]
df_src_names = df_src_names[df_src_names['pii_type'].isin(PII_TYPES)]
# df_src_names = df_src
# df_src_names = df_src_names[['pii_type', 'prompt', 'value']]

df_src_train_names = df_src_names[df_src_names['split'] == 'train']

df_src_val_names = df_src_names[df_src_names['split'] == 'val']
print(df_gen.columns)
print(df_src.columns)
print(df_gen)
print(df_src_names)

# Use dataframes with tokens directly - no string preprocessing needed
print("\nUsing token-based matching...")
df_gen_filtered = df_gen.copy()

# Check if tokens column exists in generated dataframe
if 'tokens' not in df_gen_filtered.columns:
    print("WARNING: 'tokens' column not found in generated dataframe.")
    print("Token-based matching requires 'tokens' column. Please ensure the generation file includes this column.")
    # Continue anyway - the check_names function will handle this

print(f"Using {len(df_gen_filtered)} generated values with token-based matching")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_found_train, counter_train = check_names(df_src_train_names, df_gen_filtered, "train", OUTPUT_DIR, match_mode="all_prefixes")
df_found_val, counter_val     = check_names(df_src_val_names,   df_gen_filtered, "val",   OUTPUT_DIR, match_mode="all_prefixes")

df_found_all = pd.concat([df_found_train, df_found_val])

# Check if all idx are equal to 0 (name starts at token position 0, space has no token) - separately for train and val
# how can continuation be 58 if just 20 tokens? because characters.
for split_name, df_split in [('train', df_found_train), ('val', df_found_val)]:
    exceptions_split = df_split[df_split['idx'] != 0]
    if len(exceptions_split) == 0:
        print(f"\n[{split_name}] All idx values are equal to 0 (name starts at token position 0)")
    else:
        print(f"\n[{split_name}] Found {len(exceptions_split)} exceptions where idx != 0:")
        print(exceptions_split.to_string())
        print(f"\n[{split_name}] idx value distribution:")
        print(df_split['idx'].value_counts().sort_index())

# Combined summary
exceptions = df_found_all[df_found_all['idx'] != 0]
print(f"\n[TOTAL] {len(exceptions)} exceptions out of {len(df_found_all)} total matches")

# Keep only completions where idx == 0 (name starts at token position 0)
df_found_train = df_found_train[df_found_train['idx'] == 0]
df_found_val = df_found_val[df_found_val['idx'] == 0]
df_found_all = pd.concat([df_found_train, df_found_val])
print(f"\n[FILTERED] Kept {len(df_found_all)} matches with idx == 0")

# df_found_values = df_found_all['value_found'].unique().tolist()

# # remaining_values: generated values that were NOT matched to any name in df_src
# # these still need idx == 0 check (the name detected by NER should start at position 0)
# # Exclude values containing "___" (placeholders) or numbers
# def has_numbers(s):
#     return any(c.isdigit() for c in s)

# # Get original values from filtered dataframe
# values_gen_original = df_gen_filtered['value'].unique().tolist()

# remaining_values = [value for value in values_gen_original 
#                     if value not in df_found_values 
#                     and isinstance(value, str) 
#                     and '___' not in value
#                     and not has_numbers(value)]
# print(f"\nRemaining values (not in df_src, excluding ___ and numbers): {len(remaining_values)}")

# 1) Build a fast lookup for found values (Index/hash-based)
found_idx = pd.Index(df_found_all["value_found"].dropna().unique())

# 2) Work on the full column first (avoid unique() until the very end)
s = df_gen_filtered["value"]

# If you already know s is strings/NA, skip the isinstance filter.
mask = (
    s.notna()
    & ~s.isin(found_idx)
    & ~s.str.contains("___", regex=False)
    & ~s.str.contains(r"\d", regex=True)
)

# 3) Only now dedupe if you truly need unique remaining values
remaining_values = s[mask].drop_duplicates().tolist()

print("Remaining:", len(remaining_values))

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

