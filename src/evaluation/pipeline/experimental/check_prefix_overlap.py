# for 10B check sum and no prefix token, especially low probap

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Check prefix overlap and generate exclusion list')
parser.add_argument('--exclude-prefix', action='store_true', 
                    help='Exclude the prefix (full entries) instead of other sequences in the group')
args = parser.parse_args()

EXCLUDE_PREFIX = args.exclude_prefix

path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_10_batch.csv'
df = pd.read_csv(path)

# to expand more then
df = df[df['prompt'] == 'Name: ']
df = df[df['pii_rate'] == 0.1]
df = df[df['n_epochs'] == 3]
# df = df[df['n_epochs'] != 10]
# df = df[df['dataset_size'] == 10]
# df = df[df['model'] == '1B']

# print(df.columns)
# print(df)
# print(np.exp(df['ll'].prod()))
# print(np.sum(np.exp(df['ll'])))

from torch import tensor

# lists = [[l0.item() for l0 in eval(l)] for l in df['list_tokens'].tolist()]
lists = df['list_tokens'].tolist()
# print(lists)
# lcp = os.path.commonprefix(lists)
# print(lcp)
def parse_tensor_list_string(t: str) -> list[int]:
    import re
    return [int(x) for x in re.findall(r"tensor\(\s*(\d+)", t)]

lists = [parse_tensor_list_string(l) for l in lists]
# print(lists)


from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Node:
    children: Dict[int, "Node"] = field(default_factory=dict)
    count: int = 0
    idxs: List[int] = field(default_factory=list)

def maximal_shared_prefixes_int(seqs: List[List[int]]) -> List[Tuple[List[int], List[List[int]], List[int]]]:
    root = Node()

    # Build trie over integer tokens
    for i, seq in enumerate(seqs):
        node = root
        for tok in seq:
            node = node.children.setdefault(tok, Node())
            node.count += 1
            node.idxs.append(i)

    results: List[Tuple[List[int], List[List[int]], List[int]]] = []

    def dfs(node: Node, prefix: List[int]):
        shared = node.count >= 2
        child_has_shared = any(child.count >= 2 for child in node.children.values())

        # Maximal shared prefix: shared here, but cannot extend to another shared node
        if shared and not child_has_shared:
            # unique sequences in stable order
            seen = set()
            group = []
            group_indices = []
            for idx in node.idxs:
                if idx not in seen:
                    seen.add(idx)
                    group.append(seqs[idx])
                    group_indices.append(idx)
            results.append((prefix.copy(), group, group_indices))

        for tok, child in node.children.items():
            prefix.append(tok)
            dfs(child, prefix)
            prefix.pop()

    for tok, child in root.children.items():
        dfs(child, [tok])

    # sort: longest prefixes first, then lexicographic
    # results.sort(key=lambda x: (-len(x[0]), x[0]))
    results.sort(key=lambda x: (len(x[0]), x[0]))
    return results

results = maximal_shared_prefixes_int(lists)
print(f"Total sequences: {len(lists)}")
print(f"Total prefix groups: {len(results)}")

# Check if any prefix is a full entry (matches a complete sequence)
# Convert lists to tuples for comparison
lists_as_tuples = [tuple(seq) for seq in lists]

prefixes_that_are_full_entries = []
for pref, group, indices in tqdm(results, desc="Checking prefixes"):
    pref_tuple = tuple(pref)
    # Check if this prefix matches any complete sequence
    matching_indices = [i for i, seq in enumerate(lists_as_tuples) if seq == pref_tuple]
    if matching_indices:
        prefixes_that_are_full_entries.append((pref, group, indices, matching_indices))

print(f"\nFound {len(prefixes_that_are_full_entries)} prefix groups where the prefix is a full entry")
print("="*60)

# Display prefixes that are full entries
for i, (pref, group, indices, matching_indices) in enumerate(prefixes_that_are_full_entries, 1):
    print(f"\nPrefix Group {i}:")
    print(f"  Prefix: {pref}")
    print(f"  Prefix length: {len(pref)}")
    print(f"  Number of sequences sharing this prefix: {len(group)}")
    print(f"  This prefix appears as a FULL ENTRY in rows: {matching_indices}")
    
    print("  Full entry details:")
    for idx in matching_indices:
        if 'value' in df.columns:
            value = df.iloc[idx]['value']
            if 'll' in df.columns:
                prob = np.exp(df.iloc[idx]['ll'])
                print(f"    Row {idx}: value={value}, probability={prob:.6e}")
            else:
                print(f"    Row {idx}: value={value}")
        else:
            print(f"    Row {idx}: (value column not found)")
    
    print("  All sequences in this prefix group:")
    for idx in indices[:5]:  # Show first 5 sequences
        seq = lists[idx]
        if 'value' in df.columns:
            value = df.iloc[idx]['value']
            if 'll' in df.columns:
                prob = np.exp(df.iloc[idx]['ll'])
                print(f"    Row {idx}: seq={seq}, value={value}, probability={prob:.6e}")
            else:
                print(f"    Row {idx}: seq={seq}, value={value}")
        else:
            print(f"    Row {idx}: seq={seq}")
    if len(indices) > 5:
        print(f"    ... and {len(indices) - 5} more sequences")

# Collect values to exclude
values_to_exclude = []
indices_to_exclude = set()

if EXCLUDE_PREFIX:
    # Exclude the prefix (full entries), keep other sequences in the group
    print(f"\n{'='*60}")
    print(f"Mode: EXCLUDE PREFIX (full entries)")
    print(f"{'='*60}")
    for pref, group, indices, matching_indices in prefixes_that_are_full_entries:
        # Exclude full entries (the prefixes)
        for idx in matching_indices:
            indices_to_exclude.add(idx)
else:
    # Keep full entries, exclude other sequences in the group (default behavior)
    print(f"\n{'='*60}")
    print(f"Mode: EXCLUDE OTHER SEQUENCES (keep full entries)")
    print(f"{'='*60}")
    for pref, group, indices, matching_indices in prefixes_that_are_full_entries:
        # Keep full entries, exclude other sequences in the group
        matching_set = set(matching_indices)
        for idx in indices:
            if idx not in matching_set:
                indices_to_exclude.add(idx)

print(f"\nCollecting values to exclude...")
print(f"Total rows to exclude: {len(indices_to_exclude)}")

# Calculate probability mass excluded
if 'll' in df.columns:
    # Calculate total probability mass in dataframe
    total_prob_mass = np.sum(np.exp(df['ll']))
    
    # Calculate probability mass for excluded entries
    excluded_prob_mass = 0.0
    for idx in indices_to_exclude:
        excluded_prob_mass += np.exp(df.iloc[idx]['ll'])
    
    # Calculate percentage
    excluded_percentage = (excluded_prob_mass / total_prob_mass * 100) if total_prob_mass > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"PROBABILITY MASS ANALYSIS")
    print(f"{'='*60}")
    print(f"Total probability mass in dataframe: {total_prob_mass:.6e}")
    print(f"Probability mass excluded: {excluded_prob_mass:.6e}")
    print(f"Percentage of probability mass excluded: {excluded_percentage:.4f}%")
    print(f"{'='*60}\n")
else:
    print("Warning: 'll' column not found, cannot calculate probability mass")

# Get values for rows to exclude
if 'value' in df.columns:
    for idx in sorted(indices_to_exclude):
        value = df.iloc[idx]['value']
        values_to_exclude.append({'value': value, 'row_index': idx})
    
    # Create DataFrame and save to CSV
    exclude_df = pd.DataFrame(values_to_exclude)
    if EXCLUDE_PREFIX:
        output_path = path.replace('.csv', '_exclude_prefix_values.csv')
        print(f"Excluding prefix (full entry) values")
    else:
        output_path = path.replace('.csv', '_exclude_values.csv')
        print(f"Excluding other sequence values (keeping full entries)")
    exclude_df.to_csv(output_path, index=False)
    print(f"Saved {len(exclude_df)} values to exclude to: {output_path}")
    print(f"Columns: {exclude_df.columns.tolist()}")
    print(f"First few values to exclude:")
    print(exclude_df.head(10))
else:
    print("Warning: 'value' column not found, cannot create exclusion list")