# for 10B check sum and no prefix token, especially low probap

import pandas as pd
import numpy as np
import os
path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_False_1B_1_batch.csv'
df = pd.read_csv(path)

df = df[df['prompt'] == 'Name: ']
df = df[df['pii_rate'] == 0.1]
# df = df[df['n_epochs'] == 3]
df = df[df['n_epochs'] != 10]
# df = df[df['dataset_size'] == 10]
# df = df[df['model'] == '1B']

# print(df.columns)
# print(df)
# print(np.exp(df['ll'].prod()))
# print(np.sum(np.exp(df['ll'])))

print(df.columns)
exit()
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
print(len(lists))
print(len(results))

# Calculate cumulative probability for each group and sort by it
if 'll' in df.columns:
    results_with_probs = []
    for pref, group, indices in results:
        cumulative_prob = sum(np.exp(df.iloc[idx]['ll']) for idx in indices)
        results_with_probs.append((pref, group, indices, cumulative_prob))
    
    # Sort by cumulative probability (descending)
    results_with_probs.sort(key=lambda x: x[3], reverse=True)
    
    # Display the first 10 (highest cumulative probability) groups
    for i, (pref, group, indices, cumulative_prob) in enumerate(results_with_probs[:10], 1):
        print(f"\n{'='*60}")
        print(f"Group {i}")
        print(f"Prefix: {pref}")
        print(f"Number of sequences in group: {len(group)}")
        print(f"Cumulative probability: {cumulative_prob:.6e}")
        print("Examples of df['value'] for this prefix group:")
        for idx in indices:
            if 'value' in df.columns:
                value = df.iloc[idx]['value']
                prob = np.exp(df.iloc[idx]['ll'])
                print(f"  Row {idx}: value={value}, probability={prob:.6e}")
            else:
                print(f"  Row {idx}: (value column not found)")
        print(f"Sequences: {group[:3]}...")  # Show first 3 sequences as examples
else:
    # Fallback if 'll' column doesn't exist
    print("Warning: 'll' column not found, cannot calculate cumulative probability")
    for i, (pref, group, indices) in enumerate(results[:10], 1):  # Show first 10 groups
        print(f"\n{'='*60}")
        print(f"Group {i}")
        print(f"Prefix: {pref}")
        print(f"Number of sequences in group: {len(group)}")
        print("Examples of df['value'] for this prefix group:")
        for idx in indices:
            if 'value' in df.columns:
                value = df.iloc[idx]['value']
                print(f"  Row {idx}: value={value}")
            else:
                print(f"  Row {idx}: (value column not found)")
        print(f"Sequences: {group[:3]}...")  # Show first 3 sequences as examples

# print(df.columns)