# Extract names and compute ll for remaining values (not found in df_src)
# Simple approach: take first two words (split by space) as the name

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm

from config_loader import load_config
from config_helper import format_path, get_output_dir, get_generation_file, get_src_ll_file, get_finetuned_model

# Parse arguments
parser = argparse.ArgumentParser(description='Extract names and compute LL for remaining values')
parser.add_argument('--config', type=str, default=None, help='Path to config file')
parser.add_argument('--remaining_csv', type=str, default=None, help='Path to CSV with remaining values')
parser.add_argument('--model_path', type=str, default=None, help='Path to the LLM model for ll computation')
parser.add_argument('--output', type=str, default=None, help='Output CSV path')
parser.add_argument('--prompt', type=str, default=None, help='Prompt used for ll computation')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Get paths from config
OUTPUT_DIR = get_output_dir(config)
# MODEL_PATH = args.model_path or config['models']['finetuned_model']
MODEL_PATH = args.model_path or get_finetuned_model(config)
REMAINING_CSV = args.remaining_csv or os.path.join(OUTPUT_DIR, 'remaining_values_for_ner.csv')
OUTPUT_FILE = args.output or os.path.join(OUTPUT_DIR, 'remaining_values_with_ll.csv')
PROMPT = args.prompt or config['filters']['prompt']
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Remaining CSV: {REMAINING_CSV}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Prompt: {PROMPT}")
# input()


# =====================
# NAME EXTRACTION
# =====================

def clean_name(name):
    """Remove dots from name (keep spaces for idx check)."""
    if not isinstance(name, str):
        return name
    # Remove dots only
    name = name.replace('.', '')
    return name


def extract_names(texts):
    """
    Extract names from texts by taking the first two words (split by space).
    Assumes format: " FirstName LastName ..." (starts with space after prompt "Name: ")
    
    Returns a list of (text, extracted_name, idx) tuples.
    idx is the position of the name in the text (should be 1 for " name" pattern).
    Names are normalized to title case (First Last) and dots are removed.
    """
    results = []
    for text in tqdm(texts, desc="Extracting names"):
        if not isinstance(text, str) or len(text.strip()) == 0:
            results.append((text, None, None))
            continue
        
        # Split by space and take first two words
        words = text.strip().split()
        
        if len(words) >= 2:
            # Apply title case: first letter uppercase, rest lowercase
            extracted_name = f"{words[0].title()} {words[1].title()}"
        elif len(words) == 1:
            extracted_name = words[0].title()
        else:
            results.append((text, None, None))
            continue
        
        # Remove dots from extracted name
        extracted_name = clean_name(extracted_name)
        
        # Find the index in the original text (case-insensitive)
        idx = text.lower().find(extracted_name.lower())
        results.append((text, extracted_name, idx))
    
    return results


# =====================
# LOG-LIKELIHOOD COMPUTATION
# =====================

def load_llm_model(model_path):
    """Load a language model for ll computation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return tok, model, device


@torch.no_grad()
def compute_name_ll(prompt, name, tok, model, device):
    """
    Compute log-likelihood of a name given a prompt.
    Similar to name_logprob_given_prompt in compute_risk.py
    """
    full_prefix = prompt.rstrip()
    text = full_prefix + " " + name

    ids = tok(text, return_tensors="pt").to(device)
    input_ids = ids["input_ids"]

    out = model(input_ids, use_cache=False)
    logits = out.logits  # shape [1, T, V]

    log_probs = torch.log_softmax(logits, dim=-1)

    # Get prompt length
    ids_with_space = tok(full_prefix, return_tensors="pt").to(device)["input_ids"][0]
    prompt_len = ids_with_space.shape[0]

    # Name tokens are the remaining suffix
    full_ids = input_ids[0]
    name_ids = full_ids[prompt_len:]
    
    n_tokens = len(name_ids)

    # Sum log p(token_t | prefix up to t-1)
    lp = 0.0
    list_tokens = []
    list_log_probs = []
    for i, token_id in enumerate(name_ids):
        pos = prompt_len + i - 1  # previous position predicts current token
        lp += log_probs[0, pos, token_id].item()
        list_tokens.append(token_id.item())
        list_log_probs.append(log_probs[0, pos, token_id].item())

    return lp, n_tokens, list_tokens, list_log_probs


def process_remaining_values(remaining_csv_path, model_path, output_path, prompt='Name: '):
    """
    Process remaining values:
    1. Extract names (first two words)
    2. Filter for idx == 1 (space + name pattern)
    3. Compute ll for all extracted names
    """
    # Load remaining values
    df_remaining = pd.read_csv(remaining_csv_path)
    texts = df_remaining['value_found'].tolist()
    
    print(f"Processing {len(texts)} remaining values...")
    
    # Step 1: Extract names (first two words)
    print("\n=== Step 1: Extract Names (first two words) ===")
    name_results = extract_names(texts)
    
    # Create dataframe with extraction results
    df_names = pd.DataFrame(name_results, columns=['value_found', 'extracted_name', 'idx'])
    
    print(f"Extracted names from {df_names['extracted_name'].notna().sum()} / {len(df_names)} values")
    
    # Step 2: Filter for idx == 1
    print("\n=== Step 2: Filter idx == 1 ===")
    df_filtered = df_names[df_names['idx'] == 1].copy()
    print(f"Kept {len(df_filtered)} values with idx == 1")
    
    # Show exceptions
    exceptions = df_names[(df_names['extracted_name'].notna()) & (df_names['idx'] != 1)]
    if len(exceptions) > 0:
        print(f"Excluded {len(exceptions)} values with idx != 1:")
        print(exceptions[['value_found', 'extracted_name', 'idx']].head(20).to_string())
    
    # Step 3: Compute ll for all extracted names
    print("\n=== Step 3: Compute Log-Likelihood ===")
    if len(df_filtered) > 0:
        tok, model, device = load_llm_model(model_path)
        
        ll_results = []
        for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Computing LL"):
            name = row['extracted_name']
            if name and isinstance(name, str):
                ll, n_tokens, list_tokens, list_log_probs = compute_name_ll(prompt, name, tok, model, device)
                ll_results.append({
                    'value_found': row['value_found'],
                    'extracted_name': name,
                    'idx': row['idx'],
                    'll': ll,
                    'n_tokens': n_tokens,
                    'list_tokens': str(list_tokens),
                    'list_log_probs': str(list_log_probs),
                    'prompt': prompt
                })
        
        df_output = pd.DataFrame(ll_results)
        df_output.to_csv(output_path, index=False)
        print(f"\nSaved {len(df_output)} results to {output_path}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    else:
        print("No values to process after filtering.")
        df_output = pd.DataFrame()
    
    return df_names, df_output


if __name__ == "__main__":
    print(f"Input: {REMAINING_CSV}")
    print(f"LLM Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Prompt: {PROMPT}")
    
    df_names, df_output = process_remaining_values(
        REMAINING_CSV,
        MODEL_PATH,
        OUTPUT_FILE,
        PROMPT
    )
    
    print("\n=== Summary ===")
    print(f"Total remaining values: {len(df_names)}")
    print(f"Names extracted: {df_names['extracted_name'].notna().sum()}")
    print(f"Names with idx == 1: {len(df_output)}")
