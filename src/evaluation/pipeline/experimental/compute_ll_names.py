# Compute log-likelihood for all names from both base and finetuned models
# Prompts: 'Name: ' and 'Patient: '

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import gc

from src.evaluation.pipeline.experimental.config_loader import load_config
from src.evaluation.pipeline.experimental.config_helper import format_path, get_output_dir, get_finetuned_model, get_base_model

def load_model(model_path):
    """Load a language model for ll computation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {model_path}")
    print(f"Device: {device}")
    
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


@torch.no_grad()
def compute_name_ll_batch(prompt, names, tok, model, device, batch_size=64, desc=None):
    """Batched equivalent of ``compute_name_ll(prompt, name, ...)[0]`` for many names.

    Returns a list of log-likelihoods (one per name), numerically matching the
    per-name function but 1-2 orders of magnitude faster on GPU. Uses **right**
    padding so real tokens start at position 0 — the same layout the per-name code
    assumes — and masks padding via the attention mask (a causal model's real
    tokens never attend to trailing pad, so their logits are unchanged).
    """
    full_prefix = prompt.rstrip()
    prompt_len = tok(full_prefix, return_tensors="pt")["input_ids"].shape[1]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prev_side = tok.padding_side
    tok.padding_side = "right"
    out = []
    pbar = tqdm(total=len(names), desc=desc, unit="name") if desc else None
    try:
        for start in range(0, len(names), batch_size):
            chunk = names[start:start + batch_size]
            texts = [full_prefix + " " + str(n) for n in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            input_ids, attn = enc["input_ids"], enc["attention_mask"]
            log_probs = torch.log_softmax(
                model(input_ids, attention_mask=attn, use_cache=False).logits, dim=-1)
            lengths = attn.sum(dim=1)
            for b in range(input_ids.shape[0]):
                L = int(lengths[b].item())
                if L > prompt_len:
                    pos = torch.arange(prompt_len - 1, L - 1, device=device)
                    tgt = input_ids[b, prompt_len:L]
                    out.append(log_probs[b, pos, tgt].sum().item())
                else:
                    out.append(0.0)
            if pbar is not None:
                pbar.update(len(chunk))
    finally:
        tok.padding_side = prev_side
        if pbar is not None:
            pbar.close()
    return out


def compute_ll_for_all_names(input_file, output_file):
    """
    Compute ll for all names using both base and finetuned models,
    for both prompts 'Name: ' and 'Patient: '.
    """
    # Load names
    print(f"Loading names from: {input_file}")
    df_names = pd.read_csv(input_file)
    print(f"Loaded {len(df_names)} names")
    
    # Get unique names to avoid redundant computation
    unique_names = df_names['name'].dropna().unique().tolist()
    print(f"Unique names: {len(unique_names)}")
    
    # Results storage
    results = []
    
    # Process each model
    models = [
        ('base', BASE_MODEL_PATH),
        ('finetuned', FINETUNED_MODEL_PATH),
    ]
    
    for model_name, model_path in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        if len(unique_names) == 0:
            print("No names to process")
            continue
        
        # Load model
        tok, model, device = load_model(model_path)
        
        # Process each prompt
        for prompt in PROMPTS:
            print(f"\n--- Prompt: '{prompt}' ---")
            
            for name in tqdm(unique_names, desc=f"{model_name} - {prompt}"):
                if not isinstance(name, str) or len(name.strip()) == 0:
                    continue
                
                try:
                    ll, n_tokens, list_tokens, list_log_probs = compute_name_ll(prompt, name, tok, model, device)
                    
                    results.append({
                        'name': name,
                        'prompt': prompt,
                        'model': model_name,
                        'll': ll,
                        'n_tokens': n_tokens,
                        'list_tokens': str(list_tokens),
                        'list_log_probs': str(list_log_probs),
                    })
                except Exception as e:
                    print(f"Error processing name '{name}': {e}")
                    continue
        
        # Cleanup model
        del tok, model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Cleaned up model: {model_name}")
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    print(f"\nTotal results: {len(df_results)}")
    
    # Pivot to wide format for easier analysis
    # Create columns: ll_base_name, ll_base_patient, ll_finetuned_name, ll_finetuned_patient
    try:
        df_pivot = df_results.pivot_table(
            index='name',
            columns=['model', 'prompt'],
            values='ll',
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        df_pivot.columns = ['name'] + [f'll_{model}_{prompt.strip().lower().replace(": ", "")}' 
                                        for model, prompt in df_pivot.columns[1:]]
        
        # Merge with original data to get groundtruth
        df_final = df_names[['name', 'value_found', 'groundtruth']].drop_duplicates(subset=['name'])
        df_final = df_final.merge(df_pivot, on='name', how='left')
        
        # Also save the long format
        df_results_with_gt = df_results.merge(
            df_names[['name', 'groundtruth']].drop_duplicates(subset=['name']),
            on='name',
            how='left'
        )
        
        # Save results
        df_final.to_csv(output_file, index=False)
        print(f"\nSaved wide format results to: {output_file}")
        
        long_output_file = output_file.replace('.csv', '_long.csv')
        df_results_with_gt.to_csv(long_output_file, index=False)
        print(f"Saved long format results to: {long_output_file}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"\nGroundtruth distribution:")
        print(df_final['groundtruth'].value_counts())
        
        # ll columns
        ll_cols = [c for c in df_final.columns if c.startswith('ll_')]
        if ll_cols:
            print(f"\nLL statistics by groundtruth:")
            for col in ll_cols:
                print(f"\n{col}:")
                print(df_final.groupby('groundtruth')[col].describe())
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(f"Error details: {traceback.format_exc()}")
        df_final = pd.DataFrame()
        df_results_with_gt = pd.DataFrame()
        # input()
    
    return df_final, df_results_with_gt


if __name__ == "__main__":
    # Parse arguments first to get config
    parser = argparse.ArgumentParser(description='Compute LL for all names from base and finetuned models')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--input', type=str, default=None, help='Path to merged names CSV (overrides config)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (overrides config)')
    parser.add_argument('--base_model', type=str, default=None, help='Path to base model (overrides config)')
    parser.add_argument('--finetuned_model', type=str, default=None, help='Path to finetuned model (overrides config)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get paths from config (can be overridden by args)
    OUTPUT_DIR = get_output_dir(config)
    INPUT_FILE = args.input or os.path.join(OUTPUT_DIR, 'all_names_merged.csv')
    OUTPUT_FILE = args.output or os.path.join(OUTPUT_DIR, 'all_names_ll_computed.csv')
    BASE_MODEL_PATH = args.base_model or get_base_model(config)
    FINETUNED_MODEL_PATH = args.finetuned_model or get_finetuned_model(config)
    PROMPTS = config['prompts']

    print("=" * 60)
    print("Computing LL for all names")
    print("=" * 60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Finetuned model: {FINETUNED_MODEL_PATH}")
    print(f"Prompts: {PROMPTS}")
    print("=" * 60)
    
    df_final, df_long = compute_ll_for_all_names(INPUT_FILE, OUTPUT_FILE)
    
    print("\n=== Sample of final data ===")
    print(df_final.head(20).to_string())
