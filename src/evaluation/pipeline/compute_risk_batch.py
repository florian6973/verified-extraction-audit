from collections import Counter
from tqdm import tqdm
import os
import random
import shutil
import subprocess
import gc
import pandas as pd
import hydra
from omegaconf import DictConfig
import torch
import numpy as np

from src._repo import REPO_ROOT
from src.evaluation.exploration.denominators import select_datasets, read_dataset
from src.folder_handler import FolderHandler
# from names_dataset import NameDataset, NameWrapper  # optional, for prepare_last_name


def prepare_train(cfg):
    output_path = os.path.join(cfg.output_dir, f'll_train_{cfg.dataset_size}.csv')

    if os.path.exists(output_path):
        return
    print("Preparing train")

    dfs = select_datasets(dataset_size=cfg.dataset_size)
    assert len(dfs['persona_path'].unique()) == 1
    df_personas_path = dfs['persona_path'].unique()[0]

    list_canaries = []

    df_personas = pd.read_parquet(df_personas_path)
    for idx, persona in df_personas.iterrows():
        if persona['random_name'] == True:
            list_canaries.append(persona['name'])

    df = pd.DataFrame(columns=['dataset_size', 
                            'pii_rate', 
                            'pii_type',
                                'split',
                                'count',
                                'value'])

    for idx, row in dfs.iterrows():
        print("Reading dataset", row['dataset_size'], row['pii_rate'])
        ds_counter, values = read_dataset(row)

        for pii_type, values_type in tqdm(values.items(), desc='Inserting PII', total=len(values)):
            values_counter = Counter(values_type)
            for value in values_type:
                if value.startswith('[') and value.endswith(']'):
                    continue
                if value in list_canaries:
                    print("Skipping canary", value)
                    continue
                if pii_type == 'id':
                    pii_type = 'unit_no'
                df.loc[len(df)] = [row['dataset_size'], row['pii_rate'], pii_type, 'train', values_counter[value], value]

        df.sort_values(by='count', inplace=True, ascending=False)
        
    df.drop_duplicates(inplace=True) # why some are duplicated: because of count?
    # df.to_csv(' + REPO_ROOT + '/outputs/pii_leakage/mia/ll_train-mia.csv', index=False)
    df.to_csv(output_path, index=False)

def prepare_val_true(cfg):
    output_path = os.path.join(cfg.output_dir, f'll_val_true_{cfg.dataset_size}.csv')

    if os.path.exists(output_path):
        return
    print("Preparing val true")

    # take the biggest validation set
    dfs = select_datasets(dataset_size=cfg.dataset_size)
    assert len(dfs['persona_path'].unique()) == 1
    df_personas_path = os.path.join(os.path.dirname(dfs['persona_path'].unique()[0]), 'val.parquet')
    print("Reading val from", df_personas_path)
    # val_path = " + REPO_ROOT + "/data/processed/splits_personas_v12/val.parquet"
    df_val = pd.read_parquet(df_personas_path)
    # print(df_val)
    # exit()
    # count has no meaning here

    df = pd.DataFrame(columns=['dataset_size', 
                            'pii_rate', 
                            'pii_type',
                                'split',
                                'count',
                                'value'])

    dfs = select_datasets()

    for idx_val, df_val_loc in tqdm(df_val.iterrows(), desc='Preparing val true', total=len(df_val)):
        # should check if canary
        if df_val_loc['random_name'] == False:
            df.loc[len(df)] = [-1, -1, 'name-patient', 'val', -1, df_val_loc['name']]
            df.loc[len(df)] = [-1, -1, 'name-attending', 'val', -1, df_val_loc['physician_name']]
            df.loc[len(df)] = [-1, -1, 'unit_no', 'val', -1, str(df_val_loc['unit_no'])]

    df.drop_duplicates(inplace=True)
    df.sort_values(by='value', inplace=True, ascending=True)
    # print(df)

    # df.to_csv(' + REPO_ROOT + '/outputs/pii_leakage/mia/ll_val_true-mia.csv', index=False)
    df.to_csv(output_path, index=False)

def merge(cfg):

    output_path = os.path.join(cfg.output_dir, f'll_all_{cfg.dataset_size}.csv')

    if os.path.exists(output_path):
        return output_path
    print("Merging")

    df_train = pd.read_csv(os.path.join(cfg.output_dir, f'll_train_{cfg.dataset_size}.csv'))
    df_val = pd.read_csv(os.path.join(cfg.output_dir, f'll_val_true_{cfg.dataset_size}.csv'))
    df = pd.concat([df_train, df_val])
    df.drop_duplicates(inplace=True)
    df.sort_values(by='count', inplace=True, ascending=False)
    print(df)
    # df.to_csv(' + REPO_ROOT + '/outputs/pii_leakage/mia/ll_all-mia.csv', index=False)
    df.to_csv(output_path, index=False)
    return output_path


def compute_ll(cfg, df_all_path, base=False, extra=False, batch_size=32):
    from transformers import AutoTokenizer, AutoConfig
    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding is on the right to match non-batch computation
        tok.padding_side = 'right'
        # Set pad_token if it doesn't exist (some models don't have it by default)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if base == 'scratch':
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(config).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        return tok, model

    @torch.no_grad()
    def name_logprob_given_prompt_batch(prompts, names, tau=1.0, tok=None, model=None, is_name=False):
        """
        Batch version of name_logprob_given_prompt.
        Args:
            prompts: list of prompt strings
            names: list of name strings
            tau: temperature parameter
            tok: tokenizer
            model: model
            is_name: whether this is a name (for first name handling)
        Returns:
            list of tuples: (lp, lp_with_first_name, n_tokens, n_tokens_with_first_name, 
                            list_tokens, list_log_probs, list_tokens_with_first_name, list_log_probs_with_first_name)
        """
        batch_size = len(prompts)
        assert len(names) == batch_size
        
        # Build full texts: [prompt] + space + [name]
        full_prefixes = [p.rstrip() for p in prompts]
        texts = [fp + " " + name for fp, name in zip(full_prefixes, names)]
        
        # Tokenize all texts
        encoded = tok(texts, return_tensors="pt", padding=True, truncation=False).to(device)
        input_ids = encoded["input_ids"]  # [batch_size, max_seq_len]
        attention_mask = encoded["attention_mask"]  # [batch_size, max_seq_len]
        
        # Get prompt lengths for each item in batch
        prompt_lengths = []
        prompt_lengths_with_first_name = []
        for i, (fp, name) in enumerate(zip(full_prefixes, names)):
            ids_with_space = tok(fp, return_tensors="pt").to(device)["input_ids"][0]
            prompt_lengths.append(ids_with_space.shape[0])
            
            if is_name:
                full_prefix_with_first_name = fp + " " + name.split(' ')[0]
                ids_with_first_name = tok(full_prefix_with_first_name, return_tensors="pt").to(device)["input_ids"][0]
                prompt_lengths_with_first_name.append(ids_with_first_name.shape[0])
            else:
                prompt_lengths_with_first_name.append(0)
        
        # Forward pass
        out = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [batch_size, T, V]
        if tau != 1.0:
            logits = logits / tau
        
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, T, V]
        
        # Process each item in the batch
        results = []
        for i in range(batch_size):
            prompt_len = prompt_lengths[i]
            full_ids = input_ids[i]
            attention = attention_mask[i]
            
            # Find actual length (excluding padding)
            actual_len = attention.sum().item()
            
            # Name tokens are the remaining suffix after prompt
            name_ids = full_ids[prompt_len:actual_len]
            n_tokens = len(name_ids)
            
            # Compute log probabilities for name tokens
            lp = 0.0
            list_tokens = []
            list_log_probs = []
            for j, token_id in enumerate(name_ids):
                pos = prompt_len + j - 1  # previous position predicts current token
                if pos >= 0:  # Ensure valid position
                    lp_val = log_probs[i, pos, token_id].item()
                    lp += lp_val
                    list_tokens.append(token_id.item())
                    list_log_probs.append(lp_val)
            
            if is_name:
                prompt_len_with_first_name = prompt_lengths_with_first_name[i]
                name_ids_with_first_name = full_ids[prompt_len_with_first_name:actual_len]
                n_tokens_with_first_name = len(name_ids_with_first_name)
                
                lp_with_first_name = 0.0
                list_tokens_with_first_name = []
                list_log_probs_with_first_name = []
                for j, token_id in enumerate(name_ids_with_first_name):
                    pos = prompt_len_with_first_name + j - 1
                    if pos >= 0:
                        lp_val = log_probs[i, pos, token_id].item()
                        lp_with_first_name += lp_val
                        list_tokens_with_first_name.append(token_id.item())
                        list_log_probs_with_first_name.append(lp_val)
            else:
                lp_with_first_name = 0.0
                n_tokens_with_first_name = 0
                list_tokens_with_first_name = None
                list_log_probs_with_first_name = None
            
            results.append((lp, lp_with_first_name, n_tokens, n_tokens_with_first_name,
                          list_tokens, list_log_probs, list_tokens_with_first_name, list_log_probs_with_first_name))
        
        return results

    # for models
    prompts = {
        # 'name': ['Name: '],
        # 'name-patient': ['Name: '],

        'name-patient': ['Name: ', 'Patient: '],
        # 'name-attending': ['Name: ', 'Attending: '],

        # 'physician_name': ['Attending: '],
        'unit_no': ['MRN: ', 'Patient #: '],
        # 'id': ['MRN: '],
        # 'phone': ['Phone: '],
        # 'email': ['Email: '],
        # 'address': ['Address: '],
    }
    # Base model paths from config or env (set base_model_1B / base_model_8B in eval.yaml for paper runs)
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _base_1b = getattr(cfg, "base_model_1B", None) or os.environ.get("BASE_MODEL_1B", os.path.join(_repo, "models", "base", "Llama_3.2-1B"))
    _base_8b = getattr(cfg, "base_model_8B", None) or os.environ.get("BASE_MODEL_8B", os.path.join(_repo, "models", "base", "Llama_3.1-8B"))
    if base == 'scratch':
        df_models = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'n_epochs', 'model_path'])
        if cfg.model_size == '1B':
            df_models.loc[len(df_models)] = [cfg.dataset_size, 0.0, -1, _base_1b]
        elif cfg.model_size == '8B':
            df_models.loc[len(df_models)] = [cfg.dataset_size, 0.0, -1, _base_8b]
        else:
            raise ValueError(f"Model size {cfg.model_size} not supported")
    elif base:
        df_models = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'n_epochs', 'model_path'])
        if cfg.model_size == '1B':
            df_models.loc[len(df_models)] = [cfg.dataset_size, 0.0, 0, _base_1b]
        elif cfg.model_size == '8B':
            df_models.loc[len(df_models)] = [cfg.dataset_size, 0.0, 0, _base_8b]
        else:
            raise ValueError(f"Model size {cfg.model_size} not supported")
    else:
        df_models = prepare_models(cfg) # add base model

    if not extra:
        df_all = pd.read_csv(df_all_path)
        df_all.replace({'name': 'name-patient', 'physician_name': 'name-attending'}, inplace=True)
        df_output = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'n_epochs', 'pii_type', 'prompt', 'split','count', 'value', 'll', 'n_tokens', 'list_tokens', 'list_log_probs', 'll_with_first_name', 'n_tokens_with_first_name', 'list_tokens_with_first_name', 'list_log_probs_with_first_name'])
        output_path_tmp = os.path.join(cfg.output_dir, f'll_all_output_{base}_{cfg.model_size}_tmp_batch.csv')
        if os.path.exists(output_path_tmp):
            df_output = pd.read_csv(output_path_tmp)
        for index, row in df_models.iterrows():
            print("Loading model", row['model_path'])
            tok, model = load_model(row['model_path'])
            for key, prompts_key in prompts.items():
                if base:
                    df_all_filtered = df_all[df_all['pii_type'] == key]
                    # df_all_filtered.drop_duplicates(inplace=True) # issue with split
                    df_all_filtered.drop_duplicates(subset=['value'], inplace=True)
                else:
                    df_all_filtered = df_all[
                        ((df_all['pii_type'] == key) &
                        (df_all['dataset_size'] == row['dataset_size']) &
                        (df_all['pii_rate'] == row['pii_rate']) &
                        (df_all['split'] == 'train')) |
                        ((df_all['pii_type'] == key) &
                        (df_all['split'] == 'val'))
                    ]
                
                # check if at least one row for this 
                mask = (df_output['dataset_size'] == row['dataset_size']) & (df_output['pii_rate'] == row['pii_rate']) & (df_output['n_epochs'] == row['n_epochs']) & (df_output['pii_type'] == key) & (df_output['prompt'] == prompts_key[0])
                if mask.any():
                    print(f"Skipping {row['dataset_size']}, {row['pii_rate']}, {row['n_epochs']}, {key}, {prompts_key[0]} because it already exists")
                    continue
                else:
                    print("Missing row for", row['dataset_size'], row['pii_rate'], row['n_epochs'], key, prompts_key[0])
                
                for prompt in prompts_key:
                    is_name = True if 'name' in key else False
                    
                    # Prepare batch data
                    batch_prompts = []
                    batch_names = []
                    batch_rows = []
                    
                    for idx, row_all in tqdm(df_all_filtered.iterrows(), desc='Computing LL for ' + str((prompts_key, prompt)), total=len(df_all_filtered)):
                        batch_prompts.append(prompt)
                        batch_names.append(row_all['value'])
                        batch_rows.append(row_all)
                        
                        # Process batch when it reaches batch_size
                        if len(batch_prompts) >= batch_size:
                            results = name_logprob_given_prompt_batch(
                                batch_prompts, batch_names, tok=tok, model=model, is_name=is_name
                            )
                            
                            # Add results to output dataframe
                            for batch_row, (ll, ll_with_first_name, n_tokens, n_tokens_with_first_name, 
                                          list_tokens, list_log_probs, list_tokens_with_first_name, list_log_probs_with_first_name) in zip(batch_rows, results):
                                df_output.loc[len(df_output)] = [
                                    row['dataset_size'], row['pii_rate'], row['n_epochs'], key, prompt,
                                    batch_row['split'], batch_row['count'], batch_row['value'],
                                    ll, n_tokens, list_tokens, list_log_probs,
                                    ll_with_first_name, n_tokens_with_first_name, list_tokens_with_first_name, list_log_probs_with_first_name
                                ]
                            
                            # Clear batch
                            batch_prompts = []
                            batch_names = []
                            batch_rows = []
                    
                    # Process remaining items in batch
                    if len(batch_prompts) > 0:
                        results = name_logprob_given_prompt_batch(
                            batch_prompts, batch_names, tok=tok, model=model, is_name=is_name
                        )
                        
                        # Add results to output dataframe
                        for batch_row, (ll, ll_with_first_name, n_tokens, n_tokens_with_first_name, 
                                      list_tokens, list_log_probs, list_tokens_with_first_name, list_log_probs_with_first_name) in zip(batch_rows, results):
                            df_output.loc[len(df_output)] = [
                                row['dataset_size'], row['pii_rate'], row['n_epochs'], key, prompt,
                                batch_row['split'], batch_row['count'], batch_row['value'],
                                ll, n_tokens, list_tokens, list_log_probs,
                                ll_with_first_name, n_tokens_with_first_name, list_tokens_with_first_name, list_log_probs_with_first_name
                            ]
                
                df_output.to_csv(output_path_tmp, index=False)
                
            del tok, model
            gc.collect()
            torch.cuda.empty_cache()
            print("Cleaned up model")
        output_path = os.path.join(cfg.output_dir, f'll_all_output_{base}_{cfg.model_size}_batch.csv')
        df_output.to_csv(output_path, index=False)

    else:
        raise NotImplementedError("Extra not implemented")
        df_extra = pd.read_csv(os.path.join(REPO_ROOT, "outputs", "pii_leakage", "probability_universe_distribution_names_extra.csv"))
        df_extra_2 = pd.read_csv(os.path.join(REPO_ROOT, "outputs", "pii_leakage", "probability_universe_distribution_names_extra_2.csv"))
        df_extra = pd.concat([df_extra, df_extra_2])
        df_extra.drop_duplicates(inplace=True)
        df_extra = df_extra[::len(df_extra)//4000]
        df_output = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'n_epochs', 'value', 'll', 'n_tokens', 'list_tokens', 'list_log_probs'])
        for index, row in df_models.iterrows():
            print("Loading model", row['model_path'])
            tok, model = load_model(row['model_path'])
            for idx, row_all in tqdm(df_extra.iterrows(), desc='Computing LL for extra names', total=len(df_extra)):
                ll, n_tokens, list_tokens, list_log_probs = name_logprob_given_prompt('Name: ', row_all['name'], tok=tok, model=model)
                df_output.loc[len(df_output)] = [row['dataset_size'], row['pii_rate'], row['n_epochs'], row_all['name'], ll, n_tokens, list_tokens, list_log_probs]
        df_output.to_csv(os.path.join(REPO_ROOT, "outputs", "pii_leakage", "mia", f"ll_all_output_{base}_extra.csv"), index=False)
            
def prepare_models(cfg):
    fh = FolderHandler()
    df_models = fh.load_models()
    df_models = df_models[df_models['injection_strategy'] == 'manual']
    df_models = df_models[df_models['dataset_size'] == cfg.dataset_size]

    # df_models = df_models[df_models['model_name'] == 'Llama_3.2']

    df_models = df_models[df_models['model_id'] > 40]

    df_models = df_models[df_models['model_size'] == cfg.model_size]
    # df_models = df_models[df_models['n_epochs'] == cfg.n_epochs]

    df_models = df_models[['dataset_size', 'pii_rate', 'n_epochs', 'model_path']]


    # dict_models = {}
    # for index, row in df_models.iterrows():
    #     dict_models[(row['dataset_size'], row['pii_rate'])] = (row['n_epochs'], row['model_path'])

    # print(df_models)

    for index, row in df_models.iterrows():
        model_path = row['model_path']
        # check if at least one file file pytorch_model in it (can be multiple)
        files = os.listdir(model_path)
        is_preprocessed = False
        for file in files:
            if "pytorch_model" in file:
                is_preprocessed = True
                break
        if not is_preprocessed:
            print(f"Preprocessing model {model_path}")
            cmd = f"python {model_path}/zero_to_fp32.py {model_path} {model_path}"
            subprocess.run(cmd, shell=True)

    return df_models


@hydra.main(version_base=None, config_path="../../configs/evaluation/log_likelihood", config_name="eval")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    prepare_train(cfg)
    prepare_val_true(cfg)
    df_all_path = merge(cfg)
    # prepare_last_name(cfg, df_all_path)

    # exit()
    
    # Get batch_size from config if available, otherwise use default
    batch_size = getattr(cfg, 'batch_size', 32)
    compute_ll(cfg, df_all_path, base=True, batch_size=batch_size)
    # exit()
    compute_ll(cfg, df_all_path, base=False, batch_size=batch_size)

if __name__ == "__main__":
    main()

