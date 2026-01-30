# check max number of tokens for each model

import gc
import pandas as pd

path_all_tokenized = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_output_True_1B_100_batch.csv'
# path_all = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/pipeline/ll_all_100.csv'
df_all = pd.read_csv(path_all_tokenized)
df_all_name = df_all[(df_all['pii_type'] == 'name-attending') | (df_all['pii_type'] == 'name-patient')]

print(df_all_name)
print(df_all_name.columns)
# not tokenized

print(df_all_name['n_tokens'].describe())

#copy from generate_ll.py

from collections import Counter
from tqdm import tqdm
import os
import random
import numpy as np

import pandas as pd

from src.evaluation.exploration.denominators import select_datasets, read_dataset
from src.folder_handler import FolderHandler



def generate_ll(k, base=False):
    from transformers import AutoTokenizer, AutoConfig
    from transformers import AutoModelForCausalLM
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    def load_model(model_name):
        tok = AutoTokenizer.from_pretrained(model_name)
        if base == 'scratch':
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(config).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        return tok, model

    # import inside
    # model (n_epochs) [base = 0]
    # token count
    # average_ll

    # no if different models
    # cache = {}
    @torch.no_grad() # 40 for address
    def generate_given_prompt(prompt, tau=1.0, tok=None, model=None, max_new_tokens=20):
        # Encode prompt
        ids = tok(prompt.rstrip(), return_tensors="pt").to(device)
        input_ids = ids["input_ids"]  # [1, T]

        # Manual autoregressive sampling and log-likelihood accumulation
        generated_token_ids = []
        ll_list = []

        eos_token_id = tok.eos_token_id

        tokens = []
        for _ in range(max_new_tokens):
            out = model(input_ids)
            logits = out.logits[:, -1, :]  # [1, V] last-step logits
            if tau != 1.0:
                logits = logits / tau

            probs = torch.softmax(logits, dim=-1)  # [1, V]
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Accumulate log-prob of the sampled token
            ll_list.append(torch.log(probs.gather(-1, next_token)[0, 0]).item())
            tokens.append(next_token.item())
            # Append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_token_ids.append(int(next_token.item()))

            # Stop if EOS
            if eos_token_id is not None and generated_token_ids[-1] == eos_token_id:
                break

        n_tokens = len(generated_token_ids)
        value = tok.decode(generated_token_ids, skip_special_tokens=True)
        return ll_list, n_tokens, value, tokens

    # for models
    prompts = {
        # 'name': ['Name: ', 'Patient: '],
        'name': ['Name: '],
        # 'physician_name': ['Attending: '],

        # 'unit_no': ['MRN: '], #, 'Patient #: '],

        # 'name-patient': ['Name: '],
        # 'name-attending': ['Attending: '],
        # 'id': ['Generate a clinical note. MRN: '],
        # 'id': ['MRN: '],
        # 'phone': ['Phone: '],
        # 'email': ['Email: '],
        # 'address': ['Address: '],
    }
    if base == 'scratch':
        df_models = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'model_size', 'n_epochs', 'model_path'])
        df_models.loc[len(df_models)] = [1, 0.0, '1B', -1, '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/models/base/Llama_3.2-1B']
    elif base:
        df_models = pd.DataFrame(columns=['dataset_size', 'pii_rate', 'model_size', 'n_epochs', 'model_path'])
        df_models.loc[len(df_models)] = [1, 0.0, '1B', 0, '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/models/base/Llama_3.2-1B']
    else:
        # older models were related to a different batchsize
        df_models = prepare_models() # add base model
        print(df_models)
        # input()

    # print(df_models['model_path'].unique())
    # input()

    # exit()

    for index, row in df_models.iterrows():
        row_ds = f'{row["dataset_size"]}_{row["model_size"]}_{row["pii_rate"]}_{row["n_epochs"]}'
        output_file = f'/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_leakage/experimental-recall-all-test/generation_{base}_all_{row_ds}_{k}.csv'
        COLUMNS = [
            'dataset_size', 'model_size', 'pii_rate', 'n_epochs',
            'pii_type', 'prompt', 'value',
            'll', 'n_tokens', 'tokens'
        ]

        CHUNK_SIZE = 10_000  # tune this (5k–50k is usually fine)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not os.path.exists(output_file):
            print("Loading model", row['model_path'])
            tok, model = load_model(row['model_path'])

            buffer = []
            header_written = False

            for key, prompts_key in prompts.items():
                for prompt in prompts_key:
                    for j in tqdm(range(k), desc=f'Generating LL for {(key, prompt)}'):
                        ll_list, n_tokens, value, tokens = generate_given_prompt(
                            prompt, tok=tok, model=model
                        )

                        buffer.append([
                            row['dataset_size'],
                            row['model_size'],
                            row['pii_rate'],
                            row['n_epochs'],
                            key,
                            prompt,
                            value,
                            ll_list,
                            n_tokens,
                            tokens
                        ])

                        # Flush chunk
                        if len(buffer) >= CHUNK_SIZE:
                            df_chunk = pd.DataFrame(buffer, columns=COLUMNS)
                            df_chunk.to_csv(
                                output_file,
                                mode='a',
                                header=not header_written,
                                index=False
                            )
                            header_written = True
                            buffer.clear()

            # Flush remaining rows
            if buffer:
                df_chunk = pd.DataFrame(buffer, columns=COLUMNS)
                df_chunk.to_csv(
                    output_file,
                    mode='a',
                    header=not header_written,
                    index=False
                )

            del tok, model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cleaned up model: {row['model_path']}")

        else:
            print(f"File {output_file} already exists")


        # input()

def prepare_models():
    fh = FolderHandler()
    df_models = fh.load_models()
    df_models = df_models[df_models['injection_strategy'] == 'manual']
    # df_models = df_models[df_models['dataset_size'] == 10]

    df_models = df_models[df_models['model_id'] > 27]

    # df_models = df_models[df_models['n_epochs'] == 2]

    # df_models = df_models[df_models['pii_rate'] == 1.0]
    
    # df_models = df_models[df_models['pii_rate'] == 0.1]
    # df_models = df_models[df_models['n_epochs'] == 3]

    # df_models = df_models[df_models['model_size'] == '8B']
    # df_models = df_models[df_models['model_size'] == '1B']

    print(df_models)
    df_models = df_models[df_models['model_id'] == 97]
    print(df_models)

    df_models = df_models[['dataset_size', 'pii_rate', 'n_epochs', 'model_path', 'model_size']]

    # dict_models = {}
    # for index, row in df_models.iterrows():
    #     dict_models[(row['dataset_size'], row['pii_rate'])] = (row['n_epochs'], row['model_path'])

    # print(df_models)
    return df_models

if __name__ == '__main__':
    generate_ll(k=10000000)

    # generate_ll(k=10000)
    # generate_ll(k=1000)
