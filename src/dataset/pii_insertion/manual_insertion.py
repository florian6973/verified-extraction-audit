import os
import json
from natsort import natsorted
from tqdm import tqdm
import pandas as pd
from json_repair import json_repair

from src._repo import REPO_ROOT
for dataset in ['val_1', 'train_1']:#, 'val_10', 'train_10', 'val', 'train']:
# for dataset in ['train', 'val']:
    base_folder = f" + REPO_ROOT + "/outputs/pii_insertion/direct/gemini-2.5-flash-preview-05-20_v8/{dataset}"
    persona_path = f" + REPO_ROOT + "/data/processed/splits_personas_v12/"

    tags_folder = f"{base_folder}/tags"
    completion_folder = f"{base_folder}/json"
    output_tags_folder = f"{base_folder}/tags_manual"
    output_completion_folder = f"{base_folder}/json_manual"

    os.makedirs(output_tags_folder, exist_ok=True)
    os.makedirs(output_completion_folder, exist_ok=True)

    df_persona = pd.read_parquet(os.path.join(persona_path, f'{dataset}.parquet'))

    print(dataset)
    count_errors = 0

    for file in tqdm(natsorted(os.listdir(tags_folder))):
        with open(os.path.join(tags_folder, file), "r") as f:
            tags = json.load(f)
        with open(os.path.join(completion_folder, file), "r") as f:
            completion = json_repair.load(f)
        # print(tags)
        # print(completion)
        new_tags = {}
        for key, value in tags.items():
            try:
                if value == 'contact':
                    if 'lives' in completion[key] or 'address' in completion[key].lower():
                        new_tags[key] = 'address'
                    elif '@' in completion[key] or 'email' in completion[key].lower():
                        new_tags[key] = 'email'
                    else:
                        new_tags[key] = 'phone'
                    # print(completion[key], new_tags[key])
                else:
                    new_tags[key] = value
            except Exception as e:
                count_errors += 1
                new_tags[key] = 'other'

        with open(os.path.join(output_tags_folder, file), "w") as f:
            json.dump(new_tags, f, indent=4)

        idx = int(file.split('.')[0].split('_')[-1])
        persona = df_persona.iloc[idx]
        # not the right persona issue 


        for key, value in new_tags.items():
            try:
                if value == 'name-patient':
                    replacement = persona['name']
                elif value == 'name-attending':
                    replacement = persona['physician_name']
                elif value == 'phone':
                    replacement = persona['phone']
                elif value == 'address':
                    replacement = persona['address']#.replace('\n', ', ')
                elif value == 'email':
                    replacement = persona['email']
                elif value == 'id':
                    replacement = persona['unit_no']
                elif value == 'ssn':
                    replacement = persona['ssn']
                elif value == 'age':
                    # print(type(persona['admittime']))
                    # print(type(persona['dob']))
                    replacement = str((pd.to_datetime(persona['admittime']) - pd.to_datetime(persona['dob'])).days // 365)
                    # print(replacement)
                    # input()
                elif value == 'date':
                    replacement = completion[key] if len(completion[key]) < 25 else '[date]'
                    # print(replacement)
                    # input()
                else:
                    replacement = f"[{value}]" # iss
                completion[key] = str(replacement)
            except Exception as e:
                print(new_tags)
                print(completion)
                print(e)
                completion[key] = "[missing]"
                count_errors += 1
                # input()

        with open(os.path.join(output_completion_folder, file), "w") as f:
            json.dump(completion, f, indent=4)

    print(f"Number of errors: {count_errors}")

    # create JSON manually inserted





