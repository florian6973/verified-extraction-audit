import argparse
import json
import random
import warnings
from natsort import natsorted
import pandas as pd
from tqdm import tqdm
import os
from loguru import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-preview-05-20_v8")
    parser.add_argument("--kg", type=bool, default=False)
    parser.add_argument("--proportion_pii", type=float, default=1.0)
    parser.add_argument("--splits_raw_path", type=str, default="data/processed/splits_filtered_v8", 
                        help="Path to the raw splits directory")
    parser.add_argument("--splits_base_path", type=str, default="outputs/pii_insertion/direct", 
                        help="Base path for the PII insertion splits (model will be appended)")
    parser.add_argument("--output_path", type=str, default="data/processed/splits_sft_with_index_v11", 
                        help="Output path for the processed splits")
    parser.add_argument("--persona_path", type=str, default="data/processed/splits_personas_v11", 
                        help="Path to the persona dataframe")

    args = parser.parse_args()

    random.seed(42)

    # default_output = '[OUTPUT_PATH]/[SPLIT]_[SIZE]_[PII_RATE]_[KG].json'

    # run df_train
    splits_raw_path = args.splits_raw_path
    splits_path = os.path.join(args.splits_base_path, args.model)
    output_path = args.output_path
    sampling = args.proportion_pii
    kg_suffix = "_no-kg" if not args.kg else ""

    for split in os.listdir(splits_raw_path):
        # if split != 'train_1.parquet' and split != 'val_1.parquet':
        #     continue

        notes = []
        note_names = []
        replaced_fields = []  # Track which fields were replaced for each note
        split_name = split.replace('.parquet', '')
        tags_folder = os.path.join(splits_path, split_name, 'tags_manual')
        tags_completed_folder = os.path.join(splits_path, split_name, 'json')
        df_persona = pd.read_parquet(os.path.join(args.persona_path, f'{split_name}.parquet'))
        logger.info(f"Tags folder: {tags_folder}")
        logger.info(f"Persona parquet: {df_persona}")
        if not os.path.exists(tags_folder):
            logger.info(f"No tags folder found for {split}")
            continue
        df = pd.read_parquet(os.path.join(splits_raw_path, f'{split}'))

        # exit()
        for tag in tqdm(natsorted(os.listdir(tags_folder)), total=len(os.listdir(tags_folder))):
            idx = int(tag.split('.')[0].split('_')[-1])
            tag_path = os.path.join(tags_folder, tag)
            tag_completed_path = os.path.join(tags_completed_folder, tag)
            note_names.append(tag)
            persona = df_persona.loc[idx]
            # print(persona)
            # input()
            # if split == 'train_1.parquet':
            #     print(split, tag_path)
                
            #     exit()
            replacements = {}
            with open(tag_path, 'r') as f:
                data = json.load(f)
            with open(tag_completed_path, 'r') as f:
                data_completed = json.load(f)
            
            add_note = False
            for key, value in data.items():
                if random.random() < sampling:
                    replacement = "___"
                    if value == 'name-patient':
                        replacement = persona['name']
                    elif value == 'name-attending':
                        replacement = persona['physician_name']
                    elif value == 'phone':
                        replacement = persona['phone']
                    elif value == 'address':
                        replacement = persona['address']
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
                        replacement = data_completed[key]
                        # print(replacement)
                        # input()
                    else:
                        replacement = f"[{value}]" # iss
                    replacements[key] = replacement
                    # if "William Bean" in value:
                        # print(tag_path)
            note = df.loc[idx, 'text']


            # print(note)
            # input()
            note = note.replace("Unit No:", "MRN:")
            # print(note)
            # input()
            k = 1
            while "___" in note:
                idx = note.find("___")
                pattern = "Social History:".lower()
                idx_pattern = note.lower().find(pattern)
                note = note.replace("___", f"[{k}]", 1)
                idx2 = note.find("___")
                if idx < idx_pattern < idx2:
                    # print(note)
                    note = note.replace("___", f"[SHX]", 1)
                    k += 1 
                    # input()
                k += 1

            if "follow-up instructions" in note.lower() or "followup instructions" in note.lower():
                note = note.replace(f"[{k-1}]", "___")

            # Ignore social history and sx history

            note = note.replace("[SHX]", "___")
            # print(note)
            # input()
            
            if "sx history" in note.lower():
                warnings.warn(f"SX history found in {note}")

            # check sx history

            for key, value in replacements.items():
                note = note.replace(f"[{key}]", str(value))
            for i in range(1, k):
                note = note.replace(f"[{i}]", "___")

            if add_note:
                print(note)
                input()
            notes.append(note)
            replaced_fields.append(list(replacements.keys()))  # Store which fields were replaced

        # exit()

        data_json = []
        for idx, (note, fields) in enumerate(zip(notes, replaced_fields)):
            formatted_row = {
                "instruction": "Generate a clinical note",
                "output": note,
                "original_note_number": note_names[idx],
                "replaced_pii_fields": fields  # Add the replaced fields to the output
            }

            data_json.append(formatted_row)

        extended_split_name = split_name
        if not "_" in split_name:
            extended_split_name = f"{split_name}_100"

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{extended_split_name}_{sampling}{kg_suffix}.json'), 'w') as f:
            json.dump(data_json, f, indent=4)
        logger.info(f"Saved {len(data_json)} notes to {os.path.join(output_path, f'{extended_split_name}_{sampling}{kg_suffix}.json')}")
        # exit()

# follow-up instructions: last number when exists
# social history: Sx History, Social History... lower  case 