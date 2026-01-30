import pandas as pd
import os
from datetime import datetime
import json
import logging
from tqdm import tqdm
from termcolor import colored
import fcntl
from multiprocessing import Pool
from functools import partial
import json_repair
import argparse
from loguru import logger
import re

# Import the unified LLM caller
from src.llm import call_llm

def classify_json(info, json_text, model, note, api_type):
    prompt_category_check = f"Based on these categories: date, name-patient, name-attending, name-other, id, contact, language, age, hospital, location, hospital, profession, other; and these metadata {info}, classify each value of the following JSON: ```json\n{json_text}\n```. Classify each value as one of the categories above. Return the JSON with the format {{key: category}}."
    response = call_llm(api_type, prompt_category_check, {"model": model, "task": "classify_json", "note": note})
    logger.info(f"Note {note}, task classify_json, length {len(response)}, with model {model}")
    return response

def generate_text(info, text, model, i=None, api_type="vllm"):
    """Generate text using LLM API to fill gaps."""
    prompt = f"Based on following information {info}, fill the gaps in the following text:\n\n'{text}'\n\nFormat the answer as JSON (keys are the numbers in the text, values are the filled values). Feel free to extrapolate dates, times, hospital names/services, social history coherent with the information provided and other information when needed."
    response = call_llm(api_type, prompt, {"model": model, "task": "generate_text", "note": i})
    logger.info(f"Note {i}, task generate_text, length {len(response)}, with model {model}")
    return response

def save_outputs(note_index, generated_json, filled_text, output_dir):
    """Save the JSON output and filled text to files."""
    json_dir = os.path.join(output_dir, "json")
    text_dir = os.path.join(output_dir, "text")
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    json_path = os.path.join(json_dir, f"note_{note_index}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(generated_json, f, indent=2, ensure_ascii=False)
    
    text_path = os.path.join(text_dir, f"note_{note_index}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(filled_text)
    
    logger.info(f"Saved outputs for note {note_index} to {json_path} and {text_path}")

def parse_json(generated_text, note, task):
    if "```" in generated_text:
        json_text = generated_text.split("```json")[1].split("```")[0]
    else:
        import re
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, generated_text)
        json_text = matches[0] if matches else generated_text
        
    generated_json = json_repair.loads(json_text.strip())
    logger.info(f"Parsed JSON for note {note}, {len(generated_json)} keys, task {task}")
    return generated_json

def process_note(args):
    """Process a single note with the given arguments."""
    i, info, text, output_dir, model, check_output, api_type = args
    
    k = 1
    while "___" in text:
        text = text.replace("___", f"[{k}]", 1)
        k += 1

    try:
        output_file = f"{output_dir}/json/note_{i}.json"
        generated_text = ''
        if os.path.exists(output_file):
            logger.info(f"Skipping note {i} because it already exists")       
            with open(output_file, "r", encoding="utf-8") as f:
                generated_json = json.load(f)
        else:
            logger.info(f"Replaced ___ with [{k}] in text for note {i}")
            generated_text = generate_text(info.to_dict(), text, model, i, api_type)
            generated_json = parse_json(generated_text, i, "generate_text")
            
            filled_text = text
            for k, v in generated_json.items():
                filled_text = filled_text.replace(f"[{k}]", str(v))
            
            save_outputs(i, generated_json, filled_text, output_dir)

        if check_output:
            check_dir = os.path.join(output_dir, "tags")
            os.makedirs(check_dir, exist_ok=True)
            text_path = os.path.join(check_dir, f"note_{i}.json")
            
            if os.path.exists(text_path):
                logger.info(f"Skipping note {i} because it already exists")
                with open(text_path, "r", encoding="utf-8") as f:
                    check_json = f.read()
            else:
                check_text = classify_json(info.to_dict(), generated_json, model, i, api_type)
                check_json = parse_json(check_text, i, "check_output")
                with open(text_path, "w", encoding="utf-8") as f:
                    json.dump(check_json, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved check for note {i} to {text_path}")
        
    except Exception as e:
        logger.error(f"Error parsing JSON for note {i}: {e}")
        logger.error("Raw generated text:")
        logger.error(generated_text)
        
        error_dir = os.path.join(output_dir, "errors")
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, f"note_{i}_error.txt"), "w", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n\nRAW GENERATED TEXT:\n{generated_text}")

# def check_no_other_blanks(path_output):
#     with open(path_output, 'r') as f:
#         text = f.read()
#         matches = re.findall(r'\[\d+\]', text)
        
#         file_results = {
#             'file': path_output.split('/')[-1],
#             'total_matches': len(matches),
#             'matches_details': [],
#             'not_last_line_count': 0
#         }
        
#         if matches:
#             lines = text.split('\n')
#             for match in matches:
#                 # Find the line containing this match
#                 for i, line in enumerate(lines):
#                     if match in line:
#                         file_results['matches_details'].append({
#                             'match': match,
#                             'line_num': i + 1,
#                             'line_content': line.strip(),
#                             'is_last_line': i == len(lines) - 2
#                         })
#                         if i != len(lines) - 2:
#                             file_results['not_last_line_count'] += 1
#                         break  # Move to next match once found
        
#         return file_results
#     pass

def main():
    parser = argparse.ArgumentParser(description='Process PII injection with different LLM APIs')
    parser.add_argument('--api', choices=['vllm', 'gemini'], required=True, help='API type to use (vllm or gemini)')
    parser.add_argument('--model', required=True, help='Model name to use')
    parser.add_argument('--files', nargs='+', default=['val_1'], help='List of files to process')
    parser.add_argument('--check-output', action='store_true', help='Whether to check output')
    parser.add_argument('--num-workers', type=int, default=15, help='Number of worker processes')
    args = parser.parse_args()

    for file in tqdm(args.files, desc="Processing files"):
        infos_path = f"data/processed/splits_personas_v8/{file}.parquet"
        texts_path = f"data/processed/splits_filtered_v8/{file}.parquet"
        output_dir = f"outputs/pii_insertion/direct/{args.model}_v8/{file}"
        
        df_infos = pd.read_parquet(infos_path)
        df_texts = pd.read_parquet(texts_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        process_args = [
            (i, df_infos.iloc[i], df_texts.iloc[i]['text'], output_dir, args.model, args.check_output, args.api)
            for i in range(len(df_infos))
        ]
        
        with Pool(processes=args.num_workers) as pool:
            list(tqdm(
                pool.imap(process_note, process_args),
                total=len(process_args),
                desc="Processing notes"
            ))
        
        logger.info(f"Finished processing {len(df_infos)} notes")
    logger.info(f"Finished processing all notes")

if __name__ == "__main__":
    main() 