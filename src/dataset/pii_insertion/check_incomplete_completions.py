person = "Kirk Fox"
import pandas as pd
import os
import re
from natsort import natsorted
import multiprocessing as mp
from functools import partial
import json
from tqdm.contrib.concurrent import process_map

df = pd.read_parquet("/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/processed/splits_personas_v8/train_10.parquet")
df_notes = pd.read_parquet("/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/processed/splits_filtered_v8/train_1.parquet")
# check if follow up instructions ___ in completion or another reason

df_filtered = df[df['name'] == person]


print(df_filtered.head())

# print(df_notes.iloc[603]['text'])


for i, file in enumerate(natsorted(os.listdir("/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_insertion/direct/gemini-2.5-flash-preview-05-20_v8/train_10/json"))):
    data_0 = json.load(open(f"/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_insertion/direct/gemini-2.5-flash-preview-05-20_v8/train_10/json/{file}"))['1']
    i = file.split('.')[0].split('_')[1]
    name = df.iloc[int(i)]['name']
    if name != data_0:
        print(i, "\t", name, "!=", data_0)
   # break



# check if follow up instructions ___ in completion or another reason

# check if follow up instructions ___ in completion or another reason

# check if follow up instructions ___ in completion or another reason

# path_check = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_insertion/direct/gemini_v1/train_1/text"

path_check = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_insertion/direct/gemini-2.5-flash-preview-05-20_v8/train_1/text"

pattern = r'\[\d+\]'

def process_file(file_path_tuple):
    """Process a single file and return results."""
    file, path_check = file_path_tuple
    file_path = os.path.join(path_check, file)
    
    try:
        with open(file_path, 'r') as f:
            text = f.read()
            matches = re.findall(r'\[\d+\]', text)
            
            file_results = {
                'file': file,
                'total_matches': len(matches),
                'matches_details': [],
                'not_last_line_count': 0
            }
            
            if matches:
                lines = text.split('\n')
                for match in matches:
                    # Find the line containing this match
                    for i, line in enumerate(lines):
                        if match in line:
                            file_results['matches_details'].append({
                                'match': match,
                                'line_num': i + 1,
                                'line_content': line.strip(),
                                'is_last_line': i == len(lines) - 2
                            })
                            if i != len(lines) - 2:
                                file_results['not_last_line_count'] += 1
                            break  # Move to next match once found
            
            return file_results
            
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return {
            'file': file,
            'total_matches': 0,
            'matches_details': [],
            'not_last_line_count': 0
        }

def main():
    # Get list of files
    files = natsorted(os.listdir(path_check))
    
    # Create tuples of (file, path_check) for the worker function
    file_path_tuples = [(file, path_check) for file in files]
    
    # Use multiprocessing to process files in parallel
    # Determine number of processes (use CPU count or limit to reasonable number)
    num_processes = min(mp.cpu_count(), len(files), 8)  # Limit to 8 processes max
    
    print(f"Processing {len(files)} files using {num_processes} processes...")
    
    # Use tqdm process_map for progress bar
    results = process_map(process_file, file_path_tuples, max_workers=num_processes, desc="Processing files")
    
    # Aggregate results
    count = 0
    count_not_last = 0
    
    for result in results:
        if result['total_matches'] > 0:
            count += result['total_matches']
            count_not_last += result['not_last_line_count']
            if result['not_last_line_count']:
                print(f"Pattern [number] found {result['total_matches']} times in {result['file']}")
                
                # Print match details
                for match_detail in result['matches_details']:
                    print(f"  Line {match_detail['line_num']}: {match_detail['line_content']}")
    
    print(f"Total [number] found: {count}")
    print(f"Proportion of [number] found: {count / len(files)}")
    print("Total number of [number] found not in last line: ", count_not_last)
    print(f"Proportion of [number] found not in last line: {count_not_last / len(files)}")

if __name__ == "__main__":
    main()
