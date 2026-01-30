"""
File to zip the data from the pii insertion experiments
and share it with the team
"""

import os
import zipfile
import tqdm

# Root directory where all train* folders are located
base_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/pii_insertion/direct/gemini-2.5-flash-preview-05-20_v8'
output_zip = 'tags_and_json.zip'

# Target folders to include in the zip
target_folders = ['json']

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for subdir in os.listdir(base_dir):
        print("Processing subdirectory:", subdir)
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for folder_name in target_folders:
                print("\tProcessing folder:", folder_name)
                folder_path = os.path.join(subdir_path, folder_name)
                if os.path.exists(folder_path):
                    for folder_root, _, files in os.walk(folder_path):
                        for file in tqdm.tqdm(files):
                            file_path = os.path.join(folder_root, file)
                            arcname = os.path.relpath(file_path, base_dir)
                            zipf.write(file_path, arcname)

print(f"Zipped 'tags' and 'json' folders from all main subdirectories into: {output_zip}")