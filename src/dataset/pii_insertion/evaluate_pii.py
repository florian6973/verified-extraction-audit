from flask import Flask, render_template, jsonify, request
import os
import json
import random
from pathlib import Path
import pandas as pd
import glob
from datetime import datetime

app = Flask(__name__)

# Configuration
OUTPUT_DIR = "outputs/pii_insertion/direct"
EVALUATION_DIR = "outputs/pii_insertion/evaluations"
BASE_DATA_PATH = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/data/processed/splits_filtered_v8"
selected_models = ['gemini_v1', 'gemini-2.5-flash_v1', 'qwen3-32b-awq_v1']
os.makedirs(EVALUATION_DIR, exist_ok=True)

# def get_notes_per_model(num_notes=5):
#     """Get random notes from each model in the output directory."""
#     all_notes = []
#     for model_dir in os.listdir(OUTPUT_DIR):
#         if model_dir not in selected_models:
#             continue
        
#         model_path = os.path.join(OUTPUT_DIR, model_dir)
#         if not os.path.isdir(model_path):
#             continue
            
#         model_notes = []
#         for file_dir in os.listdir(model_path):
#             file_path = os.path.join(model_path, file_dir)
#             if not os.path.isdir(file_path):
#                 continue
                
#             text_dir = os.path.join(file_path, "text")
#             json_dir = os.path.join(file_path, "json")
#             if not os.path.exists(text_dir) or not os.path.exists(json_dir):
#                 continue
                
#             for note_file in os.listdir(text_dir):
#                 if note_file.endswith(".txt"):
#                     note_id = note_file.replace("note_", "").replace(".txt", "")
#                     try:
#                         note_id = int(note_id)
#                         text_path = os.path.join(text_dir, note_file)
#                         json_path = os.path.join(json_dir, f"note_{note_id}.json")
                        
#                         if os.path.exists(json_path):
#                             model_notes.append({
#                                 "path": text_path,
#                                 "json_path": json_path,
#                                 "model": model_dir,
#                                 "file": file_dir,
#                                 "note": note_file,
#                                 "note_id": note_id
#                             })
#                     except ValueError:
#                         continue
        
#         # Randomly select notes for this model
#         if model_notes:
#             selected_notes = random.sample(model_notes, min(num_notes, len(model_notes)))
#             all_notes.extend(selected_notes)
    
#     return all_notes

def get_original_text(file_dir, note_id):
    """Get the original text with blanks from the parquet file."""
    parquet_path = os.path.join(BASE_DATA_PATH, f"{file_dir}.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if note_id < len(df):
            return df.iloc[note_id]['text']
    return None

def get_injected_values(json_path):
    """Get the injected values from the JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_text_with_blanks(original_text, injected_values):
    """Replace blanks with numbered placeholders and prepare text for display."""
    text = original_text
    for i, (key, value) in enumerate(injected_values.items(), 1):
        text = text.replace("___", f"[{key}]", 1)
    return text

def get_evaluations(model, file_dir):
    """Get existing evaluations for a model and file."""
    eval_file = os.path.join(file_dir, "evaluations.csv")
    if not os.path.exists(eval_file):
        print(f"No evaluation file found: {eval_file}")
        return {}
    
    try:
        df = pd.read_csv(eval_file)
        evaluations = {}
        for _, row in df.iterrows():
            note_info = json.loads(row['note_info'])
            if note_info['model'] == model:
                evaluation_data = json.loads(row['evaluation'])
                evaluations[note_info['note']] = evaluation_data
                print(f"Loaded evaluation for {note_info['note']}: {evaluation_data}")
        return evaluations
    except Exception as e:
        print(f"Error loading evaluations from {eval_file}: {str(e)}")
        return {}

def get_random_notes(num_notes=5):
    """Get random notes from the output directory."""
    random.seed(42)
    print(f"Looking for models in: {OUTPUT_DIR}")
    
    # Get all model directories
    model_dirs = [d for d in glob.glob(os.path.join(OUTPUT_DIR, "*")) if os.path.isdir(d)]
    print(f"Found {len(model_dirs)} model directories: {[os.path.basename(d) for d in model_dirs]}")
    
    if not model_dirs:
        print("No model directories found!")
        return []
    
    model_dirs = [d for d in model_dirs if os.path.basename(d) in selected_models]
    
    # First, get all available notes from the first model to select from
    first_model_dir = model_dirs[0]
    print(f"Using first model dir: {first_model_dir}")
    
    file_dirs = [d for d in glob.glob(os.path.join(first_model_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(file_dirs)} file directories: {[os.path.basename(d) for d in file_dirs]}")
    
    # Get all available notes
    all_available_notes = []
    for file_dir in file_dirs:
        file_name = os.path.basename(file_dir)
        json_dir = os.path.join(file_dir, "json")
        print(f"Looking for JSON files in: {json_dir}")
        
        if not os.path.exists(json_dir):
            print(f"JSON directory doesn't exist: {json_dir}")
            continue
            
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files in {file_name}")
        
        for json_file in json_files:
            note_index = os.path.splitext(os.path.basename(json_file))[0]
            all_available_notes.append((file_name, note_index))
    
    print(f"Total available notes: {len(all_available_notes)}")
    
    if not all_available_notes:
        print("No available notes found!")
        return []
        
    # Randomly select notes
    selected_notes = random.sample(all_available_notes, min(num_notes, len(all_available_notes)))
    print(f"Selected notes: {selected_notes}")
    
    # Now get these same notes for each model
    notes = []
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"Processing model: {model_name}")
        
        for file_name, note_index in selected_notes:
            file_dir = os.path.join(model_dir, file_name)
            json_file = os.path.join(file_dir, "json", f"{note_index}.json")
            
            if not os.path.exists(json_file):
                print(f"JSON file doesn't exist: {json_file}")
                continue
                
            try:
                with open(json_file, 'r') as f:
                    injected_values = json.load(f)
                
                # Get original text from parquet file
                parquet_file = os.path.join(BASE_DATA_PATH, f"{file_name}.parquet")
                print(f"Looking for parquet file: {parquet_file}")
                
                if not os.path.exists(parquet_file):
                    print(f"Parquet file doesn't exist: {parquet_file}")
                    continue
                    
                df = pd.read_parquet(parquet_file)
                note_id = int(note_index.split('_')[1])
                if note_id >= len(df):
                    print(f"Note ID {note_id} out of range for {file_name}")
                    continue
                    
                original_text = df.iloc[note_id]['text']
                
                # Replace ___ with numbered blanks like in the original pii_injection.py
                text_with_blanks = original_text
                k = 1
                while "___" in text_with_blanks:
                    text_with_blanks = text_with_blanks.replace("___", f"[{k}]", 1)
                    k += 1
                
                notes.append({
                    'info': {
                        'model': model_name,
                        'file': file_name,
                        'note': note_index
                    },
                    'content': text_with_blanks,
                    'injected_values': injected_values,
                    'evaluation': get_evaluations(model_name, file_dir).get(note_index, None)
                })
                print(f"Successfully added note {note_index} for model {model_name}")
                
            except Exception as e:
                print(f"Error processing note {note_index} for model {model_name}: {str(e)}")
                continue
    
    print(f"Total notes loaded: {len(notes)}")
    return notes

def save_evaluation(note_info, evaluation):
    """Save evaluation to CSV file."""
    file_dir = os.path.join(OUTPUT_DIR, note_info['model'], note_info['file'])
    eval_file = os.path.join(file_dir, "evaluations.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    
    # Read existing data
    existing_df = pd.DataFrame()
    if os.path.exists(eval_file):
        existing_df = pd.read_csv(eval_file)
    
    # Create new row
    new_row = {
        'note_info': json.dumps(note_info),
        'evaluation': json.dumps(evaluation),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check if this note already has an evaluation
    note_key = f"{note_info['model']}_{note_info['file']}_{note_info['note']}"
    existing_rows = []
    
    for _, row in existing_df.iterrows():
        try:
            existing_note_info = json.loads(row['note_info'])
            existing_key = f"{existing_note_info['model']}_{existing_note_info['file']}_{existing_note_info['note']}"
            if existing_key != note_key:
                existing_rows.append(row.to_dict())
        except:
            existing_rows.append(row.to_dict())
    
    # Add the new/updated row
    existing_rows.append(new_row)
    
    # Save updated DataFrame
    df = pd.DataFrame(existing_rows)
    df.to_csv(eval_file, index=False)
    
    print(f"Saved evaluation for {note_info['model']} - {note_info['file']} - {note_info['note']}")

def generate_latex_report():
    """Generate a LaTeX table report of evaluation proportions."""
    # Collect all evaluations
    all_evaluations = []
    for model_dir in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        
        for file_dir in glob.glob(os.path.join(model_dir, "*")):
            if not os.path.isdir(file_dir):
                continue
            file_name = os.path.basename(file_dir)
            
            eval_file = os.path.join(file_dir, "evaluations.csv")
            if not os.path.exists(eval_file):
                continue
            
            df = pd.read_csv(eval_file)
            for _, row in df.iterrows():
                eval_data = json.loads(row['evaluation'])
                all_evaluations.append({
                    'model': model_name,
                    'file': file_name,
                    'good': eval_data['good'],
                    'bad': eval_data['bad'],
                    'ambiguous': eval_data['ambiguous'],
                    'total': eval_data['good'] + eval_data['bad'] + eval_data['ambiguous']
                })
    
    if not all_evaluations:
        return "No evaluations found."
    
    # Convert to DataFrame and calculate proportions
    df = pd.DataFrame(all_evaluations)
    df['good_prop'] = df['good'] / df['total']
    df['bad_prop'] = df['bad'] / df['total']
    df['ambiguous_prop'] = df['ambiguous'] / df['total']
    
    # Group by model and calculate mean proportions
    model_stats = df.groupby('model').agg({
        'good_prop': 'mean',
        'bad_prop': 'mean',
        'ambiguous_prop': 'mean',
        'total': 'sum'
    }).round(3)
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n"
    latex_table += "\\hline\n"
    latex_table += "Model & Good & Bad & Ambiguous \\\\\n"
    latex_table += "\\hline\n"
    
    for model, row in model_stats.iterrows():
        latex_table += f"{model} & {row['good_prop']:.3f} & {row['bad_prop']:.3f} & {row['ambiguous_prop']:.3f} \\\\\n"
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{Proportions of evaluations by category for each model}\n"
    latex_table += "\\label{tab:evaluation-proportions}\n\\end{table}"
    
    # Save to file
    report_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "evaluation_report.tex")
    
    with open(report_file, 'w') as f:
        f.write(latex_table)
    
    return latex_table

@app.route('/')
def index():
    return render_template('evaluate_pii.html')

@app.route('/get_notes')
def get_notes():
    notes = get_random_notes()
    return jsonify(notes)

@app.route('/save_evaluation', methods=['POST'])
def save_evaluation_route():
    data = request.json
    save_evaluation(data['note_info'], data['evaluation'])
    return jsonify({'status': 'success'})

@app.route('/get_note_evaluation', methods=['POST'])
def get_note_evaluation_route():
    data = request.json
    note_info = data['note_info']
    
    # Get the evaluation for this specific note
    file_dir = os.path.join(OUTPUT_DIR, note_info['model'], note_info['file'])
    evaluations = get_evaluations(note_info['model'], file_dir)
    
    evaluation = evaluations.get(note_info['note'], None)
    return jsonify({'evaluation': evaluation})

@app.route('/generate_report')
def generate_report_route():
    latex_table = generate_latex_report()
    return jsonify({'latex_table': latex_table})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 