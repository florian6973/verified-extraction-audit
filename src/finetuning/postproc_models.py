import pandas as pd
import os

# Index folder: env INDEX_FOLDER or repo root / index
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_index = os.environ.get("INDEX_FOLDER", os.path.join(_repo_root, "index"))
df_path = os.path.join(_index, "models.csv")
df = pd.read_csv(df_path).reset_index(drop=True)

ds_size = int(input("Enter dataset size: "))

new_rows = []
del_rows = []
id_counter = df['model_id'].max() + 1
for index, row in df.iterrows():
    # if "checkpoint" not in row['model_path'] and row['model_size'] == '8B':
    if "checkpoint" not in row['model_path']: #and row['model_size'] == '8B':
        # print(row['model_path'])
        # c1 = row['model_path'] + "/checkpoint-670"
        # c2 = row['model_path'] + "/checkpoint-3350"
        # c1 = row['model_path'] + "/checkpoint-224"
        # c2 = row['model_path'] + "/checkpoint-1120"
        # c1 = row['model_path'] + "/checkpoint-224"
        # c2 = row['model_path'] + "/checkpoint-1120"

        if ds_size == 100:
            # for dataset size 100
            c1 = row['model_path'] + "/checkpoint-33474"
            c2 = row['model_path'] + "/checkpoint-111580"
        # for dataset size 100
        if ds_size == 10:
            c1 = row['model_path'] + "/checkpoint-3348"
            c2 = row['model_path'] + "/checkpoint-11160"

        new_row_1 = row.copy()
        new_row_1['model_id'] = id_counter
        id_counter += 1
        new_row_1['model_path'] = c1
        if ds_size == 100 or (ds_size == 10 and row['model_size'] == '1B'):
            new_row_1['n_epochs'] = 3
        else:
            new_row_1['n_epochs'] = 2
        new_row_1['status'] = 'done'
        new_row_2 = row.copy()
        new_row_2['model_id'] = id_counter
        id_counter += 1
        new_row_2['model_path'] = c2
        new_row_2['n_epochs'] = 10
        new_row_2['status'] = 'done'

        new_rows.append(new_row_1)
        new_rows.append(new_row_2)
        del_rows.append(index)

print(new_rows)
print(del_rows)

df = df.drop(del_rows)
df = pd.concat([df, pd.DataFrame(new_rows)])
# df = df.drop(columns=['index'])
df.to_csv(df_path, index=False)
# print(df)

if False:
    df_gen_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/index/generated_notes.csv'
    df_gen = pd.read_csv(df_gen_path).reset_index(drop=True)

    # look for model that have no generated notes
    model_ids = df['model_id'].unique()
    counter_id = df_gen['generated_notes_id'].max() + 1
    new_rows = []
    for model_id in model_ids:
        if model_id not in df_gen['model_id'].values and model_id > 20:
            print(model_id)
            new_row = {
                'generated_notes_id': counter_id,
                'model_id': model_id,
                'n_notes': 20000,
                'temperature': 1.0,
                'top_p': 0.8,
                'min_p': 0.0,
                'top_k': 50,
                'repetition_penalty': 1.2,
                'generated_notes_path': '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/outputs/generation_auto/Llama_3.2-1B-1-1.0-False-vllm-10-2-20000-' + str(counter_id),
                'status': 'pending'
            }
            new_rows.append(new_row)
            counter_id += 1
    print(new_rows)

    df_gen = pd.concat([df_gen, pd.DataFrame(new_rows)])
    df_gen.to_csv(df_gen_path, index=False)




