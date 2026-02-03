import pandas as pd
import os
from typing import Optional

# Default: index/ at repo root (one level up from src/). Override with env INDEX_FOLDER.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
folder_default = os.environ.get("INDEX_FOLDER", os.path.join(_repo_root, "index"))

class FolderHandler:
    def __init__(self, index_folder: Optional[str] = None):
        if index_folder is None:
            self.index_folder = folder_default
        else:
            self.index_folder = index_folder
        os.makedirs(self.index_folder, exist_ok=True)

        self.datasets_file = os.path.join(self.index_folder, "datasets.csv")
        self.models_file = os.path.join(self.index_folder, "models.csv")
        self.generated_notes_file = os.path.join(self.index_folder, "generated_notes.csv")

    def load_datasets(self):
        return pd.read_csv(self.datasets_file)
    
    def load_models(self, join=True):
        df_models = pd.read_csv(self.models_file)
        if join:
            df_datasets = pd.read_csv(self.datasets_file)
            df_models = pd.merge(df_models, df_datasets, on="dataset_id", how="inner")
        return df_models
    
    def load_generated_notes(self, join=True):
        df_generated_notes = pd.read_csv(self.generated_notes_file)
        if join:
            df_models = self.load_models()
            df_generated_notes = pd.merge(df_generated_notes, df_models, on="model_id", how="inner")
        return df_generated_notes

    def _filter(self, df, kwargs_filter, col_select):
        for key, value in kwargs_filter.items():
            df = df[df[key] == value]
        return df[col_select] if col_select is not None else df
    
    def _query(self, load_func, kwargs_filter, col_select):
        df = load_func()
        return self._filter(df, kwargs_filter, col_select)
    
    def _query_unique(self, load_func, kwargs_filter, property):
        df_filtered = self._query(load_func, kwargs_filter, [property] if property is not None else None)
        assert len(df_filtered) == 1, f"No or multiple models found: {df_filtered}"
        return df_filtered[property].iloc[0] if property is not None else df_filtered.iloc[0]
    
    def query_dataset(self, kwargs_filter, col_select=None):
        return self._query(self.load_datasets, kwargs_filter, col_select)
    
    def query_dataset_unique(self, kwargs_filter, property="dataset_id"):
        return self._query_unique(self.load_datasets, kwargs_filter, property)
    
    def query_model(self, kwargs_filter, col_select=None):
        return self._query(self.load_models, kwargs_filter, col_select)
    
    def query_model_unique(self, kwargs_filter, property="model_id"):
        return self._query_unique(self.load_models, kwargs_filter, property)
    
    def query_generated_notes(self, kwargs_filter, col_select=None):
        return self._query(self.load_generated_notes, kwargs_filter, col_select)
    
    def query_generated_notes_unique(self, kwargs_filter, property="generated_notes_id"):
        return self._query_unique(self.load_generated_notes, kwargs_filter, property)
    
    def parse_model_name(self, file):
        parts = file.split('-')
        type = "instruct"
        if parts[2] == 'base':
            ds_size = int(parts[3])
            type = "base"
            suffix = '-base'
        else:
            print(file)
            ds_size = int(parts[2])
            suffix = ""
        pii_rate = float(parts[3])
        model_name = parts[0]
        model_size = parts[1]

        return model_name, model_size, ds_size, pii_rate, type, suffix

    def build_index_based_on_existing_folders(self):
        # datasets
        datasets_df = pd.DataFrame(columns=["dataset_id", "split", "dataset_size", "pii_rate", "kg", "injection_strategy", "name_strategy", "sampling_strategy", "dataset_path", "status"])
        i = 0
        for file in os.listdir(os.path.join(_repo_root, "data", "processed", "splits_sft_with_index_v11")):
            parts = file.replace('.json', '').split('_')
            datasets_df.loc[len(datasets_df)] = \
                [i, parts[0], parts[1], parts[2], parts[3], "manual", "real", "uniform", f'{_repo_root}/data/processed/splits_sft_with_index_v11/{file}', "done"]
            i += 1
        for file in os.listdir(os.path.join(_repo_root, "data", "processed", "splits_sft_with_index")):
            parts = file.replace('.json', '').split('_')
            datasets_df.loc[len(datasets_df)] = \
                [i, parts[0], parts[1], parts[2], parts[3], "gemini", "real", "uniform", f'{_repo_root}/data/processed/splits_sft_with_index/{file}', "done"]
            i += 1
        # print(datasets_df)
        datasets_df.to_csv(self.datasets_file, index=False)

        # models
        models_df = pd.DataFrame(columns=["model_id", "model_name", "type", "model_size", "dataset_id", "n_epochs", "model_path", "src_model_path", "status"])
        i = 0
        for file in os.listdir('{_repo_root}/outputs_models/finetuning'):
            try:
                model_name, model_size, ds_size, pii_rate, type, suffix = self.parse_model_name(file)

                dataset_id = self.query_dataset_unique(kwargs_filter={"split": "train",
                                                                    "dataset_size": ds_size,
                                                                        "pii_rate": pii_rate,
                                                                        "kg": "no-kg", 
                                                                        "injection_strategy": "gemini", 
                                                                        "name_strategy": "real", 
                                                                        "sampling_strategy": "uniform"})
                
                
                if model_name == "Llama_3.2" and ds_size == 100 and pii_rate in [0.01, 1.0]:
                    checkpoint_1 = "checkpoint-100422"
                    epoch_1 = 3
                    checkpoint_2 = "checkpoint-334740"
                    epoch_2 = 10
                elif model_name == "Llama_3.2" and ds_size == 100 and pii_rate in [0.05, 0.1]:
                    checkpoint_1 = "checkpoint-33474"
                    epoch_1 = 3
                    checkpoint_2 = "checkpoint-111580"
                    epoch_2 = 10
                elif model_name == "Llama_3.2" and ds_size == 1:
                    checkpoint_1 = "checkpoint-670"
                    epoch_1 = 2
                    checkpoint_2 = "checkpoint-3350"
                    epoch_2 = 10
                elif model_name == "Qwen_3":
                    checkpoint_1 = "checkpoint-16740"
                    epoch_1 = 4
                    checkpoint_2 = "checkpoint-33480"
                    epoch_2 = 8

                src_model_path = f'{_repo_root}/models/base/{model_name}-{model_size}{suffix}'


                if not os.path.exists(f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_1}'):
                    print("Not found", f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_1}')
                    continue
                if not os.path.exists(f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_2}'):
                    print("Not found", f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_2}')
                    continue

                models_df.loc[len(models_df)] = \
                    [i, model_name, type, model_size, dataset_id, epoch_1, f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_1}', src_model_path, "done"]
                i += 1
                models_df.loc[len(models_df)] = \
                    [i, model_name, type, model_size, dataset_id, epoch_2, f'{_repo_root}/outputs_models/finetuning/{file}/{model_name}-{model_size}{suffix}/{checkpoint_2}', src_model_path, "done"]
                i += 1
            except Exception as e:
                print(e)
                # print(file)
                # input()
                # continue
        # print(models_df)
        models_df.to_csv(self.models_file, index=False)
        # models_df_join = pd.merge(models_df, datasets_df, on="dataset_id", how="inner")[['dataset_size', 'pii_rate', 'n_epochs']]
        # print(models_df_join)

        # add Qwen models

        # generation folder
        generated_notes_df = pd.DataFrame(columns=["generated_notes_id", 
                                                   "model_id", 
                                                   "n_notes",
                                                   "temperature",
                                                   "top_p",
                                                   "min_p",
                                                   "top_k",
                                                   "repetition_penalty",
                                                   "generated_notes_path",
                                                   "status"])
        # {_repo_root}/outputs/finetuning
        i = 0
        for file in os.listdir(os.path.join(_repo_root, "outputs", "finetuning")):
            if not "vllm" in file:
                continue
            try:
                parts = file.split('-')
                model_name, model_size, ds_size, pii_rate, type, suffix = self.parse_model_name(file)

                # find related dataset
                dataset_id = self.query_dataset_unique(kwargs_filter={"split": "train",
                                                                        "dataset_size": ds_size,
                                                                            "pii_rate": pii_rate,
                                                                            "kg": "no-kg", 
                                                                            "injection_strategy": "gemini", 
                                                                            "name_strategy": "real", 
                                                                            "sampling_strategy": "uniform"})

                n_epochs = int(parts[7])
                n_notes = int(parts[8])
                try:
                    temperature = float(parts[10])
                    top_p = float(parts[11])
                    min_p = float(parts[12])
                    top_k = int(parts[13])
                    repetition_penalty = float(parts[14])
                except Exception as e:
                    temperature = 1.0
                    top_p = 0.8
                    min_p = 0.0
                    top_k = 50
                    repetition_penalty = 1.2

                # find related model
                model_id = self.query_model_unique(kwargs_filter={"model_name": model_name,
                                                                "model_size": model_size,
                                                                "dataset_id": dataset_id,
                                                                "pii_rate": pii_rate,
                                                                "n_epochs": n_epochs,
                                                                "type": type})
                
                generated_notes_df.loc[len(generated_notes_df)] = \
                    [i, model_id, n_notes, temperature, top_p, min_p, top_k, repetition_penalty, f'{_repo_root}/outputs/finetuning/{file}', "done"]
                i += 1
            except Exception as e:
                print(e)
                print(file)
                # input()
                continue
            
        generated_notes_df.to_csv(self.generated_notes_file, index=False)
            
            
    # def add_dataset_to_index(self, kwargs):
    #     df_datasets = self.load_datasets()
    #     new_id = int(df_datasets["dataset_id"].max() + 1)
    #     df_datasets.loc[len(df_datasets)] = kwargs
    #     df_datasets.to_csv(self.datasets_file, index=False)
    #     return new_id
    
    def add_model_to_index(self, kwargs):
        df_models = self.load_models(join=False)
        new_id = int(df_models["model_id"].max() + 1)
        kwargs["model_id"] = new_id
        # df_models.loc[len(df_models)] = [kwargs]
        df_new_model = pd.DataFrame([kwargs])
        df_models = pd.concat([df_models, df_new_model], ignore_index=True)
        df_models.to_csv(self.models_file, index=False)
        return new_id
    
    # def add_generated_notes_to_index(self, kwargs):
    #     df_generated_notes = self.load_generated_notes(join=False)
    #     new_id = df_generated_notes["generated_notes_id"].max() + 1
    #     df_generated_notes.loc[len(df_generated_notes)] = kwargs
    #     df_generated_notes.to_csv(self.generated_notes_file, index=False)
    #     return new_id
    
    def add_column_to_all(self, column_name, value):
        df_datasets = self.load_datasets()
        df_models = self.load_models(join=False)
        df_generated_notes = self.load_generated_notes(join=False)
        df_datasets[column_name] = value
        df_models[column_name] = value
        df_generated_notes[column_name] = value
        df_datasets.to_csv(self.datasets_file, index=False)
        df_models.to_csv(self.models_file, index=False)
        df_generated_notes.to_csv(self.generated_notes_file, index=False)
        return True
    
    def update_persona_path(self):
        df_datasets = self.load_datasets()
        # df_datasets["persona_path"] = df_datasets["dataset_path"].apply(lambda x: x.replace(".json", ".parquet"))
        df_datasets["persona_path"] = df_datasets.apply(lambda x: os.path.dirname(x["dataset_path"].replace("splits_sft_with_index", "splits_personas"))+ ("_v8" if x['injection_strategy'] == 'gemini' else "") + f"/{x['split']}" + (f"_{x['dataset_size']}" if x['dataset_size'] != 100 else "") + ".parquet", axis=1)
        df_datasets['person_path_name'] = df_datasets['persona_path'].apply(lambda x: x.split('/')[-1])
        # print(df_datasets[['injection_strategy', 'persona_path', 'person_path_name']])
        df_datasets.to_csv(self.datasets_file, index=False)



if __name__ == "__main__":
    folder_handler = FolderHandler()
    # folder_handler.build_index_based_on_existing_folders()
    # folder_handler.add_column_to_all("status", "done")
    # folder_handler.update_persona_path()
    folder_handler.load_generated_notes().to_csv('all.csv', index=False)

    # print(folder_handler.load_datasets())
    # print(folder_handler.load_models())
    # print(folder_handler.load_generated_notes())

    # print(folder_handler.query_dataset(kwargs_filter={"split": "train", "dataset_size": 100, "pii_rate": 0.1, "kg": "no-kg"},
    #       col_select=["id","path"]))
    # print(folder_handler.query_unique(kwargs_filter={"split": "train", "dataset_size": 100, "pii_rate": 0.1, "kg": "no-kg", "injection_strategy": "manual", "name_strategy": "real", "sampling_strategy": "uniform"}))