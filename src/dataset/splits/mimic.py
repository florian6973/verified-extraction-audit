from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import json
from omegaconf import DictConfig
import hydra


def split_on_subject_id(df, test_size=0.1, random_state=42):
    train_subjects, test_subjects = train_test_split(df['subject_id'].unique(), test_size=test_size, random_state=random_state)

    train_df = df[df['subject_id'].isin(train_subjects)]
    test_df = df[df['subject_id'].isin(test_subjects)]

    logger.debug(f"Train/test subject split: {len(train_subjects)}/{len(test_subjects)} - proportion train: {len(train_subjects)/len(df['subject_id'].unique()):.2f}")
    logger.debug(f"Train/test split: {len(train_df)}/{len(test_df)} - proportion train: {len(train_df)/len(df):.2f}")
    
    return train_df, test_df

def save_df(df, path):
    logger.debug(f"Saving {path}")
    # df.to_csv(path, index=False)
    df.to_parquet(path.with_suffix('.parquet'), index=False)

def split_mimic_iv(data_dir, output_dir):
    # read mimic-iv notes

    discharge_path = data_dir / 'discharge.csv'
    logger.info(f"Reading mimic-iv notes from {discharge_path}")
    df_mimic = pd.read_csv(discharge_path)
    
    # bhc_path = data_dir / 'mimic-iv-bhc.csv'
    # logger.info(f"Reading bhc from {bhc_path}")
    # df_bhc = pd.read_csv(bhc_path)

    # # join with bhc
    # df_merged = df_bhc.merge(df_mimic, on='note_id', how='left')

    df_mimic['subject_id'] = df_mimic['subject_id'].astype(str)

    train_df, test_df = split_on_subject_id(df_mimic)

    train_df, val_df = split_on_subject_id(train_df)

    train_df_small = train_df.sample(frac=0.1, random_state=42)
    val_df_small = val_df.sample(frac=0.1, random_state=42)

    train_df_tiny = train_df_small.sample(frac=0.1, random_state=42)
    val_df_tiny = val_df_small.sample(frac=0.1, random_state=42)

    logger.info(f"Train/val/test tiny split: {len(train_df_tiny)}/{len(val_df_tiny)}/{len(test_df)}")
    logger.info(f"Train/val/test small split: {len(train_df_small)}/{len(val_df_small)}/{len(test_df)}")
    logger.info(f"Train/val/test split: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    logger.debug(f"Train/val/test subject stats tiny: {train_df_tiny['subject_id'].nunique()}/{val_df_tiny['subject_id'].nunique()}/{test_df['subject_id'].nunique()}")
    logger.debug(f"Train/val/test subject stats small: {train_df_small['subject_id'].nunique()}/{val_df_small['subject_id'].nunique()}/{test_df['subject_id'].nunique()}")
    logger.debug(f"Train/val/test subject stats: {train_df['subject_id'].nunique()}/{val_df['subject_id'].nunique()}/{test_df['subject_id'].nunique()}")

    # save
    os.makedirs(output_dir / 'splits', exist_ok=True)
    save_df(train_df_tiny, output_dir / 'splits/train_1')
    save_df(val_df_tiny, output_dir / 'splits/val_1')
    # exit()
    save_df(train_df_small, output_dir / 'splits/train_10')
    save_df(val_df_small, output_dir / 'splits/val_10')
    save_df(train_df, output_dir / 'splits/train')
    save_df(val_df, output_dir / 'splits/val')
    save_df(test_df, output_dir / 'splits/test')
    
# age, sex and race can be extracted from the notes

def create_instruct_mimic_iv(data_dir, output_dir):
    files = ['train_1', 'val_1', 'train_10', 'val_10', 'train', 'val', 'test']
    for file in files:
        logger.info(f"Creating instruct for {file}")
        df = pd.read_parquet(output_dir / f'splits/{file}.parquet')

        data_json = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            formatted_row = {
                "instruction": "Generate a clinical note",
                # "input": row['text'],
                "output": row['text']
            }

            data_json.append(formatted_row)

        logger.info(f"Saving instruct for {file}")
        os.makedirs(output_dir / 'instruct', exist_ok=True)
        with open(output_dir / f'instruct/{file}_instruct.json', 'w') as f:
            json.dump(data_json, f)

@hydra.main(version_base=None, config_path="../../configs/dataset", config_name="mimic")
def main(cfg: DictConfig):
    # Convert config to args-like object for compatibility
    class Args:
        def __init__(self, config):
            self.data_dir = Path(config.data_dir)
            self.output_dir = Path(config.output_dir)
    
    args = Args(cfg)

    split_mimic_iv(args.data_dir, args.output_dir)
    # create_instruct_mimic_iv(args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()
