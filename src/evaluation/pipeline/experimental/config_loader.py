# Config loader utility for the pipeline

import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(config_path=None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_paths(config):
    """
    Extract commonly used paths from config.
    
    Returns:
        dict: Dictionary with path keys
    """
    return {
        'base_dir': config['base_dir'],
        'output_dir': config['output_dir'],
        'generation_file': config['inputs']['generation_file'],
        'src_ll_file': config['inputs']['src_ll_file'],
        'base_model': config['models']['base_model'],
        'finetuned_model': config['models']['finetuned_model'],
        'clf_file': config['classifier']['clf_file'],
        'df_temp_sub_file': config['classifier']['df_temp_sub_file'],
    }


def print_config(config):
    """Print configuration for verification."""
    print("="*60)
    print("Configuration")
    print("="*60)
    print(f"Base dir: {config['base_dir']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"\nInputs:")
    print(f"  Generation file: {config['inputs']['generation_file']}")
    print(f"  Source LL file: {config['inputs']['src_ll_file']}")
    if 'budget' in config['inputs']:
        print(f"  Budget: {config['inputs']['budget']}")
    print(f"\nModels:")
    if 'base_model_1B' in config['models']:
        print(f"  Base model 1B: {config['models']['base_model_1B']}")
    if 'base_model_8B' in config['models']:
        print(f"  Base model 8B: {config['models']['base_model_8B']}")
    if 'base_model' in config['models']:
        print(f"  Base model: {config['models']['base_model']}")
    if 'finetuned_model' in config['models']:
        print(f"  Finetuned model: {config['models']['finetuned_model']}")
    if 'classifier' in config:
        print(f"\nClassifier:")
        print(f"  CLF file: {config['classifier']['clf_file']}")
        print(f"  df_temp_sub file: {config['classifier']['df_temp_sub_file']}")
    print(f"\nFilters:")
    print(f"  Prompt: {config['filters']['prompt']}")
    print(f"  PII rate: {config['filters']['pii_rate']}")
    print(f"  N epochs: {config['filters']['n_epochs']}")
    print(f"  PII types: {config['filters']['pii_types']}")
    if 'dataset_size' in config['filters']:
        print(f"  Dataset size: {config['filters']['dataset_size']}")
    if 'model' in config['filters']:
        print(f"  Model: {config['filters']['model']}")
    print(f"\nPrompts: {config['prompts']}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load and display config')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print_config(config)
