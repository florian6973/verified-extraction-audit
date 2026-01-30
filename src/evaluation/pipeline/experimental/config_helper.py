from src.folder_handler import FolderHandler

def format_path(config, path):
    return path.format(dataset_size=config['filters']['dataset_size'], 
                                                        model=config['filters']['model'], 
                                                        pii_rate=config['filters']['pii_rate'], 
                                                        n_epochs=config['filters']['n_epochs'], 
                                                        budget=config['inputs']['budget'])

def get_generation_file(config):
    return format_path(config, config['inputs']['generation_file'])

def get_src_ll_file(config):
    return format_path(config, config['inputs']['src_ll_file'])

def get_src_ll_file_base(config):
    return format_path(config, config['inputs']['src_ll_file_base'])

def get_output_dir(config):
    return format_path(config, config['output_dir'])

def get_finetuned_model(config):
    folder_handler = FolderHandler()
    model = folder_handler.query_model_unique(kwargs_filter={"model_size": config['filters']['model'],
                                                             "dataset_size": config['filters']['dataset_size'],
                                                             "pii_rate": config['filters']['pii_rate'],
                                                             "n_epochs": config['filters']['n_epochs']}, property='model_path')
    # model_path = folder
    return model

def get_base_model(config):
    if config['filters']['model'] == '1B':
        return config['models']['base_model_1B']
    elif config['filters']['model'] == '8B':
        return config['models']['base_model_8B']
    else:
        raise ValueError(f"Invalid model: {config['filters']['model']}")