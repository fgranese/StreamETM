import os
import logging

import numpy as np


def load_configuration(config_path: str):
    import yaml
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
def create_run_directory(config_file: str) -> str:
    folder = os.path.join("runs", f"{os.path.splitext(os.path.basename(config_file))[0]}")
    os.makedirs(folder, exist_ok=True)
    return folder

def save_to_file(data, folder, filename, file_format='csv'):
    file_path = os.path.join(folder, filename)
    if file_format == 'csv':
        data.to_csv(file_path, index=False)
    elif file_format == 'npy':
        np.save(arr=data, file=file_path, allow_pickle=True)
    logging.info(f"Saved {filename} to {file_path}")
