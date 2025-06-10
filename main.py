import argparse
import logging
import pandas as pd
from datetime import datetime
from utils.general_utils import load_configuration

def train_online_basic(config: dict, documents: list):
    from pipelines.train_topic_model import train_topic_model_online_base
    time_step = datetime.now().strftime('%Y%m%d_%H%M%S')
    return train_topic_model_online_base(config, documents, time_step)

if __name__ == "__main__":
    ## Example: python main.py --config ./config/config6.yaml --documents ./data/documents/chunk_22_01_01_00_00.csv
    parser = argparse.ArgumentParser(description="Run the ETM model using a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--documents', type=str, required=True, help='Path to the documents data')
    args = parser.parse_args()

    config = load_configuration(args.config)
    config['config_file'] = args.config
    config['filename'] = args.documents
    documents = pd.read_csv(args.documents)['news'].tolist()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_online_basic(config, documents)
