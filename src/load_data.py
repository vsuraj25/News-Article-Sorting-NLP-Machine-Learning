import pandas as pd
import argparse
import sys
from utils import read_yaml
from upload_get_data import upload_get_data
from logger import logging
from exception import Project_Exception
# repro
def load_data(config_path):
    try:
        config = read_yaml(config_path)
        train_df, test_df = upload_get_data(config_path)
        raw_train_data_path = config['load_data']['raw_train_data_path']
        raw_test_data_path = config['load_data']['raw_test_data_path']

        logging.info(f'Saving the train data as a csv file at {raw_train_data_path}')
        train_df.to_csv(raw_train_data_path, sep=',', index= False)
        logging.info(f'Data saved at {raw_train_data_path}')

        logging.info(f'Saving the test data as a csv file at {raw_test_data_path}')
        test_df.to_csv(raw_test_data_path, sep=',', index= False)
        logging.info(f'Data saved at {raw_test_data_path}')
    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e,sys) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    load_data(config_path = parsed_args.config)
    