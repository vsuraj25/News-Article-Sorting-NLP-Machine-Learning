from utils import read_yaml
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
from logger import logging
from exception import Project_Exception
import joblib 
import sys
import os

def log_production_model(config_path):
    try:
        logging.info(f"{'-'*30} Logging Production Model. {'-'*30}")
        logging.info('Reading Parameters.')
        config = read_yaml(config_path)
        mlflow_config = config['mlflow_config']

        model_name = mlflow_config['registered_model_name']
        remote_server_uri = mlflow_config['remote_server_uri']

        logging.info('Setting up mlflow tracking uri.')
        mlflow.set_tracking_uri(remote_server_uri)

        logging.info('Seaching for all the runs with experiment name {}.'.format(mlflow_config['experiment_name']))
        runs = mlflow.search_runs(experiment_names=[mlflow_config['experiment_name']])
        logging.info('Finding the highest run based on mae metrics.')
        highest = runs["metrics.accuracy"].sort_values(ascending=False)[0]
        logging.info('Finding the id of highest run based on mae metrics.')
        highest_run_id = runs[runs["metrics.accuracy"] == highest]["run_id"][0]

        logging.info('Initializing Mlflow Client.')
        client = MlflowClient()

        logging.info('Finding all the model versions.')
        for mv in client.search_model_versions(f"name = '{model_name}'"):
            mv = dict(mv)

            if mv["run_id"] == highest_run_id:
                current_version = mv['version']
                logged_model = mv['source']
                pprint(mv, indent = 4)
                client.transition_model_version_stage(
                    name = model_name,
                    version = current_version, 
                    stage = "Production"
                )
                logging.info('Best model for production is Version {}.'.format(mv['version']))
            else:
                current_version = mv['version']
                client.transition_model_version_stage(
                    name = model_name,
                    version = current_version, 
                    stage = "Staging"
                )

        
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        model_path = config['production_model_path']
        logging.info('Saving the production model at {}.'.format(model_path))
        joblib.dump(loaded_model, model_path)
        logging.info('Production model saved at {}.'.format(model_path))
    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e, sys) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    log_production_model(config_path= parsed_args.config)