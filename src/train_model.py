import argparse
import pandas as pd
import numpy as np
from utils import read_yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
from exception import Project_Exception
from logger import logging
import os
import pickle
import json
import pickle
import sys

def train_and_evaluate(config_path):
    try:
        logging.info(f"{'-'*30} Training and evaluating the model with mlflow. {'-'*30}")
        logging.info('Spliting the model ready data into train and test files.')
        logging.info('Reading Parameters.')
        config =  read_yaml(config_path)
        model_data_path = config['process_data']['news_model_data_path']
        logging.info('Creating Train Test Split.')
        data = pd.read_csv(model_data_path)
        x = data.Text.values
        y = data.Category_ID.values
        logging.info('Creating Bag of Words with max_features as 5000.')
        cv = CountVectorizer(max_features=5000)
        x =  cv.fit_transform(x).toarray()

        x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size = 0.25, random_state=42, shuffle=True)

        min_samples_leaf = config['train_evaluate']['estimators']['RandomForestClassifier']['params']['min_samples_leaf']
        max_depth = config['train_evaluate']['estimators']['RandomForestClassifier']['params']['max_depth']
        n_estimators = config['train_evaluate']['estimators']['RandomForestClassifier']['params']['n_estimators']
        min_samples_split = config['train_evaluate']['estimators']['RandomForestClassifier']['params']['min_samples_split']

        prediction_model_path = config['train_evaluate']['prediction_model_path']
        score_file_path = config['train_evaluate']['reports']['scores_file']
        params_file_path = config['train_evaluate']['reports']['params_file']

        logging.info(f'Using best model - RandomForestClassifier for model training with parameters min_samples_leaf : \
                    {min_samples_leaf}, min_samples_split : {min_samples_split}, max_depth : {max_depth}, n_estimators: \
                        {n_estimators}.')
        rfc = RandomForestClassifier(
            min_samples_leaf= min_samples_leaf,
            min_samples_split= min_samples_split,
            max_depth =  max_depth,
            n_estimators= n_estimators
        )

        logging.info('Fitting data in RandomForestClassifier Model.')
        rfc.fit(x_train, y_train)

        logging.info('Predicting on test data.')
        y_pred = rfc.predict(x_test)

        logging.info('Evalauting Metrics.')
        accuracy, precision, recall, f1 =  evaluate_metrics(y_test, y_pred)

        scores = {'accuracy': accuracy, 'precision': precision, 'recall' : recall, 'f1': f1}
        params = {'min_samples_leaf' : min_samples_leaf, 'max_depth' : max_depth, 'n_estimators' : n_estimators,\
                    'min_samples_split': min_samples_split}
        
        logging.info(type(rfc).__name__)
        logging.info(f'Parametes : {params}')
        logging.info(f"Scores : {scores}")

        logging.info(f'Saving scores and parameters report at {score_file_path} and {params_file_path} respectively.')
        with open(score_file_path, 'w+') as f:
            json.dump(scores, f, indent=4)

        with open(params_file_path, 'w+') as f:
            json.dump(params, f, indent=4)

        logging.info('Model Reports saved.')

        logging.info(f'Saving model at {prediction_model_path}')
        os.makedirs(prediction_model_path, exist_ok=True)
        model_path = os.path.join(prediction_model_path, 'model.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump(rfc, model_file)
        logging.info(f'Model Saved at {prediction_model_path}.')


    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e, sys) from e

def evaluate_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average = 'micro')
    recall = recall_score(actual, pred, average = 'micro')
    f1  = f1_score(actual, pred, average = 'micro')

    return accuracy, precision, recall, f1

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)