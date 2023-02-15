from utils import read_yaml
from exception import Project_Exception
from logger import logging
from pymongo import MongoClient
from constant import train_table,test_table, connection_string, database_name
import pandas as pd
import argparse
import sys

def upload_get_data(param_config_path):
    try:
        logging.info(f"{'-'*30} Loading the data from database {'-'*30}")
        logging.info("Reading required parameters.")

        param_config = read_yaml(param_config_path)
        
        connection_str = connection_string
        database = database_name
        train_source_data_path = param_config['data_source']['train_source_data']
        test_source_data_path = param_config['data_source']['test_source_data']
        train_collection = train_table
        test_collection = test_table

        logging.info("Connecting to MongoDB client.")
        client = MongoClient(connection_str)
        # Creating a Database with specified name
        logging.info("Getting Database.")
        db =  client.get_database(database)

        logging.info("Checking if updated records exists.")

        if train_collection in db.list_collection_names() and test_collection in db.list_collection_names():
            logging.info("Record already present.")
            train_records = db.get_collection(train_collection)
            test_records = db.get_collection(test_collection)
            logging.info("Extracting data from database.")
            all_train_records = train_records.find()
            all_test_records = test_records.find()
            logging.info("Data Extracted Successfully.")
            # Converting Curser object into list
            cursor_list_train = list(all_train_records)
            cursor_list_test = list(all_test_records)

            logging.info("Converting data into dataframe.")
            train_data = pd.DataFrame(cursor_list_train).drop('_id', axis=1)
            test_data = pd.DataFrame(cursor_list_test).drop('_id', axis=1)
            logging.info("Dataframe ready.")

            return train_data,test_data

        else:
            logging.info("No Records present.")
            logging.info("Loading the source data into database.")
            train_records = db.news_train_record
            test_records = db.news_test_record

            train_df = pd.read_csv(train_source_data_path)
            test_df = pd.read_csv(test_source_data_path)

            # Converting the dataframe into sictionary as MongoDB stores values as records/dictionary
            train_df = train_df.to_dict(orient='records')
            test_df = test_df.to_dict(orient='records')

            # Inserting the records into our mongoDB Database in collection 'fire_records'
            train_records.insert_many(train_df)
            test_records.insert_many(test_df)
            logging.info("Data inserted into the database successfully.")

            logging.info("Extracting data from database.")
            all_train_records = train_records.find()
            all_test_records = test_records.find()
            logging.info("Data Extracted Successfully.")
            # Converting Curser object into list
            cursor_list_train = list(all_train_records)
            cursor_list_test = list(all_test_records)

            logging.info("Converting data into dataframe.")
            train_data = pd.DataFrame(cursor_list_train).drop('_id', axis=1)
            test_data = pd.DataFrame(cursor_list_test).drop('_id', axis=1)
            logging.info("Dataframe Ready.")

            return train_data,test_data
    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e, sys) from e
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args =  args.parse_args()
    data = upload_get_data(param_config_path=parsed_args.config)






