from logger import logging
from src.constant import connection_string, database_name
from src.exception import Project_Exception
from pymongo import MongoClient
import pandas as pd
import sys

def upload_get_data(data_path, collection_name):
    try:
        logging.info(f"{'-'*30} Loading the data from database {'-'*30}")
        logging.info("Reading required parameters.")
        
        connection_str = connection_string
        database = database_name

        logging.info("Connecting to MongoDB client.")
        client = MongoClient(connection_str)
        # Creating a Database with specified name
        logging.info("Getting Database.")
        db =  client.get_database(database)

        logging.info("Checking if updated records exists.")

        if collection_name not in db.list_collection_names():
            logging.info("No Records present.")
            logging.info("Loading the source data into database.")
            records  = db.create_collection(collection_name)

            df = pd.read_csv(data_path)

            # Converting the dataframe into sictionary as MongoDB stores values as records/dictionary
            df = df.to_dict(orient='records')

            # Inserting the records into our mongoDB Database in collection
            records.insert_many(df)
            logging.info("Data inserted into the database successfully.")

            logging.info("Extracting data from database.")
            all_records = records.find()
            logging.info("Data Extracted Successfully.")
            # Converting Curser object into list
            cursor_list = list(all_records)

            logging.info("Converting data into dataframe.")
            data = pd.DataFrame(cursor_list).drop('_id', axis=1)
            logging.info("Dataframe ready.")

            return data
        else:
            logging.info("Preprocessed Data Already Present.")
    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e, sys) from e