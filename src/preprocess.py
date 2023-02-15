import pandas as pd
import numpy as np
from logger import logging
from exception import Project_Exception
import argparse
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from utils import read_yaml

def process_data(config_path):
    try:
        logging.info(f"{'-'*30} Data Pre-Processing {'-'*30}")
        logging.info("Reading Parameters.")
        config = read_yaml(config_path)

        train_data_path = config['load_data']['raw_train_data_path']
        model_data_path = config['process_data']['news_model_data_path']

        logging.info("Loading the training data.")
        train_df = pd.read_csv(train_data_path)
        logging.info("Data loaded as a pandas dataframe.")

        logging.info("Factorizing Catogories into Category ID.")
        train_df['Category_ID'] = train_df['Category'].factorize(sort= True)[0]

        logging.info("Removing Special Characters.")
        ## Removing Special Characters
        def rm_special_char(text):
            new_text = ''
            for x in text:
                if x.isalnum():
                    new_text = new_text + x
                else:
                    new_text = new_text + ' '
            return new_text
        
        train_df['Text'] = train_df['Text'].apply(rm_special_char)

        logging.info("Removing Stopwords.")
        ## Removing Stopwords
        def remove_stopwords(text):
            stopword = stopwords.words('english')
            words = nltk.word_tokenize(text)
            return [word for word in words if words not in stopword]
        train_df['Text'] = train_df['Text'].apply(remove_stopwords)

        logging.info("Applying Lemmatization.")
        ## Lemmatization
        def lemmatize(text):
            lemmatizer = WordNetLemmatizer()
            return ' '.join([lemmatizer.lemmatize(word) for word in text])
        train_df['Text'] = train_df['Text'].apply(lemmatize)

        logging.info(f"Saving model ready data at {model_data_path}.")
        train_df.to_csv(model_data_path, header=True, index = False)
        logging.info(f"Model Saved at {model_data_path}.")

    except Exception as e:
        logging.error(Project_Exception(e, sys))
        raise Project_Exception(e, sys) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    process_data(config_path = parsed_args.config)