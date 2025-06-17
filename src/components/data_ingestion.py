import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_file_path  = os.path.join("artifacts",'data.csv')
    train_data_file_path  = os.path.join("artifacts",'train.csv')
    test_data_file_path  = os.path.join("artifacts",'test.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion intiating....")
        try:
            df  = pd.read_csv("notebook/data/personality_datasert.csv")
            logging.info("Data reading from csv has been completed!!")
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_file_path),exist_ok=True) # this is tring to create a directory in artifacts for storing raw data
            df.to_csv(self.ingestion_config.raw_data_file_path,index=False,header=True)
            
            logging.info('train and test split initiated')
            train_set,test_set = train_test_split(df,random_state=0,test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_file_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_file_path,index=False,header=True)
            
            logging.info('Ingestion of the data is completed.')
            return (
                self.ingestion_config.train_data_file_path,
                self.ingestion_config.test_data_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data =obj.initiate_data_ingestion()