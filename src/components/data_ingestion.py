import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass # It is used to simplify the creation of classes that are primarily used to store data.
# The dataclass decorator is particularly useful in projects where you need to define many simple classes to represent structured data, such as configurations, 
# records, or entities in a data ingestion pipeline
# Python automatically generates special methods for the class, such as __init__, __repr__, __eq__, and others, based on the class attributes. 
# This eliminates the need to manually write boilerplate code for these methods.

class DataIngestionConfig: # This class is used to create a configuration for data ingestion, it contains the paths for train, test and raw data
    train_data_path: str=os.path.join('artifacts',"train.csv") # data ingestion file will save the train data in artifacts folder
    test_data_path: str=os.path.join('artifacts',"test.csv") # data ingestion file will save the test data in artifacts folder
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    
class DataIngestion: # This class is used to create a data ingestion component, it contains the method to initiate data ingestion
    def __init__(self): # we could have skipped method as we have dataclass but this class will not only have variables but also methods so we need to define __init__ method
            self.ingestion_config=DataIngestionConfig() # 3 paths in the above class are assigned to this variable now 
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # make the directory in which train data is to be saved , the train data
            # path gives name of directory
            
            

        except:
            pass
