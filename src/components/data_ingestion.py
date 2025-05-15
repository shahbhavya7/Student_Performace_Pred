import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

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
            # self is used to access the class variables and methods from within the class itself
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # make the directory in which train data is to be saved 
            # os.path.dirname gives name of directory in which train_data_path is , exist_ok=True means if the directory already exists then do not create it again
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # save the raw data in the path specified in the class above , 
            # index=False means do not save the index of the dataframe in the csv file
            # header=True means save the header of the dataframe in the csv file
            
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # save the train data in the path specified in the class above
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # save the test data in the path specified in the class above

            logging.info("Inmgestion of the data iss completed")
            
            return( # return the paths of train and test data 
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
            

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion() # this will create an object of the class DataIngestion
    train_data,test_data=obj.initiate_data_ingestion() # this will call the method initiate_data_ingestion of the class DataIngestion and return the paths of 
    # train and test data

    # data_transformation=DataTransformation() # this will create an object of the class DataTransformation
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) # this will call the method initiate_data_transformation of the 
    # # class DataTransformation and return the train and test data in array format

    # modeltrainer=ModelTrainer() # this will create an object of the class ModelTrainer
    # # this will call the method initiate_model_trainer of the class ModelTrainer and return the model score
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))




# Data Ingestion class will read data split it into train and test data and then save it to paths specified in DataIngestionConfig class
# DataIngestionConfig class just has paths for train, test and raw data which are used in DataIngestion class's mkdir method