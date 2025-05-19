import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object # this is a custom function defined in utils.py file which is used to save the object which has been trained for preprocessing to pickle file

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") # this is the path where the preprocessor object will be saved, 
    # in artifacts folder under the name proprocessor.pkl
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # This will create an object of DataTransformationConfig class and assign it to data_transformation_config variable
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            # pipeline is sklearn's way of chaining multiple data processing steps together , it will take the data and apply the transformations in the order they are defined
            
            num_pipeline= Pipeline( # this pipeline does two things, first it will impute the missing values in the numerical columns with median and then it 
                # will scale the data using standard scaler
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            
            cat_pipeline=Pipeline( # this pipeline does two things, first it will impute the missing values in the categorical columns with most frequent value and then it
            # then it will one hot encode the data and then scale the data using standard scaler

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )
            
            
            logging.info(f"Categorical columns: {categorical_columns}") # 
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer( # we combine the two pipelines into one using column transformer, this will apply the numerical pipeline to the 
                # numerical columns and categorical pipeline to the categorical columns
                # column transformer is used to apply different transformations to different columns of the dataframe in a single step
                [
                ("num_pipeline",num_pipeline,numerical_columns), # transformer takes the name of the transformer, the transformer object and the columns to
                # which it should be applied
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )
            return preprocessor # return the preprocessor object which will be used to transform the data
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() # call the function and save the object in preprocessing_obj variable
            
            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            save_object( # create a pickle file of the preprocessor object using the save_object function defined in utils.py
                # it takes the file path and the object to be saved as parameters
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj # obj here refers to the preprocessor object which is created using the get_data_transformer_object function
                # that has all the transformations defined in it , saving it to a pickle file will allow us to use it later at the time of prediction
            )
        
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[ # this will combine the input features and target features into a single array after applying the transformations
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return ( # return the train and test arrays which will be used to train the model
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path, # also return the path of the preprocessor object file which will be used later to load the object
                # at the time of prediction
            )
        except Exception as e:
            raise CustomException(e,sys)        
        
            
