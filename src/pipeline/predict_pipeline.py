import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features): # predict method takes the features as input and returns the predicted values
        try:
            model_path=os.path.join("artifacts","model.pkl") # storing the model path from the artifacts folder in variable 
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path) # loading the model from the model path by calling the load_object and passing the model_path as argument
            preprocessor=load_object(file_path=preprocessor_path) # same as above
            print("After Loading")
            data_scaled=preprocessor.transform(features) # preprocessing the features by calling the transform method of the preprocessor object and passing the features as argument
            preds=model.predict(data_scaled) # predicting the values by calling the predict method of the model object and passing the preprocessed features as argument
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self, 
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int): # this is the constructor for the class and it takes the features as input and assigns them to the class variables

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self): # this method returns the features as a dataframe
        try:
            custom_data_input_dict = { # this is the dictionary that collects the features from the constructor and stores them in a dictionary
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict) # this returns the dictionary as a dataframe

        except Exception as e:
            raise CustomException(e, sys)

