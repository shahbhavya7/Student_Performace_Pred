import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl") # this is the path where the trained model will be saved, in artifacts folder under the name model.pkl

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = { # define the models to be trained
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
        
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            # it returns a dictionary of model names and their scores this is stored in model_report
            
            ## Retrieving best model score from dict by sorting the values of the dict in descending order
            best_model_score = max(sorted(model_report.values()))

            ## Correspondingly retrieving the best model name from the dict

            best_model_name = list(model_report.keys())[ # list is used to convert the keys of the dict to a list and then the list index is used to get the name of the model
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            # model_report values are conv to list first then the index func searches which index has the best_model_score value and returns the index
            # then again the model_report keys are conv to list and using the index retrieved from the previous step, we retirve name of the model
            # that has the best_model_score
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object( # save the best model to the path specified in the config using the save_object function which uses pickle saving
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test) # predict the test data using the best model

            r2_square = r2_score(y_test, predicted) # calculate the r2 score of the predicted data
            return r2_square # return the r2 score of the best model
        
        except Exception as e:
            raise CustomException(e,sys)
            