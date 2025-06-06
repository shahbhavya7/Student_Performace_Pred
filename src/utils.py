import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param): 
    try:
        report = {}

        for i in range(len(list(models))): # loop through models fitting y_train and y_test and getting the score and saving it in the report dict
            model = list(models.values())[i] #  get the model from the models dict
            para=param[list(models.keys())[i]]  # get the params for the model from the params dict

            gs = GridSearchCV(model,para,cv=3) # fit the model with the params and cv=3 for cross validation , gridsearchCV is used to find the best params for the model
            # it fits tries every combination of the params and returns the best params for the model and cross validation is used to avoid overfitting
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_) # gets the best params for the model from the gridsearchCV and applies them to the model
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score # report[list(models.keys())[i]] retrieves the name of the model from model list and then assigns 
            # the score to the name in the report dict

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path): # loads the object from the file path i.e loads the model from the file path and returns it
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
