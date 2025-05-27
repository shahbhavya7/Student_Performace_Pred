from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__) # this is the name of the file

app=application # this is the name of the file i.e app.py

@app.route('/') # this is the route for the home page
def index():
    return render_template('index.html')  

@app.route('/predictdata',methods=['GET','POST']) # this is the route for the predict data page with get and post methods
def predict_datapoint():
    if request.method=='GET': # if the method is get then render the home page
        return render_template('home.html')
    else: # if the method is post then predict the data using the predict pipeline by passing the data from the form i.e post request
        data=CustomData( # this function is defined in the custom_data.py file and it returns a dataframe with the data from the form
            gender=request.form.get('gender'), # getting gender from the form using the name attribute, request.form is used to get the data from the form
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')), # getting data and converting it to float
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame() # captures the returned dataframe from the custom_data function
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline() # creating an object of the predict pipeline class
        print("Mid Prediction") # this is just for debugging
        results=predict_pipeline.predict(pred_df) # calling the predict function of the predict pipeline class and passing the dataframe as an argument
        print("after Prediction")
        return render_template('home.html',results=results[0]) # rendering the home page with the results , results[0] is the first element of the list returned by the predict function
        # first element of the list is the predicted value
        
if __name__=="__main__":
    app.run(host="0.0.0.0")        


    