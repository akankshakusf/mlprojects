# Importing packages
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importing the PredictPipeline script
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Creating a Flask app name
application = Flask(__name__)

app = application

# Defining a route for the home page 
@app.route('/')
def index():
    return render_template('index.html')

# Defining a route to handle predictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # Handling GET request
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Handling POST request: it captures form data, creates a data instance and processes it.
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),  
            writing_score=float(request.form.get('reading_score')) 
        )
        # Converting to DataFrame.
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  

        # Creating a prediction pipeline and making predictions.
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # Returning the prediction results 
        return render_template('home.html', results=results[0])

# Conditional statement that runs the application on host 0.0.0.0 
if __name__ == "__main__":
    app.run(host="0.0.0.0")
