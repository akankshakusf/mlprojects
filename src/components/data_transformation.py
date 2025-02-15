#import packages
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import internal logging, error handling, and object persistence
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

#data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler



# Dataclass to configure paths for data preprocessing objects
@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

# Class for handling all data transformation tasks
class DataTransformation:
    def __init__(self):
        # Initialize configuration settings from DataTransformationConfig
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation for variety of column data types
        '''    
        try:
            # Define columns to be processed by numerical and categorical pipelines
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            # Pipeline for numerical data with imputation and scaling
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())                    
                ]
            )

            # Pipeline for categorical data with imputation, encoding, and scaling
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) 
                ]
            )
            # Log configured columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            # Combine pipelines into a single preprocessor using ColumnTransformer
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        # Handle any exceptions that occur during the setup of the transformer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Load data from specified paths
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # Log the completion of data loading
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Obtain a preprocessor object configured for the data
            preprocessing_obj=self.get_data_transformer_obj()
            target_column_name='math_score'
            numerical_columns=["reading_score", "writing_score"]

            # Separate features and target variable for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # Separate features and target variable for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Apply transformations to training and testing data
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            # Combine features and targets into arrays for training and testing
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            # Log and save the preprocessing object
            logging.info("Saved preprocessing object")  

            #this comes from utils.py
            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            ) 

            # Return processed data arrays and path to the saved preprocessing object
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path                   
                   )

         # Handle any exceptions that occur during data processing
        except Exception as e:
            raise CustomException(e,sys)