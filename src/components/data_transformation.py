#import packages
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data trasformation for varity of column data types
        '''    
        try:
            numerical_columns=["reading_score", "writing_score"]
            categorical_columns=["gender",
                                  "race_ethnicity", 
                                  "parental_level_of_education", 
                                  "lunch", 
                                  "test_preparation_course"]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standard",StandardScaler())                    
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("standard",StandardScaler(with_mean=False)) 
                ]
            )
            logging.info("Categorical columns :{categorical_columns}")
            logging.info("Numerical columns {numerical_columns}")


            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()
            target_column_name='math_score'
            numerical_columns=["reading_score", "writing_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")  

            #this comes from utils.py
            save_object(
                filepath=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            ) 

            return(train_arr,
                   test_arr,
                   self.data_tranformation_config.preprocessor_obj_file_path                   
                   )

        except Exception as e:
            raise CustomException(e,sys)