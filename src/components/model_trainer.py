#import packages
import os
import sys
from dataclasses import dataclass

# Import internal logging, error handling, and object persistence
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

#import modelling Algorithm
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#performance metric
from sklearn.metrics import r2_score

# Dataclass to store the configuration for the model trainer
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

# Define a class for training models
class ModelTrainer:
    def __init__(self):
        # Initialize configuration using the ModelTrainerConfig dataclass
        self.model_trainer_config=ModelTrainerConfig()

    # Function to initiate the model training process
    def initiate_model_trainer(self,train_array,test_array):
        try:
             # Log the start of data separation for training and testing
            logging.info("Spliting train and test input data")

            # Extract features and target variable from train and test arrays
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Dictionary to store machine learning models with their initial configuration
            models = {                
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),            
                'XGBRegressor':XGBRegressor(), 
                'CatBoosting Regressor':CatBoostRegressor(verbose=False),               
                'AdaBoost Regressor':AdaBoostRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            
            model_report:dict=evaluate_models(X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                             models=models,
                                             param=params)            

            
            #to get best model score from the dict
            best_model_score=max(sorted(model_report.values()))

            #to get the best model key from the dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]



            # Check if the best model's performance is satisfactory
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            # Log the identification of the best model
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model to a specified path
            save_object(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Predict using the best model and calculate the R-squared value
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
