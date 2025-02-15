#import packages
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Define a dataclass decorator to hold the configuration for data ingestion
@dataclass
class DataIngestionConfig:
    # Paths are constructed using os.path.join for train,test and raw data
    train_data_path: str = os.path.join('artifacts', "train.csv")  
    test_data_path: str = os.path.join('artifacts', "test.csv")    
    raw_data_path: str = os.path.join('artifacts', "data.csv")     

# Class to handle data ingestion processes
class DataIngestion:
    def __init__(self):
        # Initialize the configuration when a DataIngestion object is created
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log entry into the data ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            # Read data from a CSV file into a DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")


            # Ensure the directory for saving data exists; create if it does not
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # Make Log entry
            logging.info("Train test split is initiated")


            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            # Save the training data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the testing data
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            # Make Log entry
            logging.info("Ingestion of the data is completed")

            # Return the paths to the saved train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Handle any exceptions that occur during the ingestion process
            raise CustomException(e, sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()

