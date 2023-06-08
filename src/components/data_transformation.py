import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path_train=os.path.join('artifacts','preprocessed_train.csv')
    preprocessor_obj_file_path_test=os.path.join('artifacts','preprocessed_test.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self,train_df,test_df):
        try:
            logging.info('Data Transformation initiated')
            
            #### Encoding categrical Features: ##########
            train_df_encoded = pd.get_dummies(train_df,drop_first=True)
            test_df_encoded = pd.get_dummies(test_df,drop_first=True)

            logging.info('Data Transformation initiated')
            return train_df_encoded,test_df_encoded

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            ## drop un-necessary data
            train_df.drop(columns='Loan_ID',axis=1,inplace=True)
            test_df.drop(columns='Loan_ID',axis=1,inplace=True)

            train_df,test_df = self.get_data_transformation(train_df,test_df)

            target_column_name = 'Loan_Status_Y'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path_train,
                obj=train_df
            )
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path_test,
                obj=test_df
            )
            logging.info("Preprocessed csv's file saved")

            return (
                input_feature_train_df,
                target_feature_train_df,
                input_feature_test_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path_train,
                self.data_transformation_config.preprocessor_obj_file_path_test
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)