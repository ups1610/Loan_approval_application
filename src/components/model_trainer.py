# Basic Import
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
from src.logger import logging

from src.utils import save_object
from src.utils import handle_missing

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_feature,train_target_feature,test_feature,test_target_feature):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, X_test,y_train , y_test = (
                train_feature,
                test_feature,
                train_target_feature,
                test_target_feature
            )
            X_train,X_test = handle_missing(train_feature,test_feature)
         
            best_model = LogisticRegression(solver='liblinear')
            
            best_model.fit(X_train,y_train)
            y_pred = best_model.predict(X_test)
            print((100*"="))
            print("Test Accuracy: ",accuracy_score(y_test,y_pred))
            print("Test F1 Score: ",f1_score(y_test,y_pred))
            print(100*"=")
            logging.info("Confusion Matrix on Test Data")
            logging.info(f"{pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)}")

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)