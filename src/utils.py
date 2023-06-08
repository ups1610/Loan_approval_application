import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def handle_missing(X_train,X_test):
    try:        
       imp = SimpleImputer(strategy='mean')
       imp_train = imp.fit(X_train)
       X_train = imp_train.transform(X_train)
       X_test_imp = imp_train.transform(X_test)
        
       return X_train,X_test_imp 

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)