import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(model_path)
            print(features)
            data_scaled = pd.get_dummies(features,drop_first=True)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Gender:str,
                 Married:str,
                 Dependents:str,
                 Education:str,
                 Self_Employed:str,
                 ApplicantIncome:float,
                 CoapplicantIncome:float,
                 LoanAmount:float,
                 Loan_Amount_Term:float,
                 Credit_History:float,
                 Property_Area:str
                 ):
        
        self.Gender=Gender
        self.Married=Married
        self.Dependents=Dependents
        self.Education=Education
        self.Self_Employed=Self_Employed
        self.Applicantincome=ApplicantIncome
        self.Co_app_income = CoapplicantIncome
        self.LoanAmount=LoanAmount
        self.Loan_Amount_Term=Loan_Amount_Term
        self.Credit_history=Credit_History
        self.property_area=Property_Area

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender':[self.Gender],
                'Married':[self.Married],
                'Dependents':[self.Dependents],
                'Education':[self.Education],
                'Self_Employed':[self.Self_Employed],
                'ApplicantIncome':[self.Applicantincome],
                'CoapplicantIncome':[self.Co_app_income],
                'LoanAmount':[self.LoanAmount],
                'Loan_Amount_Term':[self.Loan_Amount_Term],
                'Credit_History':[self.Credit_history],
                'Property_Area':[self.property_area]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)