import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            model=load_object(model_path)
            print(features)
            print(features.info())
            data_scaled = pd.get_dummies(features,drop_first=True)
            print(data_scaled)
            if features['Gender'][0]=='Male':
                data_scaled['Gender_Male']=[1]
            else:
                data_scaled['Gender_Male']=[0]

            if features['Married'][0]=="Yes":
                data_scaled['Married_Yes']=[1]
            else:
                data_scaled['Married_Yes']=[0]

            if features['Dependents'][0]=="0":
                data_scaled['Dependents_1']=[0]
                data_scaled['Dependents_2']=[0]
                data_scaled['Dependents_3+']=[0]
            elif features['Dependents'][0]=="1": 
                data_scaled['Dependents_1']=[1]
                data_scaled['Dependents_2']=[0]
                data_scaled['Dependents_3+']=[0]
            elif features['Dependents'][0]=="2": 
                data_scaled['Dependents_1']=[0]
                data_scaled['Dependents_2']=[1]
                data_scaled['Dependents_3+']=[0]
            else:
                data_scaled['Dependents_1']=[0]
                data_scaled['Dependents_2']=[0]
                data_scaled['Dependents_3+']=[1]   

            if features['Education'][0]=="Graduate":
                data_scaled['Education_Not Graduate']=[0]
            else:
                data_scaled['Education_Not Graduate']=[1]   

            if features['Self_Employed'][0]=="Yes":
                data_scaled['Self_Employed_Yes']=[1]
            else:
                data_scaled['Self_Employed_Yes']=[0]

            if features['Property_Area'][0]=="Urban":
                data_scaled['Property_Area_Semiurban']=[0]
                data_scaled['Property_Area_Urban']=[1]
            elif features['Property_Area'][0]=="Rural":
                data_scaled['Property_Area_Semiurban']=[0]
                data_scaled['Property_Area_Urban']=[0]
            else:
                data_scaled['Property_Area_Semiurban']=[1]
                data_scaled['Property_Area_Urban']=[0]                            
            
            print(data_scaled)
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
                 ApplicantIncome:int,
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