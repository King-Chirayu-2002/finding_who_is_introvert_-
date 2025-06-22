import sys
import os
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


from src.utils import save_object

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# class DropRedundantCategorical(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.9):
#         self.threshold = threshold
#         self.to_drop = []

#     def fit(self, X, y=None):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)

#         cols = X.columns
#         n = len(cols)
#         to_drop = set()

#         for i in range(n):
#             for j in range(i + 1, n):
#                 v = cramers_v(X[cols[i]], X[cols[j]])
#                 if v > self.threshold:
#                     to_drop.add(cols[j])  # drop the later feature

#         self.to_drop = list(to_drop)
#         return self

#     def transform(self, X):
#         return X.drop(columns=self.to_drop)

@dataclass
class DataTransformationConfig:
    preprocessing_file_path:str =  os.path.join('artifacts','preprocessing.pkl')


class DataTransformation():
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()
    
    def getDataTranformer(self):
        try:
            num_featues = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']
            cat_features = ['Stage_fear','Drained_after_socializing']
            num_pipeline = Pipeline(
                steps= [
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline= Pipeline(
                steps=[
                   ('encoding', OneHotEncoder(drop='first')) 
                ]
            )
            preprocessor  = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,num_featues),
                    ('categorical_pipeline',cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            CustomException(e,sys)
    
    def initiateDataTransformation(self,train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('reading of data completed') 
            
            preprocessing_obj =  self.getDataTranformer()
            logging.info('we got the data tranformer pipeline')
            
            target_feature = 'Personality'
            input_feature_train_df = train_df.drop(target_feature,axis=1)
            target_feature_train_df  = train_df[target_feature]
            input_feature_test_df = test_df.drop(target_feature,axis=1)
            target_feature_test_df  = test_df[target_feature]
            
            logging.info('applying preprocessing on test and training input featuers')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
  
            # Encode target labels
            label_encoder = LabelEncoder()
            target_feature_train_encoded = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = label_encoder.transform(target_feature_test_df)

        # Combine features with encoded target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]


            logging.info('Data now is preprocessed')
            save_object(file_path = self.dataTransformationConfig.preprocessing_file_path,obj=preprocessing_obj)
            return (train_arr,test_arr, self.dataTransformationConfig.preprocessing_file_path)

        except Exception as e:
            raise CustomException(e,sys)
        
