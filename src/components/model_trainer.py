import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier 

@dataclass
class ModelTrainerConfig:
    mdoe_file_path  = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_file_paths  = ModelTrainerConfig()
    
    def initiateModelTrainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info('train test data has been loaded for starting model training')
            X_train,y_train,X_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            models = {
    'Logistic Regression'      : LogisticRegression(),
    'XGBoost Classifier'       : XGBClassifier(eval_metric='logloss'),
    'CatBoost Classifier'      : CatBoostClassifier(verbose=0),
    'Random Forest Classifier' : RandomForestClassifier(),
    'K-Nearest Neighbors'      : KNeighborsClassifier(),
    'Support Vector Classifier': SVC()
}
            
            param_grid = {
   'Logistic Regression': {
    'C': [0.01, 0.1, 1.0, 10],  # already included: C is inverse of regularization
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']  # l2 regularization
}
,
    'XGBoost Classifier': {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],          # Lower learning rate helps generalization
    'subsample': [0.8, 1],                 # Row sampling
    'colsample_bytree': [0.8, 1],          # Feature sampling
    'reg_lambda': [1, 10],                 # L2 regularization
    'reg_alpha': [0, 1]                    # L1 regularization
},
    'CatBoost Classifier': {
    'iterations': [100, 200],
    'depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5]  # Regularization strength
},
    'Random Forest Classifier': {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
},
    'K-Nearest Neighbors': {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan or Euclidean distance
},
    'Support Vector Classifier': {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
}
            logging.info('Now, Model training has been started ')
            model_report : dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,params = param_grid)
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            else:
                logging.info('best model found on both data sets')
            
            save_object(
                file_path=self.model_file_paths.mdoe_file_path,
                obj= models[best_model_name]
            )

        except Exception as e:
            raise CustomException(e,sys)

