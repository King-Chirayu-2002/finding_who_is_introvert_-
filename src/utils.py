import os
import sys
import dill

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir_path  = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for name, model in models.items():  
            params_grid = params.get(name,{})
            gs = GridSearchCV(model,params_grid,cv=5)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_pred)
        
            # Make predictions
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test,y_test_pred)
    
            report[name] = test_model_score  
        return report
            
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            dill.load(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)