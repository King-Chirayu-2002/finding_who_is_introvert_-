import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def predict(self,features):
        try:
            model_path  = "artifacts/model.pkl" 
            preProcessingPath = "artifacts/preprocessing.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path=preProcessingPath)
            data_scaled = preprocessor.transform(features)
            print(data_scaled)
            preds = model.predict(data_scaled)
            if preds[0]>0.5:
                print('introvert')
            else:
                print('extrovert')
            print(preds[0])
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
    
class CustomData:
    def __init__(self,
                Time_spent_Alone : float,
                Stage_fear : object,
                Social_event_attendance: float,
                Going_outside:float,
                Drained_after_socializing : object,
                Friends_circle_size:float,
                Post_frequency:float,
                ):
        self.Time_spent_Alone = Time_spent_Alone
        self.Stage_fear = Stage_fear
        self.Social_event_attendance = Social_event_attendance
        self.Going_outside = Going_outside
        self.Drained_after_socializing = Drained_after_socializing
        self.Friends_circle_size = Friends_circle_size
        self.Post_frequency = Post_frequency

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Time_spent_Alone": [self.Time_spent_Alone],
                "Stage_fear": [self.Stage_fear],
                "Social_event_attendance": [self.Social_event_attendance],
                "Going_outside": [self.Going_outside],
                "Drained_after_socializing": [self.Drained_after_socializing],
                "Friends_circle_size": [self.Friends_circle_size],
                "Post_frequency": [self.Post_frequency],
                }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)