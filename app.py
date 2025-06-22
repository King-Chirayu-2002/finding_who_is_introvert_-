from flask import request,render_template,Flask,jsonify
import numpy as np
import pandas as pd

from src.pipeline.predictPipeline import PredictionPipeline,CustomData

application = Flask(__name__)

app = application

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        data_json = request.get_json()

        # Extract data from JSON
        data = CustomData(
    Time_spent_Alone=float(data_json.get('Time_spent_Alone')),
    Stage_fear=data_json.get('Stage_fear'),
    Social_event_attendance=float(data_json.get('Social_event_attendance')),
    Going_outside=float(data_json.get('Going_outside')),
    Drained_after_socializing=data_json.get('Drained_after_socializing'),
    Friends_circle_size=float(data_json.get('Friends_circle_size')),
    Post_frequency=float(data_json.get('Post_frequency')),
)


        pred_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)

        # Send result back to frontend as JSON
        return jsonify({"result": float(results[0])})



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
    
    
    
    