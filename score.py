import json
import numpy as np
import os
from azureml.core import Model
import joblib
import sklearn
def init():
    global model
    model_path = Model.get_model_path('glass-prediction')
    print(f"Model path: {model_path}")
    model = joblib.load(model_path)
def run(data):
    try:
        data = np.array(json.loads(data))
        print(f"Data received: {data}")
        result = model.predict(data)
        print(f"result: {result}")
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
