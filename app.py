import uvicorn as uvicorn
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Crime import Crime
import pickle
import pandas as pd
import numpy as np

app = FastAPI()
input = open("model.pkl", "rb")
model = pickle.load(input)
model =model(probability=True)

@app.get('/')
def index():
    return {"message": "hello"}


@app.post('/predict')
def predict_rate(data:Crime):
    data = data.dict()
    print(data)
    year = data['year']
    month = data['month']
    area1 = data['area1']
    area2 = data['area2']
    crimeType = data['crimeType']
    pre = model.predict([[year,month,area1,area2,crimeType]])
    # if  pre[0]>0.5:
    return {
            'prediction': str(pre),
            'prediction_prob': str()
        }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
