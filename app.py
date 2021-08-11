import uvicorn as uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import numpy  as np


import new_model
from Crime import Crime, Csv_Data
import pickle
from Crime import Crime_Wo_Districts

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
input = open("model.pkl", "rb")
model = pickle.load(input)

points = [
    {
        "name": 'south',
        "lat": 24.8605,
        "lng": 67.0261
    },
    {
        # // east
        "name": 'east',
        "lat": 24.8844,
        "lng": 67.1443
    },
    {
        # // west
        "name": 'west',
        "lat": 24.8829,
        "lng": 66.9748
    },
    {
        # // central
        "name": 'central',
        "lat": 24.9313,
        "lng": 67.0374
    },
    {
        # // malir
        "name": 'malir',
        "lat": 25.0960,
        "lng": 67.1871
    }
]


@app.get('/')
def index():
    return {"message": "hello"}


@app.get('/{name}')
def get_name(name: str):
    return {"message": "hello " + name}


@app.post('/predict')
def predict_rate(data: Crime):
    data = data.dict()
    print(data)
    year = data['year']
    month = data['month']
    area1 = data['area1']
    area2 = data['area2']
    crimeType = data['crimeType']
    pre = model.predict([[year, month, area1, area2, crimeType]])
    return {
        'year': str(year),
        'month': str(month),
        'area1': str(area1),
        'area2': str(area2),
        'crimeType': str(crimeType),
        'prediction': str(pre),
    }


@app.post('/predicts')
def predict_rate_of_different_districts(data: Crime_Wo_Districts):
    data = data.dict()
    pres = []
    year = data['year']
    month = data['month']
    crimeType = data['crimeType']
    for i in range(0, 5):
        area1 = points[i].get("lat")
        area2 = points[i].get("lng")
        print(area1)
        pres.append(model.predict([[year, month, area1, area2, crimeType]]))
        print(np.asfarray(model.predict([[year, month, area1, area2, crimeType]]), float))

    return {
        'year': str(year),
        'month': str(month),
        'crimeType': str(crimeType),
        'prediction': str(pres),
        'south_prediction': str(pres[0]),
        'east_prediction': str(pres[1]),
        'west_prediction': str(pres[2]),
        'central_prediction': str(pres[3]),
        'malir_prediction': str(pres[4]),
    }


@app.post('/csvupload')
def csvupload_train(data: Csv_Data):
    data = data
    data.data.pop(0)
    acc = new_model.new_model(data.data)
    print('accuracy of ur model ', acc)
    if acc>0.1:
        return {
            "accuracy": str(acc),
            "upload": "successful",
        }
    else:
        return {
            "accuracy": str(acc),
            "upload": "failed",
        }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
