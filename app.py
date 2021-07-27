import uvicorn as uvicorn
from fastapi import FastAPI
from Crime import Crime
import pickle

app = FastAPI()
input = open("model.pkl", "rb")
model = pickle.load(input)

@app.get('/')
def index():
    return {"message": "hello"}

@app.get('/{name}')
def get_name(name:str):
    return {"message": "hello "+name}


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
    return {
            'year': str(year),
            'month': str(month),
            'area1': str(area1),
            'area2': str(area2),
            'crimeType': str(crimeType),
            'prediction': str(pre),
        }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
