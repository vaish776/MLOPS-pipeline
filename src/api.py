import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Model API")

# Load model when API starts (run train.py first)
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)


class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Iris model is ready. Use POST /predict."}


@app.post("/predict")
def predict(data: InputData):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    pred = model.predict(features)[0]
    return {"prediction": int(pred)}
