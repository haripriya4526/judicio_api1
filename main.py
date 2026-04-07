from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("judicio_outcome_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)

    return {"prediction": prediction.tolist()}

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)