from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any
import os 
print("Running inside docker:", os.path.exists("/.dockerenv"))


app = FastAPI(title="lead-score")

with open('pipeline_v1.bin', 'rb') as f_in:
   pipeline = pickle.load(f_in)


def predict_single(record):
    result1 = pipeline.predict_proba(record)[0, 1]
    return float(result1)


@app.post("/predict")
def predict(record: Dict[str, Any]):
    prob = predict_single(record)

    return {
        "conversion_probability":  prob,
        "converted": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
