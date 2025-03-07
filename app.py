from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from train_model import get_trained_model, ThreeLayerMLP

app = FastAPI()

model: ThreeLayerMLP = get_trained_model()

class DataRequest(BaseModel):
    data: List[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/classify_data")
async def classify_data(request: DataRequest):
    if not request.data:
        return {"error": "No data provided."}
    
    classification = model.predict(request.data)
    
    return {"classification": classification}
