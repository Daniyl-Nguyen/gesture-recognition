from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class DataRequest(BaseModel):
    data: List[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/classify_data")
async def classify_data(request: DataRequest):
    if not request.data:
        return {"error": "No data provided."}
    
    average = sum(request.data) / len(request.data)
    classification = average
    return {"classification": classification}
