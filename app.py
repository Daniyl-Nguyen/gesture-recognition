from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from train_model import get_trained_model, ThreeLayerMLP
from gesture_dataloader import GestureDataset  # Import your dataset class

app = FastAPI()

# Load trained model
model: ThreeLayerMLP = get_trained_model()

# Load dataset to get gesture mappings
dataset = GestureDataset("dataset")  # Ensure this points to the correct dataset path
idx_to_gesture = {idx: gesture for gesture, idx in dataset.gesture_to_idx.items()}  # Reverse mapping

class DataRequest(BaseModel):
    data: List[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/classify_data")
async def classify_data(request: DataRequest):
    if not request.data:
        return {"error": "No data provided."}
    
    classification_idx, confidence = model.predict(request.data)  # Get prediction & confidence

    if confidence < 0.7:  # Reject if confidence is too low
        return {"classification": "Unknown Gesture", "confidence": confidence}

    gesture_name = idx_to_gesture.get(classification_idx, "Unknown Gesture")

    return {"classification": gesture_name, "confidence": confidence}
