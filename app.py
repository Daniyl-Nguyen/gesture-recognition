from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
import numpy as np # Good for numerical operations if needed later

# --- Assuming these functions/variables are correctly set up ---
from train_model import get_trained_models # Your function to load models
from gesture_dataloader import GestureDataset # Your dataset class

# --- Configuration ---
DATASET_PATH = "dataset"
CONFIDENCE_THRESHOLD = 0.7
FLOATS_PER_HAND = 147 # 21 joints * 7 floats (posXYZ, rotXYZW)
EXPECTED_DATA_LENGTH = FLOATS_PER_HAND * 2

# --- FastAPI App Initialization ---
app = FastAPI()

print("Loading models...")
try:
    model_left, model_right = get_trained_models()
    print("Models loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load models: {e}")
    # Exit or prevent app startup if models are essential
    # For now, we'll let it run but predictions will fail
    model_left, model_right = None, None # Set to None to handle errors later


print("Loading dataset for gesture mapping...")
try:
    dataset = GestureDataset(DATASET_PATH) # Load once for mapping
    idx_to_gesture = {idx: gesture for gesture, idx in dataset.gesture_to_idx.items()}
    print("Gesture mapping loaded.")
    print(f"Mappings: {idx_to_gesture}")
except Exception as e:
    print(f"ERROR: Could not load dataset/gesture mapping: {e}")
    idx_to_gesture = {} # Use empty mapping if loading fails


# --- Pydantic Model for Flat Array Request ---
class FlatCombinedDataRequest(BaseModel):
    data: List[float]

    # Optional but recommended: Add a validator for the data length
    @validator('data')
    def check_data_length(cls, v):
        if len(v) != EXPECTED_DATA_LENGTH:
            raise ValueError(f"Data list must contain exactly {EXPECTED_DATA_LENGTH} elements, received {len(v)}")
        return v

# --- API Endpoints ---

@app.get("/")
async def root():
    """ Root endpoint for basic connectivity check. """
    return {"message": "Gesture Classification API Running"}

@app.post("/classify_data")
async def classify_data(request: FlatCombinedDataRequest):
    """
    Receives combined hand data as a single flat array, splits it,
    performs classification for each hand, and returns results.
    """
    results = {}
    full_data = request.data

    # --- Split the data ---
    try:
        left_data = full_data[:FLOATS_PER_HAND]    # First 147 elements
        right_data = full_data[FLOATS_PER_HAND:] # Elements from index 147 to the end
    except Exception as e:
         # Should not happen if validator works, but catch potential slicing issues
         print(f"Error splitting data: {e}")
         raise HTTPException(status_code=500, detail="Internal server error during data splitting.")


    # --- Process left hand ---
    if model_left is not None and idx_to_gesture:
        if left_data:
            try:
                classification_idx_left, confidence_left = model_left.predict(left_data)
                gesture_name_left = idx_to_gesture.get(classification_idx_left, "Unknown Index") # Handle index not in map
                # Apply confidence threshold
                if confidence_left < CONFIDENCE_THRESHOLD:
                    gesture_name_left = "Unknown Gesture" # Overwrite if below threshold
                results["left_hand"] = {"classification": gesture_name_left, "confidence": float(confidence_left)} # Ensure confidence is float
            except Exception as e:
                print(f"Error predicting left hand: {e}")
                results["left_hand"] = {"error": "Prediction failed"}
        else:
            results["left_hand"] = {"error": "No data provided/extracted for left hand"} # Should be caught by validator
    else:
         results["left_hand"] = {"error": "Left hand model or mapping not loaded"}


    # --- Process right hand ---
    if model_right is not None and idx_to_gesture:
        if right_data:
            try:
                classification_idx_right, confidence_right = model_right.predict(right_data)
                gesture_name_right = idx_to_gesture.get(classification_idx_right, "Unknown Index") # Handle index not in map
                # Apply confidence threshold
                if confidence_right < CONFIDENCE_THRESHOLD:
                    gesture_name_right = "Unknown Gesture" # Overwrite if below threshold
                results["right_hand"] = {"classification": gesture_name_right, "confidence": float(confidence_right)} # Ensure confidence is float
            except Exception as e:
                print(f"Error predicting right hand: {e}")
                results["right_hand"] = {"error": "Prediction failed"}
        else:
            results["right_hand"] = {"error": "No data provided/extracted for right hand"} # Should be caught by validator
    else:
         results["right_hand"] = {"error": "Right hand model or mapping not loaded"}

    return results

# --- Optional: Add main execution block for running with uvicorn ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting server with uvicorn...")
#     uvicorn.run(app, host="127.0.0.1", port=8000)