import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the test dataset
TEST_DATA_PATH = "c:\\Users\\daniy\\gesture-recognition\\test_dataset"  # Change this to your new dataset folder

# Define gesture types and model path
GESTURE_TYPES = ["fist", "fingers_spread", "wave_in", "wave_out", "no_gesture", 
                 "thumb_to_pinky", "thumbs_up", "index_point", "inward_pan", "outward_pan"]
MODEL_PATH = "c:\\Users\\daniy\\gesture-recognition\\model.h5"  # Update this path if your model is stored elsewhere

def load_model():
    """Load the trained model"""
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def process_excel_file(file_path):
    """Process a single Excel file and extract features"""
    try:
        df = pd.read_excel(file_path)
        
        # Extract only the EMG data columns (assuming the same structure as training data)
        emg_columns = [col for col in df.columns if col.startswith('emg')]
        
        if not emg_columns:
            print(f"Warning: No EMG columns found in {file_path}")
            return None
            
        # Extract features (using the same approach as your training code)
        # This may need adjustment based on how your model expects input
        X = df[emg_columns].values
        
        # Normalize data (same as training)
        X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-10)
        
        return X
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def test_model():
    """Test the model on the new dataset"""
    model = load_model()
    
    results = []
    
    # Walk through the test dataset directory
    for person_name in os.listdir(TEST_DATA_PATH):
        person_path = os.path.join(TEST_DATA_PATH, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        # Loop through gesture types
        for gesture in os.listdir(person_path):
            gesture_path = os.path.join(person_path, gesture)
            
            if not os.path.isdir(gesture_path) or gesture not in GESTURE_TYPES:
                continue
                
            # Process left and right hands
            for hand in ["left", "right"]:
                hand_path = os.path.join(gesture_path, hand)
                
                if not os.path.isdir(hand_path):
                    continue
                    
                # Process each Excel file in the directory
                for file_name in os.listdir(hand_path):
                    if file_name.endswith(".xlsx"):
                        file_path = os.path.join(hand_path, file_name)
                        
                        # Process the Excel file
                        X = process_excel_file(file_path)
                        
                        if X is not None and len(X) > 0:
                            # Make prediction
                            predictions = model.predict(X)
                            predicted_class = np.argmax(predictions, axis=1)
                            
                            # Get the most common prediction (majority vote)
                            most_common_prediction = np.bincount(predicted_class).argmax()
                            
                            # Map the prediction index back to the gesture name
                            predicted_gesture = GESTURE_TYPES[most_common_prediction]
                            
                            # Calculate confidence
                            confidence = np.mean(predictions[:, most_common_prediction])
                            
                            # Record the result
                            result = {
                                "person": person_name,
                                "actual_gesture": gesture,
                                "hand": hand,
                                "predicted_gesture": predicted_gesture,
                                "confidence": confidence,
                                "correct": predicted_gesture == gesture,
                                "file": file_name
                            }
                            
                            results.append(result)
                            
                            print(f"Tested: {person_name} - {gesture} - {hand} - Prediction: {predicted_gesture} - Confidence: {confidence:.2f}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df

def analyze_results(results_df):
    """Analyze and display the test results"""
    if len(results_df) == 0:
        print("No results to analyze")
        return
        
    # Overall accuracy
    overall_accuracy = results_df["correct"].mean()
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    # Accuracy by person
    person_accuracy = results_df.groupby("person")["correct"].mean().sort_values(ascending=False)
    print("\nAccuracy by Person:")
    print(person_accuracy)
    
    # Accuracy by gesture
    gesture_accuracy = results_df.groupby("actual_gesture")["correct"].mean().sort_values(ascending=False)
    print("\nAccuracy by Gesture:")
    print(gesture_accuracy)
    
    # Accuracy by hand
    hand_accuracy = results_df.groupby("hand")["correct"].mean()
    print("\nAccuracy by Hand:")
    print(hand_accuracy)
    
    # Confusion Matrix
    y_true = results_df["actual_gesture"]
    y_pred = results_df["predicted_gesture"]
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=GESTURE_TYPES)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURE_TYPES, yticklabels=GESTURE_TYPES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=GESTURE_TYPES)
    print("\nClassification Report:")
    print(report)
    
    # Save detailed results to CSV
    # results_df.to_csv("test_results.csv", index=False)
    # print("\nDetailed results saved to 'test_results.csv'")

if __name__ == "__main__":
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test dataset directory not found at {TEST_DATA_PATH}")
        print("Please create the directory with the correct structure or update the TEST_DATA_PATH variable.")
    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update the MODEL_PATH variable to point to your trained model.")
    else:
        print(f"Testing model on data in {TEST_DATA_PATH}")
        results = test_model()
        analyze_results(results)
