import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from train_model import get_trained_models, ThreeLayerMLP
from gesture_dataloader import GestureDataset

# Define the Participants dataset path
PARTICIPANTS_DATA_PATH = "c:\\Users\\daniy\\gesture-recognition\\Participants"
DATASET_PATH = "dataset"

# Define constants
FLOATS_PER_HAND = 147  # 21 joints * 7 floats (posXYZ, rotXYZW)

# Define gesture labels (make sure these match what the model was trained on)
dataset = GestureDataset(DATASET_PATH)
idx_to_gesture = {idx: gesture for gesture, idx in dataset.gesture_to_idx.items()}
GESTURE_LABELS = [idx_to_gesture[i] for i in range(len(idx_to_gesture))]

def process_csv_file(file_path):
    """
    Process a single CSV file and extract features as a flat list.
    """
    try:
        df = pd.read_csv(file_path)
        # Extract only the Position and Rotation columns
        feature_columns = ["PositionX", "PositionY", "PositionZ", "RotationX", "RotationY", "RotationZ", "RotationW"]
        if not all(col in df.columns for col in feature_columns):
            print(f"Warning: Missing required columns in {file_path}")
            return None
        # Extract features and flatten into a single list
        X = df[feature_columns].values.flatten().tolist()
        # Ensure the data length matches the expected length for one hand
        if len(X) != FLOATS_PER_HAND:
            print(f"Warning: Data length mismatch in {file_path}. Expected {FLOATS_PER_HAND}, got {len(X)}")
            return None
        return X
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def classify_hand_data(hand_data, model, confidence_threshold=0.5):
    """
    Classify hand gesture data using the loaded model
    """
    # Convert data to tensor for model prediction
    data_tensor = torch.tensor(hand_data, dtype=torch.float32)
    
    # Use the model to get prediction
    with torch.no_grad():
        output = model(data_tensor.unsqueeze(0))
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Convert probabilities to numpy for easier handling
    probs_np = probabilities.numpy()
    
    # Get the index of the highest probability
    predicted_index = np.argmax(probs_np)
    confidence = probs_np[predicted_index]
    
    # Apply confidence threshold
    if confidence >= confidence_threshold:
        classification = GESTURE_LABELS[predicted_index]
    else:
        classification = "Unknown Gesture"
    
    return {
        "classification": classification,
        "confidence": float(confidence),
        "all_probabilities": {GESTURE_LABELS[i]: float(probs_np[i]) for i in range(len(GESTURE_LABELS))}
    }

def evaluate_participants_with_threshold(left_hand_model, right_hand_model, confidence_threshold=0.5):
    """
    Evaluate the data within the Participants directory using the specified confidence threshold.
    """
    results = []

    # Walk through the Participants dataset directory
    for participant_name in os.listdir(PARTICIPANTS_DATA_PATH):
        participant_path = os.path.join(PARTICIPANTS_DATA_PATH, participant_name)
        if not os.path.isdir(participant_path):
            continue

        # Loop through gesture folders
        for gesture_folder in os.listdir(participant_path):
            gesture_path = os.path.join(participant_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            # Extract gesture name from the folder name
            try:
                _, gesture = gesture_folder.split("-", 1)
                if gesture == "Unknown":
                    gesture = "Unknown Gesture"
            except ValueError:
                print(f"Skipping folder {gesture_folder}: Unable to extract gesture name.")
                continue

            # Process Left and Right hands
            for hand in ["Left", "Right"]:
                hand_path = os.path.join(gesture_path, hand)
                if not os.path.isdir(hand_path):
                    continue

                # Process each CSV file in the directory
                for file_name in os.listdir(hand_path):
                    if not file_name.endswith(".csv"):
                        continue
                    file_path = os.path.join(hand_path, file_name)

                    # Process the CSV file
                    data = process_csv_file(file_path)
                    if data is None:
                        continue

                    # Select the appropriate model for the hand
                    model = left_hand_model if hand == "Left" else right_hand_model
                    
                    # Classify the hand data
                    hand_result = classify_hand_data(data, model, confidence_threshold)
                    
                    # Create a result dictionary
                    clean_result = {
                        "Participant": participant_name,
                        "Hand": hand,
                        "Expected Gesture": gesture,
                        "Predicted Gesture": hand_result["classification"],
                        "Confidence": hand_result["confidence"],
                        "File": file_name,
                        "Correct": hand_result["classification"] == gesture,
                        "Threshold": confidence_threshold
                    }
                    
                    results.append(clean_result)
                    print(f"Processed {file_name}: Expected '{gesture}', Predicted '{clean_result['Predicted Gesture']}' with {clean_result['Confidence']:.2f} confidence (threshold: {confidence_threshold:.2f})")

    return results

def evaluate_all_thresholds():
    """
    Evaluate participant data with different confidence thresholds 
    and plot accuracy against threshold using anonymized participant names.
    """
    print("Loading pretrained models from train_model.py...")
    left_hand_model, right_hand_model = get_trained_models()

    # Get unique participant names for anonymization
    participant_names = sorted([p for p in os.listdir(PARTICIPANTS_DATA_PATH) if os.path.isdir(os.path.join(PARTICIPANTS_DATA_PATH, p))])
    participant_map = {name: f"Participant {i+1}" for i, name in enumerate(participant_names)}
    
    # Thresholds from 0.0 to 1.0, incrementing by 0.01
    thresholds = np.arange(0.0, 1.01, 0.01)
    all_results = []
    threshold_accuracies = []
    
    # Track per-participant accuracies using anonymized names
    participant_accuracies = {anon_name: [] for anon_name in participant_map.values()}
    
    for threshold in thresholds:
        print(f"\nEvaluating with confidence threshold: {threshold:.2f}")
        # Pass the participant map to the evaluation function if needed, or apply mapping after
        results = evaluate_participants_with_threshold(left_hand_model, right_hand_model, threshold)
        all_results.extend(results)
        
        # Calculate accuracy for this threshold
        threshold_df = pd.DataFrame(results)
        if not threshold_df.empty:
            accuracy = threshold_df["Correct"].mean() * 100
            threshold_accuracies.append(accuracy)
            print(f"Accuracy at threshold {threshold:.2f}: {accuracy:.2f}%")
            
            # Calculate per-participant accuracy and map to anonymized names
            participant_stats = threshold_df.groupby("Participant")["Correct"].mean() * 100
            # Ensure all participants (even those with 0 accuracy) are represented for this threshold
            current_threshold_participant_acc = {anon_name: 0.0 for anon_name in participant_map.values()}
            for participant, acc in participant_stats.items():
                if participant in participant_map: # Check if participant is in our map
                    anonymized_name = participant_map[participant]
                    current_threshold_participant_acc[anonymized_name] = acc
                else:
                     print(f"Warning: Participant '{participant}' found in data but not in initial directory scan.")

            # Append the accuracy for each anonymized participant for the current threshold
            for anonymized_name, acc in current_threshold_participant_acc.items():
                 participant_accuracies[anonymized_name].append(acc)

        else:
            threshold_accuracies.append(0)
            # Append 0 accuracy for all participants if no results for this threshold
            for anonymized_name in participant_accuracies:
                participant_accuracies[anonymized_name].append(0)
            print(f"No valid results for threshold {threshold:.2f}")
    
    # Save all results to CSV (original participant names are still in this raw data)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results_all_thresholds.csv", index=False)
    
    # Plot threshold vs accuracy
    plt.figure(figsize=(12, 7)) # Adjusted size slightly
    plt.plot(thresholds, threshold_accuracies, marker='o', linestyle='-', label='Overall', linewidth=2)
    
    # Add per-participant accuracy lines using anonymized names
    for anonymized_name, accuracies in participant_accuracies.items():
         # Ensure the length matches the number of thresholds
        if len(accuracies) == len(thresholds):
            plt.plot(thresholds, accuracies, linestyle='--', label=anonymized_name, alpha=0.7)
        else:
            print(f"Warning: Length mismatch for {anonymized_name}. Expected {len(thresholds)}, got {len(accuracies)}. Skipping plot line.")

    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Gesture Recognition Accuracy vs. Confidence Threshold')
    plt.grid(True)
    plt.legend(loc='best') # Changed legend location for potentially better placement
    plt.tight_layout() # Adjust layout
    plt.savefig('threshold_vs_accuracy.png')
    plt.show()
    
    # Find best threshold
    best_idx = np.argmax(threshold_accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = threshold_accuracies[best_idx]
    print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2f}%")
    
    # Save threshold vs accuracy data with anonymized participant columns
    threshold_data = pd.DataFrame({
        'Threshold': thresholds,
        'Overall_Accuracy': threshold_accuracies # Renamed for clarity
    })
    
    # Add per-participant accuracy columns using anonymized names
    for anonymized_name, accuracies in participant_accuracies.items():
        # Ensure the length matches before adding the column
        if len(accuracies) == len(thresholds):
            threshold_data[anonymized_name] = accuracies
        else:
             print(f"Warning: Length mismatch for {anonymized_name}. Cannot add column to threshold_vs_accuracy.csv.")

        
    threshold_data.to_csv("threshold_vs_accuracy.csv", index=False)
    
    # Generate per-gesture accuracy at best threshold
    best_threshold_results = [r for r in all_results if r["Threshold"] == best_threshold]
    gesture_df = pd.DataFrame(best_threshold_results)
    gesture_stats = pd.DataFrame(best_threshold_results).groupby("Expected Gesture")["Correct"].agg(['mean', 'count'])
    gesture_stats['accuracy'] = gesture_stats['mean'] * 100
    
    print("\nAccuracy by gesture at best threshold:")
    for gesture, row in gesture_stats.iterrows():
        print(f"{gesture}: {row['accuracy']:.2f}% ({int(row['count'] * row['mean'])}/{int(row['count'])} correct)")
    
    return threshold_data

if __name__ == "__main__":
    print("Starting offline evaluation with multiple confidence thresholds...")
    evaluate_all_thresholds()
    print("\nEvaluation completed. Results saved to CSV and graph saved to 'threshold_vs_accuracy.png'.")
