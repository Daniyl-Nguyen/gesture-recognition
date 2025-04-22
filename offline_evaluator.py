import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random  # Added for downsampling
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

def evaluate_participants_with_threshold(left_hand_model, right_hand_model, confidence_threshold):
    """
    Evaluate the data within the Participants directory using the specified confidence threshold,
    applying downsampling to the 'Unknown Gesture' class to achieve a 1:1 ratio with all other gestures combined.
    """
    all_files_metadata = []
    unknown_gesture_files = []
    minority_gesture_files = []
    minority_gesture_counts = {}  # Keep this for potential debugging/info prints

    # --- 1. Collect all file paths and metadata ---
    print("Collecting file metadata...")
    for participant_name in os.listdir(PARTICIPANTS_DATA_PATH):
        participant_path = os.path.join(PARTICIPANTS_DATA_PATH, participant_name)
        if not os.path.isdir(participant_path):
            continue

        for gesture_folder in os.listdir(participant_path):
            gesture_path = os.path.join(participant_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            try:
                _, gesture = gesture_folder.split("-", 1)
                is_unknown = (gesture == "Unknown")
                if is_unknown:
                    gesture = "Unknown Gesture"  # Standardize name
            except ValueError:
                print(f"Skipping folder {gesture_folder}: Unable to extract gesture name.")
                continue

            for hand in ["Left", "Right"]:
                hand_path = os.path.join(gesture_path, hand)
                if not os.path.isdir(hand_path):
                    continue

                for file_name in os.listdir(hand_path):
                    if not file_name.endswith(".csv"):
                        continue
                    file_path = os.path.join(hand_path, file_name)
                    
                    file_info = {
                        "path": file_path,
                        "participant": participant_name,
                        "gesture": gesture,
                        "hand": hand,
                        "file_name": file_name
                    }
                    
                    all_files_metadata.append(file_info)
                    
                    if is_unknown:
                        unknown_gesture_files.append(file_info)
                    else:
                        minority_gesture_files.append(file_info)
                        # Count minority gestures (still useful for info)
                        minority_gesture_counts[gesture] = minority_gesture_counts.get(gesture, 0) + 1

    print(f"Collected metadata for {len(all_files_metadata)} files.")
    print(f"Found {len(unknown_gesture_files)} 'Unknown Gesture' files and {len(minority_gesture_files)} other gesture files.")
    if minority_gesture_counts:
        print(f"Minority gesture counts: {minority_gesture_counts}")

    # --- 2. Determine downsampling target and apply downsampling (1:1 Ratio) ---
    files_to_process = []
    if not minority_gesture_files:
        print("Warning: No minority gesture files found. Evaluating all collected files (mostly Unknown Gesture).")
        # Decide how to handle this: evaluate all, or only unknown? Evaluating all for now.
        files_to_process = all_files_metadata
    elif not unknown_gesture_files:
        print("Warning: No 'Unknown Gesture' files found. Evaluating only minority gestures.")
        files_to_process = minority_gesture_files
    else:
        # Calculate the total count of all minority gestures
        total_minority_count = len(minority_gesture_files)
        # Calculate the target count for Unknown Gesture (1 times the minority count)
        target_unknown_count = total_minority_count * 1
        
        print(f"Total count of minority gestures: {total_minority_count}")
        print(f"Target count for 'Unknown Gesture' (1x minority): {target_unknown_count}")

        downsampled_unknown_files = unknown_gesture_files
        # Downsample if 'Unknown Gesture' count exceeds the target count (1x minority)
        if len(unknown_gesture_files) > target_unknown_count and total_minority_count > 0:
            print(f"Downsampling 'Unknown Gesture' from {len(unknown_gesture_files)} to {target_unknown_count} samples (to achieve 1:1 ratio).")
            downsampled_unknown_files = random.sample(unknown_gesture_files, target_unknown_count)
        elif total_minority_count == 0:
             print("No minority gestures found, processing only 'Unknown Gesture' files without downsampling.")
             # Keep downsampled_unknown_files as is (which is all unknown_gesture_files)
        else:
            # No downsampling needed if unknown count is already <= 1x minority count
            print(f"'Unknown Gesture' count ({len(unknown_gesture_files)}) is not greater than target count ({target_unknown_count}). No downsampling needed.")

        # Combine minority gestures and (potentially downsampled) unknown gestures
        files_to_process.extend(minority_gesture_files)
        files_to_process.extend(downsampled_unknown_files)
        print(f"Total files to process after potential downsampling: {len(files_to_process)} ({len(minority_gesture_files)} minority + {len(downsampled_unknown_files)} unknown)")

    # --- 3. Process the selected files ---
    results = []
    print(f"Processing {len(files_to_process)} files with threshold {confidence_threshold:.2f}...")
    for file_info in files_to_process:
        file_path = file_info["path"]
        participant_name = file_info["participant"]
        gesture = file_info["gesture"]
        hand = file_info["hand"]
        file_name = file_info["file_name"]

        # Process the CSV file
        data = process_csv_file(file_path)
        if data is None:
            print(f"Skipping file due to processing error: {file_path}")
            continue

        # Select the appropriate model for the hand
        model = left_hand_model if hand == "Left" else right_hand_model
        
        # Classify the hand data
        hand_result = classify_hand_data(data, model, confidence_threshold)
        
        # Create a result dictionary
        correct_prediction = hand_result["classification"] == gesture
        clean_result = {
            "Participant": participant_name,
            "Hand": hand,
            "Expected Gesture": gesture,
            "Predicted Gesture": hand_result["classification"],
            "Confidence": hand_result["confidence"],
            "File": file_name,
            "Correct": correct_prediction,
            "Threshold": confidence_threshold
        }
        
        results.append(clean_result)

    return results

def evaluate_all_thresholds():
    """
    Evaluate participant data with different confidence thresholds 
    and plot accuracy against threshold using anonymized participant names.
    1:1 Downsampling is applied within evaluate_participants_with_threshold.
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
        # evaluate_participants_with_threshold now handles downsampling internally
        results = evaluate_participants_with_threshold(left_hand_model, right_hand_model, threshold)
        
        # Check if results were generated for this threshold
        if not results:
            print(f"No results generated for threshold {threshold:.2f} (possibly due to downsampling/data issues). Assigning 0% accuracy.")
            threshold_accuracies.append(0)
            # Append 0 accuracy for all participants if no results for this threshold
            for anonymized_name in participant_accuracies:
                participant_accuracies[anonymized_name].append(0)
            continue

        # Extend all_results only if results were generated
        all_results.extend(results)
        
        # Calculate accuracy for this threshold using the generated results
        threshold_df = pd.DataFrame(results)
        accuracy = threshold_df["Correct"].mean() * 100
        threshold_accuracies.append(accuracy)
        print(f"Accuracy at threshold {threshold:.2f}: {accuracy:.2f}%")
        
        # Calculate per-participant accuracy and map to anonymized names
        participant_stats = threshold_df.groupby("Participant")["Correct"].mean() * 100
        current_threshold_participant_acc = {anon_name: 0.0 for anon_name in participant_map.values()}
        
        for participant, acc in participant_stats.items():
            if participant in participant_map:
                anonymized_name = participant_map[participant]
                current_threshold_participant_acc[anonymized_name] = acc

        # Append the accuracy for each anonymized participant for the current threshold
        for anonymized_name, acc in current_threshold_participant_acc.items():
            participant_accuracies[anonymized_name].append(acc)

    # Save all results to CSV (original participant names are still in this raw data)
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("evaluation_results_all_thresholds_downsampled.csv", index=False)

    # Plot threshold vs accuracy
    plt.figure(figsize=(12, 7))
    plt.plot(thresholds, threshold_accuracies, marker='o', linestyle='-', label='Overall (Downsampled)', linewidth=2)

    # Add per-participant accuracy lines using anonymized names
    for anonymized_name, accuracies in participant_accuracies.items():
        plt.plot(thresholds, accuracies, linestyle='--', label=anonymized_name, alpha=0.7)

    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Gesture Recognition Accuracy vs. Confidence Threshold (Unknown Gesture Downsampled)')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('threshold_vs_accuracy_downsampled.png')
    plt.show()
    
    # Find best threshold
    best_idx = np.argmax(threshold_accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = threshold_accuracies[best_idx]
    print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2f}% (based on 1:1 downsampled data)")
    
    # Save threshold vs accuracy data with anonymized participant columns
    threshold_data = pd.DataFrame({
        'Threshold': thresholds,
        'Overall_Accuracy_Downsampled': threshold_accuracies
    })
    
    for anonymized_name, accuracies in participant_accuracies.items():
        threshold_data[anonymized_name] = accuracies
        
    threshold_data.to_csv("threshold_vs_accuracy_downsampled.csv", index=False)
    
    # Generate per-gesture accuracy at best threshold
    best_threshold_results = [r for r in all_results if r["Threshold"] == best_threshold]
    gesture_df = pd.DataFrame(best_threshold_results)
    gesture_stats = pd.DataFrame(best_threshold_results).groupby("Expected Gesture")["Correct"].agg(['mean', 'count'])
    gesture_stats['accuracy'] = gesture_stats['mean'] * 100
    
    print(f"\nAccuracy by gesture at best threshold ({best_threshold:.2f}) (based on 1:1 downsampled data):")
    for gesture, row in gesture_stats.iterrows():
        print(f"{gesture}: {row['accuracy']:.2f}% ({int(row['count'] * row['mean'])}/{int(row['count'])} correct)")
    
    return threshold_data

if __name__ == "__main__":
    print("Starting offline evaluation with multiple confidence thresholds and 1:1 Unknown Gesture downsampling...")
    evaluate_all_thresholds()
    print("\nEvaluation completed. Results saved to CSV (downsampled) and graph saved to 'threshold_vs_accuracy_downsampled.png'.")
    