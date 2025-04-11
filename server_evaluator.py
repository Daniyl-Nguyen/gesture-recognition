import os
import pandas as pd
import requests
import time

# Define the Participants dataset path and API endpoint
PARTICIPANTS_DATA_PATH = "c:\\Users\\daniy\\gesture-recognition\\Participants"
API_ENDPOINT = "http://127.0.0.1:8000/classify_data"

# Define constants
FLOATS_PER_HAND = 147  # 21 joints * 7 floats (posXYZ, rotXYZW)
EXPECTED_DATA_LENGTH = FLOATS_PER_HAND  # Updated to expect data for a single hand

def check_models_loaded():
    """
    Basic check if server is responding
    """
    try:
        test_data = [0] * FLOATS_PER_HAND * 2  # Data for both hands
        requests.post(API_ENDPOINT, json={"data": test_data})
        return True
    except:
        return False

def process_csv_file(file_path):
    """
    Process a single CSV file and extract features as a flat list.
    """
    try:
        df = pd.read_csv(file_path)
        # Extract only the Position and Rotation columns (assuming the same structure as the example data)
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

def evaluate_participants():
    """
    Evaluate the data within the Participants directory by sending it to the classify endpoint.
    """
    # First check if models are loaded
    if not check_models_loaded():
        print("Error: Hand models are not loaded. Please start the server with proper model loading.")
        print("Hint: Check the API server logs for model loading errors.")
        return
        
    results = []

    # Walk through the Participants dataset directory
    for participant_name in os.listdir(PARTICIPANTS_DATA_PATH):
        participant_path = os.path.join(PARTICIPANTS_DATA_PATH, participant_name)
        if not os.path.isdir(participant_path):
            continue

        # Loop through gesture folders (e.g., Dan-Goodbye)
        for gesture_folder in os.listdir(participant_path):
            gesture_path = os.path.join(participant_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            # Extract gesture name from the folder name (e.g., "Dan-Goodbye" -> "Goodbye")
            try:
                _, gesture = gesture_folder.split("-", 1)
            except ValueError:
                print(f"Skipping folder {gesture_folder}: Unable to extract gesture name.")
                continue

            # Process Left and Right hands
            for hand in ["Left", "Right"]:  # Updated to match folder names
                hand_path = os.path.join(gesture_path, hand)
                if not os.path.isdir(hand_path):
                    continue

                # Process each CSV file in the directory
                for file_name in os.listdir(hand_path):
                    if not file_name.endswith(".csv"):  # Updated to handle .csv files
                        continue
                    file_path = os.path.join(hand_path, file_name)

                    # Process the CSV file
                    data = process_csv_file(file_path)
                    if data is None:
                        continue

                    # Properly format data for the API (which expects both hands)
                    # Instead of padding with zeros, duplicate the hand data
                    formatted_data = data + data  # Duplicate the hand data to fill both hand slots
                    
                    # Send data to the classify endpoint
                    try:
                        response = requests.post(API_ENDPOINT, json={"data": formatted_data})
                        if response.status_code == 200:
                            result = response.json()
                            # Only consider the result for the current hand
                            hand_result = result.get('left_hand' if hand == 'Left' else 'right_hand', {})
                            
                            # Create a cleaner result dictionary with better column names
                            clean_result = {
                                "Participant": participant_name,
                                "Hand": hand,
                                "Expected Gesture": gesture,
                                "Predicted Gesture": hand_result.get("classification", "Unknown"),
                                "Confidence": hand_result.get("confidence", 0.0),
                                "File": file_name,
                                # Store if prediction was correct
                                "Correct": hand_result.get("classification", "Unknown") == gesture
                            }
                            
                            results.append(clean_result)
                            print(f"Processed {file_name}: Expected '{gesture}', Predicted '{clean_result['Predicted Gesture']}' with {clean_result['Confidence']:.2f} confidence")
                        else:
                            print(f"Error: Received status code {response.status_code} for {file_name}")
                    except Exception as e:
                        print(f"Error sending data for {file_name}: {e}")

    # Save results to a CSV file with clear column names
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    if not results_df.empty:
        accuracy = results_df["Correct"].mean() * 100
        print(f"\nOverall accuracy: {accuracy:.2f}%")
        
        # Group by participant and calculate per-participant accuracy
        participant_stats = results_df.groupby("Participant")["Correct"].agg(['mean', 'count'])
        participant_stats['accuracy'] = participant_stats['mean'] * 100
        
        print("\nPer-participant accuracy:")
        for participant, row in participant_stats.iterrows():
            print(f"{participant}: {row['accuracy']:.2f}% ({int(row['count'] * row['mean'])}/{int(row['count'])} correct)")
    
    # Save the results
    results_df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation completed. Results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    evaluate_participants()
