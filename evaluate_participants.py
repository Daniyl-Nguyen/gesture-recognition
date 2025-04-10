import os
import pandas as pd
import requests

# Define the Participants dataset path and API endpoint
PARTICIPANTS_DATA_PATH = "c:\\Users\\daniy\\gesture-recognition\\Participants"
API_ENDPOINT = "http://127.0.0.1:8000/classify_data"

# Define constants
FLOATS_PER_HAND = 147  # 21 joints * 7 floats (posXYZ, rotXYZW)
EXPECTED_DATA_LENGTH = FLOATS_PER_HAND  # Updated to expect data for a single hand

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

                    # Pad the data with zeros for the other hand
                    if hand == "Left":
                        data = data + [0] * FLOATS_PER_HAND  # Pad for the right hand
                    elif hand == "Right":
                        data = [0] * FLOATS_PER_HAND + data  # Pad for the left hand

                    # Send data to the classify endpoint
                    try:
                        response = requests.post(API_ENDPOINT, json={"data": data})
                        if response.status_code == 200:
                            result = response.json()
                            result.update({
                                "participant": participant_name,
                                "gesture": gesture,
                                "hand": hand,  # Updated to use "Left" or "Right"
                                "file": file_name
                            })
                            results.append(result)
                            print(f"Processed {file_name}: {result}")
                        else:
                            print(f"Error: Received status code {response.status_code} for {file_name}")
                    except Exception as e:
                        print(f"Error sending data for {file_name}: {e}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation completed. Results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    evaluate_participants()
