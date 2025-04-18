import csv
import random
import os

# --- Configuration ---
PARTICIPANTS = ["Finch", "Mostafa", "Pouya", "Mykola"]
SAMPLES_PER_PARTICIPANT_HAND = 50 # Number of files to generate per hand for each participant
BASE_OUTPUT_DIR = r"c:\Users\daniy\gesture-recognition\Participants" # Changed base directory
# --- End Configuration ---


# Define standard joint names based on the example format
LEFT_JOINT_NAMES = [
    "Wrist", "Left_IndexProximal", "Left_IndexIntermediate", "Left_IndexDistal", "Left_IndexDistalEnd",
    "Left_MiddleProximal", "Left_MiddleIntermediate", "Left_MiddleDistal", "Left_MiddleDistalEnd",
    "Left_PinkyProximal", "Left_PinkyIntermediate", "Left_PinkyDistal", "Left_PinkyDistalEnd",
    "Left_RingProximal", "Left_RingIntermediate", "Left_RingDistal", "Left_RingDistalEnd",
    "Left_ThumbProximal", "Left_ThumbIntermediate", "Left_ThumbDistal", "Left_ThumbDistalEnd"
]

# Assuming Right hand joints follow a similar naming convention
RIGHT_JOINT_NAMES = [name.replace("Left_", "Right_") if name != "Wrist" else name for name in LEFT_JOINT_NAMES]


def generate_nongesture_csv(base_output_dir, participant_name, hand_type, file_number):
    """Generates a single CSV file with random hand landmark data for a specific participant."""

    if hand_type not in ["Left", "Right"]:
        raise ValueError("hand_type must be 'Left' or 'Right'")

    # Define gesture folder name
    gesture_folder_name = f"{participant_name}-Unknown" # Use "Unknown"

    # Construct the full output directory path
    output_dir = os.path.join(base_output_dir, participant_name, gesture_folder_name, hand_type)
    os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

    joint_names = LEFT_JOINT_NAMES if hand_type == "Left" else RIGHT_JOINT_NAMES
    file_name = f"{participant_name}-Unknown_{hand_type}_{file_number}.csv" # Updated filename format
    output_path = os.path.join(output_dir, file_name)

    # Define headers based on the example CSV
    headers = ["Timestamp", "Hand", "Joint", "PositionX", "PositionY", "PositionZ",
               "RotationX", "RotationY", "RotationZ", "RotationW"]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        current_timestamp = 0.0 # Use the same timestamp for all joints in this frame
        for joint_name in joint_names: # Iterate through joints for the current frame
            # Generate random values using the wider observed ranges
            pos_x = random.uniform(-0.320, 0.141)
            pos_y = random.uniform(-0.359, 0.109)
            pos_z = random.uniform(0.179, 0.463)
            rot_x = random.uniform(-0.748, 0.955)
            rot_y = random.uniform(-0.832, 0.790)
            rot_z = random.uniform(-0.822, 0.862)
            rot_w = random.uniform(-0.771, 0.992) # Corrected min value based on samples

            row = [current_timestamp, hand_type, joint_name,
                    pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
            writer.writerow(row)

    print(f"Generated: {output_path}")

# --- Main Execution Logic ---

print(f"Starting generation of 'Unknown Gesture' data for {len(PARTICIPANTS)} participants...")
print(f"Generating {SAMPLES_PER_PARTICIPANT_HAND} samples per hand for each participant.")

for participant in PARTICIPANTS:
    print(f"\nProcessing participant: {participant}")
    for hand in ["Left", "Right"]:
        print(f"  Generating for {hand} hand...")
        for i in range(1, SAMPLES_PER_PARTICIPANT_HAND + 1):
            generate_nongesture_csv(BASE_OUTPUT_DIR, participant, hand, i)

print("\nFinished generating Unknown Gesture data.")
