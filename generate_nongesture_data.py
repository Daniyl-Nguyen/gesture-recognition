import csv
import random
import os

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


def generate_nongesture_csv(output_dir, hand_type, file_number, min_frames=50, max_frames=150):
    """Generates a single CSV file with random hand landmark data based on the new format."""

    if hand_type not in ["Left", "Right"]:
        raise ValueError("hand_type must be 'Left' or 'Right'")

    joint_names = LEFT_JOINT_NAMES if hand_type == "Left" else RIGHT_JOINT_NAMES
    num_frames = random.randint(min_frames, max_frames)
    file_name = f"Unknown_{hand_type}_{file_number}.csv"
    output_path = os.path.join(output_dir, file_name)

    # Define headers based on the example CSV
    headers = ["Timestamp", "Hand", "Joint", "PositionX", "PositionY", "PositionZ",
               "RotationX", "RotationY", "RotationZ", "RotationW"]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        current_timestamp = 0.0 # Use the same timestamp for all joints in this frame
        for joint_name in joint_names: # Iterate through joints for the current frame
            # Generate random values within approximate observed ranges (with buffer)
            pos_x = random.uniform(-0.1, 0.1)
            pos_y = random.uniform(-0.2, -0.1)
            pos_z = random.uniform(0.15, 0.40)
            rot_x = random.uniform(0.0, 0.8)
            rot_y = random.uniform(-0.6, 0.5)
            rot_z = random.uniform(-0.7, -0.1)
            rot_w = random.uniform(0.4, 0.8)

            row = [current_timestamp, hand_type, joint_name,
                    pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
            writer.writerow(row)

    print(f"Generated: {output_path}")

# --- How to use it ---

# Create base directories if they don't exist
base_dir = r"c:\Users\daniy\gesture-recognition\Unknown"
left_dir = os.path.join(base_dir, "Left")
right_dir = os.path.join(base_dir, "Right")
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

# Generate 10 files for Left hand
for i in range(1, 11):
    generate_nongesture_csv(left_dir, "Left", i)

# Generate 10 files for Right hand
for i in range(1, 11):
    generate_nongesture_csv(right_dir, "Right", i)

print("Finished generating Unknown data.")
