"""
This script requires the following Python packages:
- matplotlib
- numpy
- pandas

You can install them using pip:
    pip install matplotlib numpy pandas

It visualizes the first frame/timestamp of the first 'Left' hand CSV file found
in each gesture subdirectory within the 'dataset' folder, combining them into
a single grid image.
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import argparse
import os
import re
import math
import glob

# Define the bone connections for hand visualization
BONES = (
    (0, 1),
    (0, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (1, 10),
    (1, 14),
    (1, 18),
    (6, 7),
    (10, 11),
    (14, 15),
    (18, 19),
    (7, 8),
    (11, 12),
    (15, 16),
    (19, 20),
    (8, 9),
    (12, 13),
    (16, 17),
    (20, 21),
)

# Define joint mapping from your CSV format to the expected order
JOINT_MAPPING = {
    'Wrist': 0,
    'Left_ThumbProximal': 1,
    'Left_ThumbIntermediate': 2,
    'Left_ThumbDistal': 3,
    'Left_ThumbDistalEnd': 4,
    'Left_IndexProximal': 5,
    'Left_IndexIntermediate': 6,
    'Left_IndexDistal': 7,
    'Left_IndexDistalEnd': 8,
    'Left_MiddleProximal': 9,
    'Left_MiddleIntermediate': 10,
    'Left_MiddleDistal': 11,
    'Left_MiddleDistalEnd': 12,
    'Left_RingProximal': 13,
    'Left_RingIntermediate': 14,
    'Left_RingDistal': 15,
    'Left_RingDistalEnd': 16,
    'Left_PinkyProximal': 17,
    'Left_PinkyIntermediate': 18,
    'Left_PinkyDistal': 19,
    'Left_PinkyDistalEnd': 20,
}

# Updated bone connections to match the joint mapping above
HAND_BONES = [
    (0, 1),  # Wrist to Thumb Proximal
    (1, 2),  # Thumb Proximal to Intermediate
    (2, 3),  # Thumb Intermediate to Distal
    (3, 4),  # Thumb Distal to End
    
    (0, 5),  # Wrist to Index Proximal
    (5, 6),  # Index Proximal to Intermediate
    (6, 7),  # Index Intermediate to Distal
    (7, 8),  # Index Distal to End
    
    (0, 9),  # Wrist to Middle Proximal
    (9, 10), # Middle Proximal to Intermediate
    (10, 11), # Middle Intermediate to Distal
    (11, 12), # Middle Distal to End
    
    (0, 13), # Wrist to Ring Proximal
    (13, 14), # Ring Proximal to Intermediate
    (14, 15), # Ring Intermediate to Distal
    (15, 16), # Ring Distal to End
    
    (0, 17), # Wrist to Pinky Proximal
    (17, 18), # Pinky Proximal to Intermediate
    (18, 19), # Pinky Intermediate to Distal
    (19, 20), # Pinky Distal to End
]

def parse_hand_csv(csv_file, timestamp=None):
    """
    Parse hand tracking data from CSV file with the specific format

    Parameters:
        csv_file: Path to the CSV file
        timestamp: If provided, only extract data for this timestamp

    Returns:
        A numpy array of shape (num_joints, 3) containing joint positions, or None if error.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if essential columns exist
        required_cols = ['Timestamp', 'Joint', 'PositionX', 'PositionY', 'PositionZ']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: CSV file {csv_file} is missing required columns. Skipping.")
            return None

        # If timestamp is provided, filter by that timestamp
        if timestamp is not None:
            df_filtered = df[df['Timestamp'] == timestamp]
        else:
            # Otherwise, get the first timestamp in the file
            unique_timestamps = df['Timestamp'].unique()
            if len(unique_timestamps) == 0:
                print(f"Warning: No timestamps found in {csv_file}. Skipping.")
                return None
            df_filtered = df[df['Timestamp'] == unique_timestamps[0]]

        if df_filtered.empty:
             print(f"Warning: No data found for the selected timestamp in {csv_file}. Skipping.")
             return None

        # Extract position data
        positions = np.zeros((21, 3))

        for _, row in df_filtered.iterrows():
            joint_name = row['Joint']
            if joint_name in JOINT_MAPPING:
                joint_idx = JOINT_MAPPING[joint_name]
                positions[joint_idx] = [row['PositionX'], row['PositionY'], row['PositionZ']]

        return positions

    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File {csv_file} is empty.")
        return None
    except Exception as e:
        print(f"Error parsing CSV file {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_static_hand_on_ax(ax, coordinates, title):
    """
    Plots a static 3D hand pose onto a given Matplotlib Axes object,
    hiding the axes and grid lines.

    Parameters:
        ax: The Matplotlib 3D Axes object to plot on.
        coordinates: numpy array of shape (num_joints, 3).
        title: The title for the subplot.
    """
    ax.cla()

    # --- Plotting the hand (joints and bones) ---
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='blue', s=15, alpha=0.8, depthshade=True)

    for start_idx, end_idx in HAND_BONES:
        if start_idx < len(coordinates) and end_idx < len(coordinates):
            xs = [coordinates[start_idx, 0], coordinates[end_idx, 0]]
            ys = [coordinates[start_idx, 1], coordinates[end_idx, 1]]
            zs = [coordinates[start_idx, 2], coordinates[end_idx, 2]]
            ax.plot(xs, ys, zs, 'r-', linewidth=1.5)
    # --- End Plotting ---

    # --- Set limits to frame the hand ---
    min_coords = np.min(coordinates, axis=0) - 0.05
    max_coords = np.max(coordinates, axis=0) + 0.05
    center = (min_coords + max_coords) / 2
    max_range = np.max(max_coords - min_coords) * 0.6

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    # --- End Setting limits ---

    # Set title
    ax.set_title(title, fontsize=10)

    # Set viewpoint
    ax.view_init(elev=25, azim=60)

    # --- Hide axes and grid ---
    ax.set_axis_off()
    # --- End Hiding ---

def main():
    dataset_path = "dataset"
    output_filename = "combined_gestures_left_hand_first_frame.png"
    left_subfolder_name = "Left" # Define the subfolder name

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found.")
        return

    # Get all directories
    all_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    all_gesture_coords = []
    gesture_names = []

    print(f"Processing all gestures in: {dataset_path}")
    # Iterate through all directories found
    for gesture_name in all_dirs:
        # --- Processing logic for all gestures ---
        gesture_dir_path = os.path.join(dataset_path, gesture_name)
        left_folder_path = os.path.join(gesture_dir_path, left_subfolder_name) # Construct path to Left subfolder

        # Check if the Left subfolder exists
        if not os.path.isdir(left_folder_path):
            print(f"  - Gesture '{gesture_name}': '{left_subfolder_name}' subfolder not found. Skipping.")
            continue

        # Find CSV files inside the Left subfolder
        csv_files = sorted(glob.glob(os.path.join(left_folder_path, '*.csv')))

        if not csv_files:
            print(f"  - Gesture '{gesture_name}': No CSV files found in '{left_subfolder_name}' subfolder. Skipping.")
            continue

        first_left_csv = csv_files[0] # Get the first CSV file in the Left folder
        print(f"  - Gesture '{gesture_name}': Processing '{os.path.basename(first_left_csv)}' from '{left_subfolder_name}' folder")

        coordinates = parse_hand_csv(first_left_csv)

        if coordinates is not None and coordinates.shape == (21, 3):
            all_gesture_coords.append(coordinates)
            gesture_names.append(gesture_name) # Use the actual gesture name found
        else:
            print(f"    Warning: Failed to parse valid coordinates from {os.path.basename(first_left_csv)}. Skipping.")
        # --- End of processing logic ---


    if not all_gesture_coords:
        print("No valid gesture data could be processed. Exiting.")
        return

    num_gestures = len(all_gesture_coords)
    print(f"\nSuccessfully processed {num_gestures} gestures.")

    # --- Adjust layout for 2x5 grid ---
    num_rows = 2
    num_cols = 5
    # Adjust figsize for the 2x5 grid
    fig_width = num_cols * 3
    fig_height = num_rows * 3.5

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height),
                             subplot_kw={'projection': '3d'})
    # --- End layout adjustment ---


    # Ensure axes is always iterable and flatten it for easy indexing
    if num_gestures == 0: # Should be caught earlier, but handle defensively
         axes_flat = []
    elif num_rows == 1 and num_cols == 1:
         axes_flat = [axes]
    elif num_rows == 1 or num_cols == 1:
         axes_flat = axes # Already 1D
    else:
         axes_flat = axes.flatten() # Flatten the 2D array

    print(f"Creating combined plot ({num_rows}x{num_cols} grid)...")
    for i in range(num_gestures):
        # Check if index is within bounds of flattened axes
        if i < len(axes_flat):
            ax = axes_flat[i]
            coords = all_gesture_coords[i]
            title = gesture_names[i]
            plot_static_hand_on_ax(ax, coords, title)
        else:
            print(f"Warning: More gestures ({num_gestures}) than available subplots ({len(axes_flat)}). Skipping gesture '{gesture_names[i]}'.")
            break # Stop if we run out of subplots

    # Hide unused subplots
    for j in range(num_gestures, num_rows * num_cols):
         if j < len(axes_flat):
            axes_flat[j].set_visible(False)


    fig.suptitle("First Frame of All Left Hand Gestures", fontsize=16)

    # Adjust tight_layout rect and reduce padding between subplots
    # w_pad: width padding, h_pad: height padding
    plt.tight_layout(rect=[0, 0.03, 0.8, 0.8], w_pad=0, h_pad=0) # Adjust w_pad and h_pad as needed

    try:
        plt.savefig(output_filename, dpi=200)
        print(f"Combined visualization saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving the figure: {e}")

    print("Displaying combined plot...")
    plt.show()

if __name__ == "__main__":
    main()
