"""
This script requires the following Python packages:
- matplotlib
- numpy
- pandas

You can install them using pip:
    pip install matplotlib numpy pandas

If you encounter the error 'ModuleNotFoundError: No module named 'matplotlib'',
run the above installation command in your terminal/command prompt first.
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import argparse
import os
import re

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
    # Add right hand mappings if needed
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

def read_csv_data(csv_path):
    """
    Read 3D hand tracking data from a CSV file
    Expected format: Each row is a flattened frame containing x,y,z coordinates for each joint
    """
    df = pd.read_csv(csv_path)
    
    # Check if the data is already in the correct format or needs restructuring
    num_columns = df.shape[1]
    
    if num_columns == 66:  # 22 joints × 3 coordinates
        # Data is in flattened format (x1,y1,z1,x2,y2,z2,...,x22,y22,z22)
        frames = []
        for _, row in df.iterrows():
            frame = []
            values = row.values
            # Reshape to 22 joints with 3 coordinates each
            for i in range(0, num_columns, 3):
                if i+2 < num_columns:  # Safety check
                    joint = [values[i], values[i+1], values[i+2]]
                    frame.append(joint)
            frames.append(frame)
        
        return np.array(frames)
    
    elif num_columns % 3 == 0:  # Multiple of 3 indicates potential x,y,z columns for each joint
        num_joints = num_columns // 3
        frames = []
        for _, row in df.iterrows():
            frame = []
            values = row.values
            for i in range(num_joints):
                joint = [values[i*3], values[i*3+1], values[i*3+2]]
                frame.append(joint)
            frames.append(frame)
        
        return np.array(frames)
    
    else:
        # Try to infer format from column names
        print("Warning: CSV format couldn't be automatically determined.")
        print("Attempting to read data assuming each row is a flattened frame...")
        
        # Assuming all numeric data represents coordinates
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) % 3 == 0:
            num_joints = len(numeric_cols) // 3
            frames = []
            for _, row in df.iterrows():
                frame = []
                for i in range(num_joints):
                    joint = [row[numeric_cols[i*3]], row[numeric_cols[i*3+1]], row[numeric_cols[i*3+2]]]
                    frame.append(joint)
                frames.append(frame)
            
            return np.array(frames)
        
        raise ValueError("Unable to parse CSV data. Please ensure it contains 3D coordinates for hand joints.")

def extract_coordinates(df, coordinate_pattern=None, frame_index=0):
    """
    Extract only position coordinates from the dataframe, filtering out rotation or other data
    
    Parameters:
        df: DataFrame containing the data
        coordinate_pattern: Regex pattern to identify coordinate columns (default: look for x, y, z in column names)
        frame_index: Which frame to extract (row index)
    
    Returns:
        Array of shape (22, 3) containing the coordinates
    """
    # Default pattern looks for columns with x, y, z in their names
    if coordinate_pattern is None:
        coordinate_pattern = r'(?i).*(x|position.*?x|pos.*?x).*'  # Match x coordinates
    
    # Get all column names
    all_columns = df.columns.tolist()
    
    # Find columns matching the pattern for x coordinates
    x_columns = [col for col in all_columns if re.match(coordinate_pattern, col, re.IGNORECASE)]
    
    # If we can't find columns matching the pattern, fall back to assuming the data is already structured correctly
    if not x_columns:
        print("Warning: Could not identify coordinate columns using pattern. Using all numeric columns.")
        return read_csv_data(df)[frame_index]
    
    # Find corresponding y and z columns by replacing x with y and z in the column names
    coordinates = []
    for x_col in x_columns:
        # Create patterns for corresponding y and z columns
        y_col = re.sub(r'(?i)x', 'y', x_col)
        z_col = re.sub(r'(?i)x', 'z', x_col)
        
        # Check if y and z columns exist
        if y_col in all_columns and z_col in all_columns:
            # Extract the joint number or identifier if it exists in the column name
            joint_match = re.search(r'(\d+)', x_col)
            joint_id = int(joint_match.group(1)) if joint_match else len(coordinates)
            
            # Add the coordinates to our list
            if frame_index < len(df):
                coordinates.append((joint_id, [df.iloc[frame_index][x_col], 
                                             df.iloc[frame_index][y_col], 
                                             df.iloc[frame_index][z_col]]))
    
    # Sort by joint id and extract just the coordinates
    coordinates.sort(key=lambda x: x[0])
    joint_coords = [coord[1] for coord in coordinates]
    
    # If we don't have 22 joints, print a warning
    if len(joint_coords) != 22:
        print(f"Warning: Found {len(joint_coords)} joints instead of the expected 22.")
    
    return np.array(joint_coords)

def parse_hand_csv(csv_file, timestamp=None):
    """
    Parse hand tracking data from CSV file with the specific format
    
    Parameters:
        csv_file: Path to the CSV file
        timestamp: If provided, only extract data for this timestamp
    
    Returns:
        A numpy array of shape (num_joints, 3) containing joint positions
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # If timestamp is provided, filter by that timestamp
    if timestamp is not None:
        df = df[df['Timestamp'] == timestamp]
    else:
        # Otherwise, get the first timestamp in the file
        unique_timestamps = df['Timestamp'].unique()
        if len(unique_timestamps) > 0:
            df = df[df['Timestamp'] == unique_timestamps[0]]
    
    # Extract position data
    positions = np.zeros((21, 3))  # 21 joints per hand
    
    for _, row in df.iterrows():
        joint_name = row['Joint']
        if joint_name in JOINT_MAPPING:
            joint_idx = JOINT_MAPPING[joint_name]
            positions[joint_idx] = [row['PositionX'], row['PositionY'], row['PositionZ']]
    
    return positions

def viz_hand_frames(frames, output_path=None, show=False):
    """
    Visualize hand tracking data as a 3D animation
    
    Parameters:
        frames: numpy array of shape (num_frames, num_joints, 3)
        output_path: path to save the animation (without extension)
        show: whether to display the animation in a window
    """
    frames = frames.reshape(-1, 22, 3)  # Ensure correct shape
    
    # Calculate bounds for the plot
    all_coords = frames.reshape(-1, 3)
    min_coords = np.min(all_coords, axis=0) - 0.1
    max_coords = np.max(all_coords, axis=0) + 0.1
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([min_coords[0], max_coords[0]])
    ax.set_ylim3d([min_coords[1], max_coords[1]])
    ax.set_zlim3d([min_coords[2], max_coords[2]])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Movement Visualization')

    lines = []
    for bone in BONES:
        lines.append(ax.plot([], [], [], lw=2)[0])

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines
    
    def animate(i):
        for line, bone in zip(lines, BONES):
            line.set_data(frames[i, bone, 0], frames[i, bone, 1])
            line.set_3d_properties(frames[i, bone, 2])
        ax.view_init(elev=30, azim=i/2)  # Rotate view slightly for better 3D effect
        return lines
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(frames), 
                        interval=100, blit=True)
    
    if output_path:
        print(f"Saving animation to {output_path}.mp4")
        anim.save(f'{output_path}.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    
    if show:
        plt.show()
    
    plt.close()

def viz_static_hand(coordinates, output_path=None, show=True):
    """
    Create a static 3D visualization of a hand pose
    
    Parameters:
        coordinates: numpy array of shape (num_joints, 3)
        output_path: path to save the image (without extension)
        show: whether to display the image
    """
    # Create the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds for the plot
    min_coords = np.min(coordinates, axis=0) - 0.05
    max_coords = np.max(coordinates, axis=0) + 0.05
    
    # Set axis limits
    ax.set_xlim3d([min_coords[0], max_coords[0]])
    ax.set_ylim3d([min_coords[1], max_coords[1]])
    ax.set_zlim3d([min_coords[2], max_coords[2]])
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Pose Visualization')
    
    # Plot the joints as points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='blue', s=50, alpha=0.8)
    
    # Plot the bones as lines
    for start_idx, end_idx in HAND_BONES:
        if start_idx < len(coordinates) and end_idx < len(coordinates):
            xs = [coordinates[start_idx, 0], coordinates[end_idx, 0]]
            ys = [coordinates[start_idx, 1], coordinates[end_idx, 1]]
            zs = [coordinates[start_idx, 2], coordinates[end_idx, 2]]
            ax.plot(xs, ys, zs, 'r-', linewidth=2)
    
    # Set a good viewpoint
    ax.view_init(elev=30, azim=45)
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}.png")
    
    # Show the figure if requested
    if show:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize hand tracking data from CSV file')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing hand tracking data')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output path for the visualization (without extension)')
    parser.add_argument('--show', '-s', action='store_true', help='Show the visualization in a window')
    parser.add_argument('--frame', '-f', type=int, default=0, 
                        help='Frame index to visualize (default: 0, the first frame)')
    parser.add_argument('--pattern', '-p', type=str, default=None,
                        help='Regex pattern to identify coordinate columns')
    parser.add_argument('--animate', '-a', action='store_true', 
                        help='Create an animation instead of a static image')
    parser.add_argument('--timestamp', '-t', type=float, default=None, 
                        help='Specific timestamp to visualize (default: first timestamp in the file)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found")
        return
    
    # Default output path if not specified
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        args.output = f"hand_visualization_{base_name}"
        if args.frame > 0:
            args.output += f"_frame{args.frame}"
    
    try:
        # Read the CSV file
        df = pd.read_csv(args.csv_file)
        print(f"Loaded CSV data with {len(df)} rows and {len(df.columns)} columns")
        
        if args.animate:
            # Use the original animation function
            frames = read_csv_data(args.csv_file)
            print(f"Loaded {len(frames)} frames of hand tracking data")
            viz_hand_frames(frames, args.output, args.show)
        else:
            # Extract coordinates for the specified frame
            coordinates = parse_hand_csv(args.csv_file, args.timestamp)
            print(f"Extracted coordinates for frame {args.frame}")
            
            # Visualize the static hand pose
            viz_static_hand(coordinates, args.output, args.show)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
