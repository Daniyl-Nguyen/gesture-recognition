import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import os

# --- Configuration ---
# Define the threshold you want to generate the confusion matrix for
TARGET_THRESHOLD = 0.7 
# Define the input CSV file containing results for all thresholds
CSV_PATH = "evaluation_results_all_thresholds_downsampled_1_1.csv" 
# --- End Configuration ---


# Create output directory for plots if it doesn't exist
output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file containing all threshold results
try:
    df_all = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Input file not found at {CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error reading CSV file {CSV_PATH}: {e}")
    exit()

# Filter the DataFrame for the target threshold
# Use np.isclose for floating point comparison
df = df_all[np.isclose(df_all['Threshold'], TARGET_THRESHOLD)].copy()

if df.empty:
    print(f"Error: No data found for the target threshold {TARGET_THRESHOLD} in {CSV_PATH}")
    exit()

print(f"Generating confusion matrices for threshold: {TARGET_THRESHOLD}")

# Get unique participants and gestures from the filtered data
participants = df['Participant'].unique()
# Get a list of all unique gestures (both expected and predicted) from the filtered data
# Including "Unknown Gesture" if it appears in predictions at this threshold
all_gestures = sorted(set(df['Expected Gesture'].unique()) | set(df['Predicted Gesture'].unique()))

def create_confusion_matrix(data, title, filename):
    """Create and save a confusion matrix visualization"""
    if data.empty:
        print(f"Skipping {filename}: No data provided.")
        return
        
    # Convert gesture names to numeric labels based on the 'all_gestures' list
    try:
        y_true_labels = data['Expected Gesture']
        y_pred_labels = data['Predicted Gesture']
        
        # Map labels to indices, handle potential missing labels if necessary
        label_to_index = {label: i for i, label in enumerate(all_gestures)}
        y_true = [label_to_index.get(label, -1) for label in y_true_labels] # Use -1 for unknown/missing
        y_pred = [label_to_index.get(label, -1) for label in y_pred_labels]

        # Filter out any instances where mapping failed (index -1) if needed, though ideally all labels should be in all_gestures
        valid_indices = [i for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)) if true_idx != -1 and pred_idx != -1]
        if len(valid_indices) != len(y_true):
             print(f"Warning: Some labels in {filename} data were not found in the 'all_gestures' list. These instances will be ignored in metrics.")
             y_true = [y_true[i] for i in valid_indices]
             y_pred = [y_pred[i] for i in valid_indices]

        if not y_true: # Check if list is empty after filtering
             print(f"Skipping {filename}: No valid mapped labels found.")
             return

    except KeyError as e:
        print(f"Error mapping labels for {filename}: Missing key {e}. Ensure 'all_gestures' is comprehensive.")
        return
    except Exception as e:
        print(f"Error during label mapping for {filename}: {e}")
        return

    # Define labels for confusion matrix (indices 0 to n-1)
    cm_labels = range(len(all_gestures))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    
    # Convert counts to percentages by row (normalize)
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore division by zero for rows with no samples
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.nan_to_num(cm_percent)  # Replace NaN with 0

    # Calculate F1 score (macro average) - use labels present in y_true or y_pred
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    f1 = f1_score(y_true, y_pred, labels=present_labels, average='macro', zero_division=0)
    
    # Calculate accuracy (Correct Predictions / Total Predictions)
    # This calculation is equivalent to df['Correct'].mean() for the filtered dataframe
    accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true) if len(y_true) > 0 else 0
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_gestures, yticklabels=all_gestures, annot_kws={"size": 8}) # Smaller annotation font
    plt.title(f'{title} (Threshold={TARGET_THRESHOLD:.2f})\nAccuracy: {accuracy:.4f}, Macro F1 Score: {f1:.4f}')
    plt.xlabel('Predicted Gesture')
    plt.ylabel('Expected Gesture')
    plt.xticks(rotation=45, ha='right', fontsize=9) # Smaller tick labels
    plt.yticks(rotation=0, fontsize=9) # Smaller tick labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"  Saved: {filename} (Accuracy: {accuracy:.4f})")


# Create confusion matrices for each hand using the filtered data
print("\nGenerating confusion matrices for each hand...")
for hand in ['Left', 'Right']:
    hand_data = df[df['Hand'] == hand]
    title = f'Confusion Matrix for {hand} Hand'
    filename = f'confusion_matrix_{hand}_hand_thresh{TARGET_THRESHOLD:.2f}.png' # Add threshold to filename
    create_confusion_matrix(hand_data, title, filename)


# Create overall confusion matrix using the filtered data
print("\nGenerating overall confusion matrix...")
title = 'Overall Confusion Matrix'
filename = f'confusion_matrix_overall_thresh{TARGET_THRESHOLD:.2f}.png' # Add threshold to filename
create_confusion_matrix(df, title, filename)

print(f"\nConfusion matrix generation complete for threshold {TARGET_THRESHOLD:.2f}.")
