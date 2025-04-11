import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import os

# Create output directory for plots if it doesn't exist
output_dir = "confusion_matrices"
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
csv_path = "evaluation_results.csv"
df = pd.read_csv(csv_path)

# Get unique participants and gestures
participants = df['Participant'].unique()
gestures = df['Expected Gesture'].unique()

# Get a list of all unique gestures (both expected and predicted)
all_gestures = sorted(set(df['Expected Gesture'].unique()) | set(df['Predicted Gesture'].unique()))

def create_confusion_matrix(data, title, filename):
    """Create and save a confusion matrix visualization"""
    # Convert gesture names to numeric labels
    y_true = [all_gestures.index(gesture) for gesture in data['Expected Gesture']]
    y_pred = [all_gestures.index(gesture) for gesture in data['Predicted Gesture']]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(all_gestures)))
    
    # Convert counts to percentages by row (normalize)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percent = np.nan_to_num(cm_percent)  # Replace NaN with 0
    
    # Calculate F1 score (macro average)
    f1 = f1_score(y_true, y_pred, labels=range(len(all_gestures)), average='macro', zero_division=0)
    
    # Calculate accuracy
    accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true) if len(y_true) > 0 else 0
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=all_gestures, yticklabels=all_gestures)
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}, Macro F1 Score: {f1:.4f}')
    plt.xlabel('Predicted Gesture')
    plt.ylabel('Expected Gesture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Create confusion matrices for each hand
print("Generating confusion matrices for each hand...")
for hand in ['Left', 'Right']:
    hand_data = df[df['Hand'] == hand]
    title = f'Confusion Matrix for {hand} Hand'
    filename = f'confusion_matrix_{hand}_hand.png'
    create_confusion_matrix(hand_data, title, filename)
    print(f"Created confusion matrix for {hand} hand")

# Create overall confusion matrix
print("Generating overall confusion matrix...")
title = 'Overall Confusion Matrix'
filename = 'confusion_matrix_overall.png'
create_confusion_matrix(df, title, filename)
print("Created overall confusion matrix")
