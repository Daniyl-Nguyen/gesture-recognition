import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Existing function to extract the floating point values from the dataframe.
def extract_floats(df):
    cols = ['PositionX', 'PositionY', 'PositionZ', 'RotationX', 'RotationY', 'RotationZ', 'RotationW']
    floats = df[cols].values.flatten()
    return floats

# Existing function to load the gesture dataset from the provided path.
def get_dataset_from_path(dataset_path):
    """
    Returns a dictionary where the keys are gesture names and each value is another dictionary
    with hand ('Left', 'Right') keys mapping to lists of gesture data (each a 1D array of 147 floats).
    """
    dataset = {}
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                # Assuming the folder structure: ...\<gesture_name>\<gesture_hand>\file.csv
                gesture_name = file_path.split("\\")[-3]
                gesture_hand = file_path.split("\\")[-2]
                df = pd.read_csv(file_path)
                gesture_data = extract_floats(df)
                if gesture_name not in dataset:
                    dataset[gesture_name] = {}
                if gesture_hand not in dataset[gesture_name]:
                    dataset[gesture_name][gesture_hand] = []
                dataset[gesture_name][gesture_hand].append(gesture_data)
    return dataset

# Custom Dataset class for gesture data.
class GestureDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Loads the gesture dataset from a given path and maps gesture names to integer labels.
        
        Args:
            dataset_path (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.labels = []
        # Load the raw dataset using the provided loader.
        raw_dataset = get_dataset_from_path(dataset_path)
        # Create a mapping from gesture names to integer indices.
        self.gesture_to_idx = {gesture: idx for idx, gesture in enumerate(sorted(raw_dataset.keys()))}
        
        # Flatten the nested dictionary into two parallel lists: one for data and one for labels.
        for gesture_name, hand_dict in raw_dataset.items():
            for hand, gesture_list in hand_dict.items():
                for gesture_data in gesture_list:
                    self.data.append(gesture_data)
                    self.labels.append(self.gesture_to_idx[gesture_name])
                    
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample and label at the specified index.
        sample = self.data[idx]
        label = self.labels[idx]
        # Convert the sample and label to PyTorch tensors.
        sample = torch.tensor(sample, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        # Apply any optional transformations.
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Utility function to create a DataLoader for the gesture dataset.
def get_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=0, transform=None):
    dataset = GestureDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# Example usage:
if __name__ == "__main__":
    dataset_path = r"dataset"
    dataloader = get_dataloader(dataset_path, batch_size=16, shuffle=True)
    
    # Iterate over the DataLoader
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}")
