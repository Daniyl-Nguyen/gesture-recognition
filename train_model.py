from typing import List

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from gesture_dataloader import (
    GestureDataset,
)  # Assumes your dataset class is defined here

path = r"dataset"

class ThreeLayerMLP(nn.Module):
    def __init__(
        self, input_size=147, hidden1_size=256, hidden2_size=128, num_classes=10
    ):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, data: List[int]):
        # Convert data to tensor
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        x = self.forward(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(x, dim=1)
        
        # Get the index of the highest probability
        classification_idx = torch.argmax(probs).item()
        
        # Get the confidence score
        confidence = probs[0, classification_idx].item()

        return classification_idx, confidence


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    """
    Trains the model and evaluates it every 25 epochs.
    """
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Print loss and test performance every 25 epochs (or on the final epoch)
        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            test_model(model, test_loader)


def test_model(model, test_loader):
    """
    Evaluates the model on test data and prints the accuracy.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


def create_train_test_loaders(dataset, batch_size=32, test_samples_per_class=2):
    """
    Splits the dataset by taking out `test_samples_per_class` samples per class as the test set.
    The remaining samples form the training set.
    """
    # Collect indices for each class
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset):
        label_val = label.item()
        indices_by_class.setdefault(label_val, []).append(idx)

    test_indices = []
    for cls, indices in indices_by_class.items():
        # Select test_samples_per_class indices randomly (or all available if fewer)
        if len(indices) >= test_samples_per_class:
            selected = np.random.choice(
                indices, size=test_samples_per_class, replace=False
            ).tolist()
        else:
            selected = indices
        test_indices.extend(selected)

    # Training indices: those not in the test set
    all_indices = set(range(len(dataset)))
    train_indices = list(all_indices - set(test_indices))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


import os

def get_trained_models():
    """
    Trains or loads pre-trained models for left and right hand gestures,
    incorporating data from two dataset paths if training is needed.
    """
    path1 = r"dataset"
    path2 = r"dataset2" # Define the second dataset path
    batch_size = 32
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()

    model_left_path = "models/model_left.pth"
    model_right_path = "models/model_right.pth"

    # Create model directory if not exists
    os.makedirs("models", exist_ok=True)

    # --- Left Hand ---
    model_left = ThreeLayerMLP()
    if os.path.exists(model_left_path):
        print("Loading pre-trained left hand model...")
        model_left.load_state_dict(torch.load(model_left_path))
    else:
        print("Training model for left hand gestures using data from:", path1, "and", path2)
        # Create datasets for both paths
        left_hand_dataset1 = GestureDataset(path1, hand="left")
        left_hand_dataset2 = GestureDataset(path2, hand="left")
        # Combine datasets
        combined_left_dataset = ConcatDataset([left_hand_dataset1, left_hand_dataset2])

        train_loader_left, test_loader_left = create_train_test_loaders(
            combined_left_dataset, batch_size=batch_size, test_samples_per_class=5
        )
        optimizer_left = torch.optim.Adam(model_left.parameters(), lr=0.01)
        train_model(model_left, train_loader_left, test_loader_left, criterion, optimizer_left, num_epochs=num_epochs)
        torch.save(model_left.state_dict(), model_left_path)
        print("Left hand model trained and saved.")

    # --- Right Hand ---
    model_right = ThreeLayerMLP()
    if os.path.exists(model_right_path):
        print("Loading pre-trained right hand model...")
        model_right.load_state_dict(torch.load(model_right_path))
    else:
        print("Training model for right hand gestures using data from:", path1, "and", path2)
        # Create datasets for both paths
        right_hand_dataset1 = GestureDataset(path1, hand="right")
        right_hand_dataset2 = GestureDataset(path2, hand="right")
        # Combine datasets
        combined_right_dataset = ConcatDataset([right_hand_dataset1, right_hand_dataset2])

        train_loader_right, test_loader_right = create_train_test_loaders(
            combined_right_dataset, batch_size=batch_size, test_samples_per_class=5
        )
        optimizer_right = torch.optim.Adam(model_right.parameters(), lr=0.01)
        train_model(model_right, train_loader_right, test_loader_right, criterion, optimizer_right, num_epochs=num_epochs)
        torch.save(model_right.state_dict(), model_right_path)
        print("Right hand model trained and saved.")

    return model_left, model_right



def main():
    batch_size = 32
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()

    # Create the complete dataset using your custom GestureDataset
    dataset = GestureDataset(path)

    # Create train/test loaders by taking 2 samples per class for testing (approx. 20% per class)
    train_loader, test_loader = create_train_test_loaders(
        dataset, batch_size=batch_size, test_samples_per_class=5
    )

    model = ThreeLayerMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Initial test performance:")
    test_model(model, test_loader)

    train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs
    )

    print("Final test performance:")
    test_model(model, test_loader)


if __name__ == "__main__":
    main()
