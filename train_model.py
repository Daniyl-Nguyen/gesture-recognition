import torch
import torch.nn as nn

from gesture_dataloader import get_dataloader

path = r"C:\Users\masly\gesture-tracking\dataset"

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size=147, hidden1_size=256, hidden2_size=128, num_classes=10):
        super(ThreeLayerMLP, self).__init__()
        # First fully connected layer: from input to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden1_size)
        # Second fully connected layer: from first hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        # Third fully connected layer: from second hidden layer to output layer
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Pass the input through the first layer and apply ReLU
        x = self.relu(self.fc1(x))
        # Pass the result through the second layer and apply ReLU
        x = self.relu(self.fc2(x))
        # Pass through the final layer (logits output)
        x = self.fc3(x)
        return x
    
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    """
    Trains the model and tests it every 25 epochs.
    
    Args:
        model (torch.nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        criterion (loss function): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
    """
    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Print loss and evaluate model every 25 epochs (or at the final epoch)
        if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
            test_model(model, test_loader)

def test_model(model, test_loader):
    """
    Evaluates the model on test data and prints the accuracy.
    
    Args:
        model (torch.nn.Module): The neural network to evaluate.
        test_loader (DataLoader): DataLoader for test data.
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
def main():
    # CrossEntropy Loss and Adam optimizer
    # def get_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=0, transform=None):
    batch_size = 32
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    dataloader = get_dataloader(path, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ThreeLayerMLP()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_model(model, dataloader)
    train_model(model, dataloader, dataloader, criterion, optimizer, num_epochs=num_epochs)
    test_model(model, dataloader)
    
if __name__ == "__main__":
    main()