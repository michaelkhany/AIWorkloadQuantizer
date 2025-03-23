#!/usr/bin/env python3
"""
A single-file MNIST recognizer script.
- Downloads/install dependencies (if needed)
- Downloads MNIST data
- Loads a pretrained model if available, otherwise trains one epoch and saves it
- Recognizes 100 random MNIST digits from the test set and prints the results.
"""

import os
import sys
import random

# --- Dependency check ---
# The script requires torch and torchvision.
# If these modules are not found, the script will attempt to install them.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
except ImportError:
    print("Required packages not found. Installing torch and torchvision...")
    try:
        # Use subprocess to call pip for installing required packages.
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    except Exception as e:
        print("Error installing dependencies:", e)
        sys.exit(1)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

# --- Neural Network Definition ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Two convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# --- Training Function ---
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# --- Testing Function ---
def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.0f}%)\n")

# --- Main Function ---
def main():
    # Use GPU if available, otherwise CPU.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # Set random seed for reproducibility.
    torch.manual_seed(1)

    # Define a transformation to normalize the data.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST dataset.
    data_path = "./data"
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize the model.
    model = Net().to(device)
    model_path = "mnist_cnn.pt"

    # Load pretrained model if available; otherwise, train and save the model.
    if os.path.exists(model_path):
        print("Loading pretrained model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Pretrained model not found. Training model (this may take a few minutes)...")
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        num_epochs = 1  # For demonstration, we train for 1 epoch. Increase for higher accuracy.
        for epoch in range(1, num_epochs + 1):
            train_model(model, device, train_loader, optimizer, epoch)
            test_model(model, device, test_loader)
        torch.save(model.state_dict(), model_path)
        print("Model saved to", model_path)

    # --- Recognize 100 MNIST characters ---
    print("Recognizing 100 random MNIST characters from the test set:")
    # Select 100 random indices from the test dataset.
    indices = random.sample(range(len(test_dataset)), 100)
    subset = Subset(test_dataset, indices)
    subset_loader = DataLoader(subset, batch_size=1, shuffle=False)

    model.eval()
    results = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(subset_loader):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1).item()
            results.append((idx, pred, target.item()))

    # Print the predictions.
    for idx, pred, actual in results:
        print(f"Image {idx}: Predicted: {pred}, Actual: {actual}")

if __name__ == '__main__':
    main()
