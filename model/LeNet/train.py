import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

np.random.seed(42)
torch.manual_seed(42)

# Model
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_input_size = 16 * 5 * 5

        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

# Data
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 200 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

# Testing
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total

    return test_loss, test_acc

# Architecture
def print_model_summary(model, input_size=(1, 1, 28, 28)):
    print("LeNet Architecture Summary:")
    print("=" * 50)

    dummy_input = torch.randn(input_size)

    print(f"Input shape: {dummy_input.shape}")

    x = dummy_input

    x = model.pool1(F.relu(model.conv1(x)))
    print(f"After Conv1 + Pool1: {x.shape}")

    x = model.pool2(F.relu(model.conv2(x)))
    print(f"After Conv2 + Pool2: {x.shape}")

    x = x.view(x.size(0), -1)
    print(f"After Flatten: {x.shape}")

    x = F.relu(model.fc1(x))
    print(f"After FC1: {x.shape}")

    x = F.relu(model.fc2(x))
    print(f"After FC2: {x.shape}")

    x = model.fc3(x)
    print(f"After FC3 (Output): {x.shape}")

    print("=" * 50)

# Main
def main():
    batch_size = 64
    learning_rate = 0.000055
    num_epochs = 10

    train_loader, test_loader = get_data_loaders(batch_size)

    model = LeNet().to(device)
    print(f"Model architecture:\n{model}")

    print_model_summary(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    print(f'\nTraining completed!')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')

    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)

    visualize_feature_maps(model, test_loader, device)

    torch.save(model.state_dict(), 'mnist_lenet_model.pth')
    print("Model saved as 'mnist_lenet_model.pth'")

    return model

# Plotting
def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('LeNet Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('LeNet Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Feature maps
def visualize_feature_maps(model, test_loader, device):
    model.eval()

    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    sample_image = images[0:1].to(device)

    feature_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook

    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.conv2.register_forward_hook(hook_fn('conv2'))

    with torch.no_grad():
        _ = model(sample_image)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(sample_image.cpu().squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    conv1_features = feature_maps['conv1'][0]
    for i in range(min(3, conv1_features.size(0))):
        plt.subplot(1, 4, i + 2)
        plt.imshow(conv1_features[i], cmap='gray')
        plt.title(f'Conv1 Feature {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in range(min(6, conv1_features.size(0))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(conv1_features[i], cmap='gray')
        plt.title(f'Conv1 Channel {i+1}')
        plt.axis('off')
    plt.suptitle('Conv1 Feature Maps (6 channels)')
    plt.tight_layout()
    plt.show()

# Sample inference
def test_sample_predictions(model, test_loader, device, num_samples=10):
    model.eval()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    sample_images = images[:num_samples]
    sample_labels = labels[:num_samples]

    with torch.no_grad():
        sample_images_gpu = sample_images.to(device)
        outputs = model(sample_images_gpu)
        _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f'True: {sample_labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model = main()

    print("\nTesting with sample images...")
    _, test_loader = get_data_loaders(batch_size=64)
    test_sample_predictions(trained_model, test_loader, device)
