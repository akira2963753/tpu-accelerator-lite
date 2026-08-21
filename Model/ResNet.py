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

# Basic block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.dropout(out)
        out = self.fc(out)

        return out

# ResNet-18
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Data
def get_data_loaders(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
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
    print("ResNet-18 Architecture Summary:")
    print("=" * 60)

    conv_layers = 0
    fc_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers += 1
        elif isinstance(module, nn.Linear):
            fc_layers += 1

    print(f"Total Convolutional Layers: {conv_layers}")
    print(f"Total Fully Connected Layers: {fc_layers}")
    print(f"Total Trainable Layers: {conv_layers + fc_layers}")

    dummy_input = torch.randn(input_size)

    print(f"\nInput shape: {dummy_input.shape}")

    shapes = {}
    def hook_fn(name):
        def hook(module, input, output):
            shapes[name] = output.shape
        return hook

    model.conv1.register_forward_hook(hook_fn('conv1'))
    model.layer1.register_forward_hook(hook_fn('layer1'))
    model.layer2.register_forward_hook(hook_fn('layer2'))
    model.layer3.register_forward_hook(hook_fn('layer3'))
    model.layer4.register_forward_hook(hook_fn('layer4'))
    model.avgpool.register_forward_hook(hook_fn('avgpool'))
    model.fc.register_forward_hook(hook_fn('fc'))

    with torch.no_grad():
        _ = model(dummy_input)

    for layer_name, shape in shapes.items():
        print(f"After {layer_name}: {shape}")

    print("=" * 60)

# Main
def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 15

    train_loader, test_loader = get_data_loaders(batch_size)

    model = ResNet18().to(device)
    print(f"Model architecture:\n{model}")

    print_model_summary(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        scheduler.step()

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

    torch.save(model.state_dict(), 'mnist_resnet18_model.pth')
    print("Model saved as 'mnist_resnet18_model.pth'")

    return model

# Plotting
def plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('ResNet-18 Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('ResNet-18 Model Accuracy')
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
    model.layer1[0].conv1.register_forward_hook(hook_fn('layer1_block1_conv1'))
    model.layer2[0].conv1.register_forward_hook(hook_fn('layer2_block1_conv1'))
    model.layer3[0].conv1.register_forward_hook(hook_fn('layer3_block1_conv1'))

    with torch.no_grad():
        _ = model(sample_image)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(sample_image.cpu().squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    layer_names = ['conv1', 'layer1_block1_conv1', 'layer2_block1_conv1', 'layer3_block1_conv1']
    titles = ['Conv1 (64 ch)', 'Layer1 (64 ch)', 'Layer2 (128 ch)', 'Layer3 (256 ch)']

    for i, (layer_name, title) in enumerate(zip(layer_names, titles)):
        if layer_name in feature_maps:
            features = feature_maps[layer_name][0]
            plt.subplot(1, 5, i + 2)
            plt.imshow(features[0], cmap='gray')
            plt.title(title)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    if 'conv1' in feature_maps:
        conv1_features = feature_maps['conv1'][0]
        plt.figure(figsize=(12, 8))
        for i in range(min(8, conv1_features.size(0))):
            plt.subplot(2, 4, i + 1)
            plt.imshow(conv1_features[i], cmap='gray')
            plt.title(f'Conv1 Channel {i+1}')
            plt.axis('off')
        plt.suptitle('Conv1 Feature Maps (First 8 channels)')
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
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.max(probabilities, 1)[0]

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i].squeeze(), cmap='gray')
        plt.title(f'True: {sample_labels[i].item()}\nPred: {predicted[i].item()}\nConf: {confidence[i].item():.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model = main()

    print("\nTesting with sample images...")
    _, test_loader = get_data_loaders(batch_size=64)
    test_sample_predictions(trained_model, test_loader, device)
