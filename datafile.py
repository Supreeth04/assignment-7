import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
    transforms.RandomAffine(
        degrees=0, 
        scale=(0.9, 1.1),        
        translate=(0.1, 0.2),
        shear=(-10, 10),),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load training data
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load test data
test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

# Create data loaders
train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)