import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from federated_learning.config.config import *

def load_cifar10_dataset():
    """
    Load the CIFAR-10 dataset
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_classes: Number of classes
        input_channels: Number of input channels
    """
    # Define transforms with data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Simpler transforms for testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    try:
        # Create directory if it doesn't exist
        os.makedirs(CIFAR_DATA_ROOT, exist_ok=True)
        
        # Load training data
        train_dataset = datasets.CIFAR10(
            root=CIFAR_DATA_ROOT,
            train=True,
            download=True,
            transform=train_transform
        )
        
        # Load test data
        test_dataset = datasets.CIFAR10(
            root=CIFAR_DATA_ROOT,
            train=False,
            download=True,
            transform=test_transform
        )
        
        num_classes = 10  # CIFAR-10 has 10 classes
        input_channels = 3  # RGB images
        
        print(f"CIFAR-10 dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        print(f"Classes: {train_dataset.classes}")
        
        return train_dataset, test_dataset, num_classes, input_channels
    
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {str(e)}")
        raise 