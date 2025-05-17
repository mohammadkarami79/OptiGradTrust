import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from federated_learning.config.config import *

class AlzheimerDataset(Dataset):
    """
    Dataset for Alzheimer's MRI scans
    """
    def __init__(self, root_dir, transform=None, class_map=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_map (dict, optional): Dictionary mapping class folder names to class indices.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_map = class_map or {}
        
        self.image_paths = []
        self.labels = []
        
        # Scan directory for class folders
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # If class_map is not provided, create it
        if not class_map:
            self.class_map = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect image paths and labels
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = self.class_map[cls_name]
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_classes(self):
        return self.classes
    
    def get_class_map(self):
        return self.class_map


def load_alzheimer_dataset():
    """
    Load the Alzheimer's MRI dataset
    
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_classes: Number of classes
        input_channels: Number of input channels
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((ALZHEIMER_IMG_SIZE, ALZHEIMER_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((ALZHEIMER_IMG_SIZE, ALZHEIMER_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load datasets
    train_dir = os.path.join(ALZHEIMER_DATA_DIR, 'train')
    test_dir = os.path.join(ALZHEIMER_DATA_DIR, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(f"Alzheimer's dataset not found in {ALZHEIMER_DATA_DIR}. " 
                               f"Please download and extract it first.")
    
    # Load training data
    train_dataset = AlzheimerDataset(train_dir, transform=train_transform)
    class_map = train_dataset.get_class_map()
    
    # Load test data with the same class mapping
    test_dataset = AlzheimerDataset(test_dir, transform=test_transform, class_map=class_map)
    
    num_classes = len(train_dataset.get_classes())
    input_channels = 3  # RGB images
    
    print(f"Alzheimer's dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Classes: {train_dataset.get_classes()}")
    
    return train_dataset, test_dataset, num_classes, input_channels

def download_alzheimer_dataset():
    """
    Instructions for downloading the Alzheimer's MRI dataset from Kaggle
    """
    print("Alzheimer's MRI dataset needs to be downloaded manually from Kaggle.")
    print("Dataset URL: https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy")
    print("\nInstructions:")
    print("1. Download the dataset from Kaggle")
    print("2. Extract it to your configured ALZHEIMER_DATA_DIR directory")
    print(f"3. Ensure the directory structure follows: {ALZHEIMER_DATA_DIR}/train and {ALZHEIMER_DATA_DIR}/test")
    print("4. Each directory should contain class folders (MildDemented, ModerateDemented, NonDemented, VeryMildDemented)")
    print("\nOnce downloaded, you can use the load_alzheimer_dataset() function to load the data.") 