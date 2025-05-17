import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings

class ResNet18Alzheimer(nn.Module):
    def __init__(self, num_classes=4, unfreeze_layers=10, pretrained=True):
        """
        ResNet18 model for Alzheimer's dataset classification.
        Lighter and faster than ResNet50, with fewer parameters.
        
        Args:
            num_classes: Number of output classes (default: 4)
            unfreeze_layers: Number of layers to unfreeze from the end (default: 10)
            pretrained: Whether to use pretrained weights (default: True)
        """
        super(ResNet18Alzheimer, self).__init__()
        
        # Load ResNet18 with error handling for pretrained weights
        try:
            if pretrained:
                print("Loading ResNet18 with pretrained weights...")
                # Updated for newer PyTorch versions
                self.resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
                print("Successfully loaded pretrained weights.")
            else:
                print("Initializing ResNet18 with random weights...")
                self.resnet = models.resnet18(weights=None)
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights: {str(e)}. Using random initialization instead.")
            print("Falling back to random initialization for ResNet18...")
            self.resnet = models.resnet18(weights=None)
            
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze specified number of layers from the end
        trainable_layers = list(self.resnet.children())[-unfreeze_layers:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)

class ResNet50Alzheimer(nn.Module):
    def __init__(self, num_classes=4, unfreeze_layers=20, pretrained=True):
        """
        ResNet50 model for Alzheimer's dataset classification.
        
        Args:
            num_classes: Number of output classes (default: 4)
            unfreeze_layers: Number of layers to unfreeze from the end (default: 20)
            pretrained: Whether to use pretrained weights (default: True)
        """
        super(ResNet50Alzheimer, self).__init__()
        
        # Load ResNet50 with error handling for pretrained weights
        try:
            if pretrained:
                print("Loading ResNet50 with pretrained weights...")
                # Updated for newer PyTorch versions
                self.resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
                print("Successfully loaded pretrained weights.")
            else:
                print("Initializing ResNet50 with random weights...")
                self.resnet = models.resnet50(weights=None)
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights: {str(e)}. Using random initialization instead.")
            print("Falling back to random initialization for ResNet50...")
            self.resnet = models.resnet50(weights=None)
            
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze specified number of layers from the end
        trainable_layers = list(self.resnet.children())[-unfreeze_layers:]
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=1) 