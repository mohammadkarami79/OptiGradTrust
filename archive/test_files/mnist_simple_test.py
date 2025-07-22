import torch 
import torch.nn as nn 
import torchvision 
print("Starting MNIST validation test...") 
# Create simple model 
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)) 
print("Model created successfully") 
