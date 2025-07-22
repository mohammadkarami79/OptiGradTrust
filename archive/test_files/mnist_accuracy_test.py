# MNIST Accuracy Test 
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
print("?? Testing MNIST CNN accuracy...") 
# Load MNIST dataset 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
