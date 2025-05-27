import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNNMnist, self).__init__()
        # Improved CNN architecture with batch normalization
        # Now supports configurable input channels (1 for MNIST, 3 for CIFAR10/ALZHEIMER)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout(0.5)
        # Calculate flattened size after convolutions
        self.flatten_size = 3 * 3 * 64
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Better initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten dynamically - calculate size based on actual tensor
        x = x.view(x.size(0), -1)
        
        # Adjust first linear layer if needed
        if not hasattr(self, 'fc1_adjusted'):
            # Calculate actual flattened size and adjust linear layer
            actual_size = x.size(1)
            if actual_size != self.flatten_size:
                self.fc1 = nn.Linear(actual_size, 128)
                if x.is_cuda:
                    self.fc1 = self.fc1.cuda()
                # Re-initialize weights
                nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.fc1.bias, 0)
            self.fc1_adjusted = True
        
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 