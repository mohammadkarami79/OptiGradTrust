import torch
import torch.nn as nn

class GradientVAE(nn.Module):
    def __init__(self, input_size):
        super(GradientVAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc21 = nn.Linear(256, 64)   # Mean
        self.fc22 = nn.Linear(256, 64)   # Log variance
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, input_size)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar 