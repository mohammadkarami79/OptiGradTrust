# Federated Learning with Dual Attention for Malicious Client Detection

A federated learning system that combines adaptive aggregation methods with a dual attention mechanism to detect and mitigate malicious client attacks.

## Overview

This project implements a federated learning system designed to operate in non-IID data environments while being robust against various attacks from malicious clients. The system leverages a dual attention mechanism to analyze gradient features and assign trust scores to clients, which are then used to weight their contributions during aggregation.

### Key Features

- **Malicious client detection** using a dual attention model trained on gradient features
- **Multiple aggregation methods** to handle non-IID data (FedAvg, FedBN, FedProx, FedADMM)
- **Gradient feature extraction** with 6 key features including VAE reconstruction error and Shapley values
- **Attack resistance** against various attacks (scaling, sign flipping, noise addition)
- **Trust-based weighting** for gradient aggregation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-learning-dual-attention.git
cd federated-learning-dual-attention

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Running Tests

```bash
# Run the dimension test to check feature compatibility
python test_dimension_fix.py

# Run a simple federated learning test
python test_federated_learning.py

# Run all tests
python run_all_tests.py
```

### Configuration

The system can be configured by modifying `federated_learning/config/config.py`:

```python
# Dataset configuration
DATASET = 'mnist'  # Options: 'mnist', 'cifar10', 'alzheimer'
ROOT_DATASET_RATIO = 0.1  # Percentage of dataset to use as root dataset

# Training parameters
BATCH_SIZE = 64
LOCAL_EPOCHS = 1
GLOBAL_EPOCHS = 10
LOCAL_EPOCHS_ROOT = 1
LR = 0.01
CLIENT_FRACTION = 1.0  # Fraction of clients to select each round

# Client configuration
NUM_CLIENTS = 5
NUM_MALICIOUS = 1
MALICIOUS_RATIO = 0.2

# Aggregation method
AGGREGATION_METHOD = 'fedbn'  # Options: 'fedavg', 'fedbn', 'fedprox', 'fedadmm'

# Dual attention parameters
DUAL_ATTENTION_FEATURE_DIM = 6
DUAL_ATTENTION_HIDDEN_DIM = 32
DUAL_ATTENTION_NUM_HEADS = 4
DUAL_ATTENTION_EPOCHS = 50
DUAL_ATTENTION_BATCH_SIZE = 16
DUAL_ATTENTION_LEARNING_RATE = 0.001

# Attack configuration
ATTACK_TYPE = 'partial_scaling_attack'  # Options: 'scaling_attack', 'sign_flipping_attack', 'noise_attack'

# VAE parameters
VAE_HIDDEN_DIM = 64
VAE_LATENT_DIM = 32
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 32
VAE_LEARNING_RATE = 0.001

# Feature extraction
ENABLE_SHAPLEY = True  # Whether to use Shapley values as features
```

### Running Your Own Experiments

Here's a simple example of how to use the system for your own experiment:

```python
from federated_learning.data.dataset import load_dataset
from federated_learning.training.server import Server
from federated_learning.training.client import Client
import torch
import numpy as np

# Load dataset
train_dataset, test_dataset, num_classes, _ = load_dataset()

# Split into root dataset and client datasets
root_size = int(0.1 * len(train_dataset))
root_indices = np.random.choice(len(train_dataset), size=root_size, replace=False)
client_indices = np.array([i for i in range(len(train_dataset)) if i not in root_indices])

root_dataset = torch.utils.data.Subset(train_dataset, root_indices)
root_loader = torch.utils.data.DataLoader(root_dataset, batch_size=64, shuffle=True)

# Create server
server = Server()
server.set_datasets(root_loader, test_dataset)

# Create clients
num_clients = 5
num_malicious = 1
clients = []

num_samples_per_client = len(client_indices) // num_clients
malicious_indices = np.random.choice(num_clients, size=num_malicious, replace=False)

for i in range(num_clients):
    start_idx = i * num_samples_per_client
    end_idx = (i+1) * num_samples_per_client if i < num_clients - 1 else len(client_indices)
    
    client_idx = client_indices[start_idx:end_idx]
    client_dataset = torch.utils.data.Subset(train_dataset, client_idx)
    
    is_malicious = i in malicious_indices
    clients.append(Client(client_id=i, dataset=client_dataset, is_malicious=is_malicious))

# Assign clients to server
server.clients = clients

# Pretrain global model on root dataset
server._pretrain_global_model()

# Collect root gradients for VAE training
root_gradients = server._collect_root_gradients()

# Train VAE on root gradients
server.vae = server.train_vae(root_gradients)

# Extract features from root gradients for dual attention training
honest_features = []
for grad in root_gradients:
    features = server._compute_gradient_features(grad, root_gradient=root_gradients[0])
    honest_features.append(features)

honest_features = torch.stack(honest_features)

# Generate malicious features
malicious_features = server._generate_malicious_features(honest_features)

# Train dual attention model
from federated_learning.training.training_utils import train_dual_attention
all_features = torch.cat([honest_features, malicious_features])
all_labels = torch.cat([
    torch.ones(honest_features.size(0), device=server.device),
    torch.zeros(malicious_features.size(0), device=server.device)
])

server.dual_attention = train_dual_attention(
    gradient_features=all_features,
    labels=all_labels,
    epochs=50,
    batch_size=16,
    lr=0.001
)

# Run federated learning
test_errors, metrics = server.train(num_rounds=10)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(test_errors)
plt.xlabel('Round')
plt.ylabel('Test Error')
plt.title('Federated Learning Progress')
plt.savefig('training_progress.png')
```

## System Architecture

The system follows this general workflow:

1. **Root Dataset Training**: Train global model initially on trusted root dataset
2. **VAE Training**: Train VAE on trusted gradients from root training
3. **Dual Attention Training**: Train dual attention model on both:
   - Trusted features from root gradients
   - Malicious features from simulated attacks
4. **Federated Learning Flow**:
   - Each client trains on local data and computes gradient
   - Server extracts features from each gradient
   - Dual attention model assigns trust scores to each client
   - Gradients are aggregated using trust-weighted approach with selected method
   - Global model is updated with aggregated gradient

## Gradient Features

For each client gradient, we extract 6 key features:

1. **Reconstruction Error**: How well the gradient can be reconstructed by the VAE trained on trusted gradients
2. **Root Similarity**: Cosine similarity with root gradients
3. **Client Similarity**: Average cosine similarity with other client gradients
4. **Gradient Norm**: Normalized norm of the gradient
5. **Pattern Consistency**: Consistency of the gradient pattern with root gradients
6. **Shapley Value**: Contribution of the gradient to the global model (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Implementation inspired by various federated learning papers
- Dual attention mechanism based on transformer architectures
- VAE implementation adapted from PyTorch examples 