import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global parameters
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.2
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)
BATCH_SIZE = 32
LR = 0.01
GLOBAL_EPOCHS = 100
LOCAL_EPOCHS_ROOT = 200
LOCAL_EPOCHS_CLIENT = 10
MALICIOUS_EPOCHS = 40

# Attack configuration
ATTACK_TYPE = 'partial_scaling_attack'  # Options: 'none', 'label_flipping', 'scaling_attack', 'partial_scaling_attack', 'backdoor_attack', 'adaptive_attack'

# Dataset configuration
DATASET = 'MNIST'
NON_IID = True
Q = 0.5
ROOT_DATASET_SIZE = 6000
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1 