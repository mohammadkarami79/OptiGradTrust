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

# Dual Attention training parameters
MALICIOUS_EPOCHS = 30 # Number of epochs to collect malicious gradients for Dual Attention training
BENIGN_DA_EPOCHS = 20 # Number of epochs to collect benign gradients for Dual Attention training

# Attack configuration
ATTACK_TYPE = 'partial_scaling_attack'  # Options: 'none', 'label_flipping', 'scaling_attack', 'partial_scaling_attack', 'backdoor_attack', 'adaptive_attack', 'min_max_attack', 'min_sum_attack', 'alternating_attack', 'targeted_attack', 'gradient_inversion_attack'

# Dataset configuration
DATASET = 'MNIST'

# Data Distribution Configuration
# Available options for DATA_DISTRIBUTION:
#   'iid'        - Independent and Identically Distributed. Data is split equally and randomly among clients.
#                  Each client gets similar class distribution.
#
#   'label_skew' - Non-IID with Label Skew. Clients have data biased towards certain classes.
#                  The parameter Q controls how much data of a particular class goes to clients in its group.
#                  Higher Q means more concentration of class data with preferred clients.
#
#   'dirichlet'  - Non-IID with Dirichlet Distribution. Creates random heterogeneity among clients.
#                  Controlled by DIRICHLET_ALPHA parameter:
#                  - Low alpha (e.g., 0.1): Highly skewed, clients may have samples from only a few classes
#                  - Medium alpha (e.g., 0.5-1.0): Moderate heterogeneity
#                  - High alpha (e.g., 10+): More balanced, approaching IID distribution
DATA_DISTRIBUTION = 'label_skew'  # Choose from: 'iid', 'label_skew', 'dirichlet'

# Distribution parameters
Q = 0.5  # Label skew concentration: higher values (closer to 1) create more bias towards specific classes
DIRICHLET_ALPHA = 0.5  # Dirichlet concentration: lower values create more heterogeneity

# Root dataset configuration
ROOT_DATASET_SIZE = 6000
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1 