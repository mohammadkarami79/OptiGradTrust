"""
Configuration for OASIS-1 Dataset Experiments

This configuration is optimized for the OASIS-1 Cross-Sectional MRI dataset
for Alzheimer's disease classification experiments.

Dataset: OASIS-1 (Open Access Series of Imaging Studies)
- 416 subjects aged 18-96
- 100 subjects over 60 diagnosed with AD
- 4 classes based on CDR (Clinical Dementia Rating)
"""

import torch
import os

# ======================================
# HARDWARE CONFIGURATION
# ======================================

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')

DEVICE = device
VAE_DEVICE = 'cpu'  # Save GPU memory
VERBOSE = True

# ======================================
# DATASET CONFIGURATION
# ======================================

DATASET = 'OASIS'
MODEL = 'ResNet18'  # Best for medical imaging

# OASIS paths - adjust based on your setup
OASIS_DATA_ROOT = 'oasis_cross-sectional_disc1/disc1'
OASIS_DEMOGRAPHICS_CSV = None  # Set path if available

# OASIS specific settings
OASIS_IMG_SIZE = 224
OASIS_CLASSES = 4
NUM_CLASSES = OASIS_CLASSES
INPUT_CHANNELS = 3

# Class names for OASIS
OASIS_CLASS_NAMES = [
    'Nondemented',
    'VeryMildDementia', 
    'MildDementia',
    'ModerateDementia'
]

# ======================================
# FEDERATED LEARNING PARAMETERS
# ======================================

NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)

# Training parameters - optimized for OASIS
BATCH_SIZE = 16  # Smaller batch for medical images
LR = 0.001  # Conservative learning rate
LOCAL_EPOCHS_ROOT = 10
LOCAL_EPOCHS_CLIENT = 5
GLOBAL_EPOCHS = 25
CLIENT_SELECTION_RATIO = 1.0
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Learning rate scheduling
LR_DECAY = 0.95
LR_DECAY_EPOCHS = 5
MIN_LR = 1e-5

# ======================================
# DATA DISTRIBUTION
# ======================================

ENABLE_NON_IID = False  # Start with IID
DATA_DISTRIBUTION = 'iid'
DIRICHLET_ALPHA = 0.5
NON_IID_TYPE = None

# Root dataset
ROOT_DATASET_RATIO = 0.15
ROOT_DATASET_SIZE = None  # Will be calculated
ROOT_DATASET_DYNAMIC_SIZE = True
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 0

# ======================================
# MODEL CONFIGURATION
# ======================================

# ResNet settings
RESNET18_UNFREEZE_LAYERS = 8  # More layers for medical imaging
RESNET50_UNFREEZE_LAYERS = 20
RESNET_PRETRAINED = True

# ======================================
# AGGREGATION METHOD
# ======================================

GRADIENT_COMBINATION_METHOD = 'fedbn_fedprox'
AGGREGATION_METHOD = GRADIENT_COMBINATION_METHOD
FEDPROX_MU = 0.1

# ======================================
# TRUST MECHANISM
# ======================================

ENABLE_DUAL_ATTENTION = True
DUAL_ATTENTION_HIDDEN_SIZE = 200
DUAL_ATTENTION_HEADS = 8
DUAL_ATTENTION_LAYERS = 3
DUAL_ATTENTION_EPOCHS = 10

# VAE for anomaly detection
ENABLE_VAE = True
VAE_EPOCHS = 15
VAE_BATCH_SIZE = 16
VAE_LEARNING_RATE = 0.0005
VAE_PROJECTION_DIM = 64
VAE_HIDDEN_DIM = 32
VAE_LATENT_DIM = 16

# Shapley values
ENABLE_SHAPLEY = True
SHAPLEY_SAMPLES = 25
SHAPLEY_WEIGHT = 0.4
VALIDATION_RATIO = 0.15
SHAPLEY_BATCH_SIZE = 16

# Detection thresholds
MALICIOUS_WEIGHTING_METHOD = 'continuous'
MALICIOUS_PENALTY_FACTOR = 0.4
MALICIOUS_THRESHOLD = 0.55
CONFIDENCE_THRESHOLD = 0.7
DUAL_ATTENTION_THRESHOLD = 0.65

# ======================================
# RL AGGREGATION
# ======================================

RL_AGGREGATION_METHOD = 'hybrid'
RL_ACTOR_HIDDEN_DIMS = [64, 32]
RL_CRITIC_HIDDEN_DIMS = [64, 32]
RL_LEARNING_RATE = 0.001
RL_INITIAL_TEMP = 5.0
RL_MIN_TEMP = 0.5
RL_WARMUP_ROUNDS = 5
RL_RAMP_UP_ROUNDS = 10
RL_SKIP_PRETRAINING = True
RL_GAMMA = 0.99
RL_ENTROPY_COEF = 0.01

# ======================================
# ATTACK CONFIGURATION
# ======================================

ENABLE_ATTACK_SIMULATION = True
SCALING_FACTOR = 10.0
PARTIAL_SCALING_PERCENT = 0.5
NOISE_FACTOR = 5.0
FLIP_PROBABILITY = 0.5
TARGETED_CLASS = 0  # Target nondemented class

# Gradient clipping
MAX_GRADIENT_NORM = 10.0
GRADIENT_CHUNK_SIZE = 10000

# ======================================
# DIMENSION REDUCTION
# ======================================

ENABLE_DIMENSION_REDUCTION = True
DIMENSION_REDUCTION_RATIO = 0.15

# ======================================
# OUTPUT AND LOGGING
# ======================================

SAVE_MODEL = True
MODEL_SAVE_PATH = 'model_weights/'
RANDOM_SEED = 42

# Results directory
RESULTS_DIR = 'results/oasis_experiments/'
PLOTS_DIR = 'research_plots/oasis_results/'

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ======================================
# EXPERIMENT-SPECIFIC SETTINGS
# ======================================

# Multi-seed experiments (Reviewer requirement: minimum 5 seeds)
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]

# Scalability experiments (Reviewer requirement: 50+ clients)
SCALABILITY_CLIENT_COUNTS = [10, 25, 50, 100]

# Attack types to test
ATTACK_TYPES = [
    'scaling_attack',
    'partial_scaling_attack',
    'sign_flipping_attack',
    'noise_attack',
    'label_flipping'
]

# Non-IID configurations
NON_IID_CONFIGS = {
    'dirichlet_0.5': {'type': 'dirichlet', 'alpha': 0.5},
    'dirichlet_0.1': {'type': 'dirichlet', 'alpha': 0.1},
    'label_skew_70': {'type': 'label_skew', 'ratio': 0.7},
    'label_skew_90': {'type': 'label_skew', 'ratio': 0.9},
}

# Class imbalance experiments (Reviewer requirement: <5% minority)
CLASS_IMBALANCE_RATIOS = [0.10, 0.05, 0.03, 0.01]

print("\n=== OASIS Experiment Configuration Loaded ===")
print(f"Dataset: {DATASET}")
print(f"Model: {MODEL}")
print(f"Clients: {NUM_CLIENTS} ({int(FRACTION_MALICIOUS*100)}% malicious)")
print(f"Device: {DEVICE}")
