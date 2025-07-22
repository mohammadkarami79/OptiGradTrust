"""
🎯 SIMPLE OPTIMIZED CONFIGURATION FOR CIFAR-10 ACCURACY TEST

This configuration targets 80%+ accuracy for CIFAR-10 + ResNet18
Specifically designed to fix the low accuracy issue (51% → 80%+)
"""

import torch
import os

# ======================================
# BASIC CONFIGURATION
# ======================================

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = device
FORCE_GPU = torch.cuda.is_available()

# ======================================
# DATASET AND MODEL - CIFAR-10 FOCUS
# ======================================

# Dataset configuration
DATASET = 'CIFAR10'
MODEL = 'ResNet18'
INPUT_CHANNELS = 3
NUM_CLASSES = 10

# ======================================
# IMPROVED TRAINING PARAMETERS
# ======================================

# Federated Learning Parameters
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)

# KEY IMPROVEMENTS - Much stronger training
GLOBAL_EPOCHS = 20                 # ✅ Increased from 3 to 20
LOCAL_EPOCHS_ROOT = 12             # ✅ Increased from 5 to 12  
LOCAL_EPOCHS_CLIENT = 4            # ✅ Increased from 3 to 4
BATCH_SIZE = 32                    # ✅ Increased from 16 to 32
LR = 0.01                          # ✅ Good learning rate
LEARNING_RATE = LR

# Other training parameters
CLIENT_SELECTION_RATIO = 1.0
CLIENT_FRACTION = 1.0
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CLIENT_EPOCHS = LOCAL_EPOCHS_CLIENT

# Learning rate scheduling
LR_DECAY = 0.98
LR_DECAY_EPOCHS = 3
MIN_LR = 0.0001

# Data loading
NUM_WORKERS = 0
PIN_MEMORY = False  # Keep False for stability

# ======================================
# MODEL SPECIFIC CONFIGURATIONS
# ======================================

# ResNet configuration
RESNET50_UNFREEZE_LAYERS = 20
RESNET18_UNFREEZE_LAYERS = 6
RESNET_PRETRAINED = True

# Dataset paths
ALZHEIMER_DATA_ROOT = 'data/alzheimer'
CIFAR_DATA_ROOT = 'data/cifar'
MNIST_DATA_ROOT = 'data/mnist'

# CIFAR-10 specific
CIFAR_IMG_SIZE = 32
CIFAR_CLASSES = 10

# MNIST specific
MNIST_IMG_SIZE = 28
MNIST_CLASSES = 10

# Alzheimer specific
ALZHEIMER_DATA_DIR = './data/alzheimer'
ALZHEIMER_IMG_SIZE = 224
ALZHEIMER_CLASSES = 4

# ======================================
# VAE CONFIGURATION - IMPROVED
# ======================================

VAE_EPOCHS = 15                    # ✅ Increased from 12
VAE_BATCH_SIZE = 12                # ✅ Doubled from 6
VAE_LEARNING_RATE = 0.0005
VAE_PROJECTION_DIM = 128           # ✅ Increased from 64
VAE_HIDDEN_DIM = 64                # ✅ Increased from 32
VAE_LATENT_DIM = 32                # ✅ Increased from 16
GRADIENT_DIMENSION = None
VAE_DEVICE = 'cpu'  # Keep CPU for memory safety

# ======================================
# DATA DISTRIBUTION
# ======================================

# Data distribution - IID for baseline
ENABLE_NON_IID = False
DIRICHLET_ALPHA = None
LABEL_SKEW_RATIO = None
QUANTITY_SKEW_RATIO = None

# Root dataset - IMPROVED
ROOT_DATASET_RATIO = 0.18          # ✅ Increased from 0.15
ROOT_DATASET_SIZE = 4500           # ✅ Increased from 3500
ROOT_DATASET_DYNAMIC_SIZE = True
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1

# ======================================
# AGGREGATION METHOD
# ======================================

GRADIENT_COMBINATION_METHOD = 'fedbn_fedprox'
AGGREGATION_METHOD = GRADIENT_COMBINATION_METHOD

# FedProx parameters
FEDPROX_MU = 0.1

# FedADMM parameters
FEDADMM_RHO = 1.0
FEDADMM_SIGMA = 0.1
FEDADMM_ITERATIONS = 3

# FedDWA parameters
FEDDWA_WEIGHTING = 'accuracy'
FEDDWA_HISTORY_FACTOR = 0.2

# FedNova parameters
FEDNOVA_NORMALIZE_UPDATES = True

# ======================================
# DETECTION PARAMETERS - IMPROVED
# ======================================

# Dual Attention
ENABLE_DUAL_ATTENTION = True
DUAL_ATTENTION_HIDDEN_SIZE = 200   # ✅ Increased from 128
DUAL_ATTENTION_HEADS = 10          # ✅ Increased from 8
DUAL_ATTENTION_LAYERS = 3
DUAL_ATTENTION_EPOCHS = 8          # ✅ Increased from 5
DUAL_ATTENTION_BATCH_SIZE = 12     # ✅ Increased from 8
DUAL_ATTENTION_LEARNING_RATE = 0.0005

# VAE for detection
ENABLE_VAE = True

# Shapley values - ENHANCED
ENABLE_SHAPLEY = True
SHAPLEY_SAMPLES = 22               # ✅ Increased from 20
SHAPLEY_WEIGHT = 0.55              # ✅ Increased from 0.5
VALIDATION_RATIO = 0.15
SHAPLEY_BATCH_SIZE = 24            # ✅ Increased from 16

# Detection thresholds
MALICIOUS_WEIGHTING_METHOD = 'continuous'
MALICIOUS_PENALTY_FACTOR = 0.35
ZERO_ATTACK_THRESHOLD = 0.008
HIGH_GRADIENT_STD_MULTIPLIER = 2.5
CONFIDENCE_THRESHOLD = 0.68
DUAL_ATTENTION_THRESHOLD = 0.66

# ======================================
# RL AGGREGATION
# ======================================

RL_AGGREGATION_METHOD = 'hybrid'

# RL parameters
RL_ACTOR_HIDDEN_DIMS = [96, 48]
RL_CRITIC_HIDDEN_DIMS = [96, 48]
RL_LEARNING_RATE = 0.001
RL_EPSILON = 0.1
RL_MEMORY_SIZE = 1500
RL_BATCH_SIZE = 12
RL_UPDATE_FREQUENCY = 5
RL_GAMMA = 0.99
RL_ENTROPY_COEF = 0.01
RL_INITIAL_TEMP = 5.0
RL_MIN_TEMP = 0.5
RL_SKIP_PRETRAINING = True
RL_WARMUP_ROUNDS = 6
RL_RAMP_UP_ROUNDS = 12
RL_PRETRAINING_EPISODES = 100
RL_VALIDATION_MINIBATCH = 0.2
RL_SAVE_INTERVAL = 50

# ======================================
# ATTACK CONFIGURATION
# ======================================

ENABLE_ATTACK_SIMULATION = True

# Attack parameters - REALISTIC
SCALING_FACTOR = 3.5               # ✅ Balanced scaling
PARTIAL_SCALING_PERCENT = 0.35     # ✅ Balanced partial scaling
NOISE_FACTOR = 1.8                 # ✅ Realistic noise
FLIP_PROBABILITY = 0.55            # ✅ Realistic probability
TARGETED_CLASS = 7

# ======================================
# MEMORY AND OPTIMIZATION
# ======================================

# Memory management
ENABLE_MEMORY_TRACKING = True
AGGRESSIVE_MEMORY_CLEANUP = True
GRADIENT_CHUNK_SIZE = 80000
GRADIENT_AGGREGATION_METHOD = 'mean'
MAX_GRADIENT_NORM = 10.0

# Gradient optimization
ENABLE_DIMENSION_REDUCTION = True
DIMENSION_REDUCTION_RATIO = 0.12   # ✅ Less aggressive
ENABLE_PARALLEL_PROCESSING = False

# Gradient clipping
NORMALIZE_GRADIENTS = True
GRADIENT_CLIP_VALUE = 10.0

# ======================================
# MISCELLANEOUS SETTINGS
# ======================================

# Privacy
PRIVACY_MECHANISM = 'none'
DP_EPSILON = 2.0
DP_DELTA = 1e-5
DP_CLIP_NORM = 1.0

# Logging and saving
LOGGING_LEVEL = 'INFO'
SAVE_MODEL = True
MODEL_SAVE_PATH = 'model_weights/'
RANDOM_SEED = 42

# Communication rounds
COMMUNICATION_ROUNDS = GLOBAL_EPOCHS
CLIENT_LR = LR

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Detection refinements
GRADIENT_NORM_THRESHOLD_FACTOR = 2.2
GRADIENT_NORM_MIN_THRESHOLD = 0.16
SUSPICIOUS_CLIENT_PENALTY = 0.65
TRUST_SCORE_THRESHOLD = 0.58
ADAPTIVE_THRESHOLD = True
FALSE_POSITIVE_MITIGATION = True
MIN_CLIENTS_FOR_DETECTION = 2
GRADIENT_VARIANCE_THRESHOLD = 0.55
MALICIOUS_THRESHOLD = 0.68
DETECTION_SENSITIVITY = 0.72

# Additional settings
ATTACKER_IMPACT_WEIGHING = True
MALICIOUS_CLIENT_RATIO = 0.3
SCALING_FRACTION = 0.5

# Verbosity
VERBOSE = True

print("✅ SIMPLE OPTIMIZED CONFIG LOADED")
print(f"🎯 Target: 80%+ CIFAR-10 accuracy")
print(f"📊 Key settings: {GLOBAL_EPOCHS} epochs, batch {BATCH_SIZE}, LR {LR}") 