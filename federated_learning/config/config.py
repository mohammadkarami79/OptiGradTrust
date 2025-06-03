import torch
import os
import torch.nn.functional as F

# ======================================
# HARDWARE AND MEMORY CONFIGURATION
# ======================================

# GPU Configuration
FORCE_GPU = True  # Set to True to force GPU usage, if available

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)  # Explicitly set to first GPU
else:
    device = torch.device('cpu')

# Print GPU info once at startup
# Check if we've already printed GPU info to avoid repeating it
if not hasattr(torch, '_gpu_info_printed'):
    torch._gpu_info_printed = True
    print("\n=== GPU Configuration ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"\nGPU Memory:")
        print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    else:
        print("No GPU available. Using CPU.")

# Memory Management Configuration
# Controls whether VAE runs on GPU or CPU
# 'auto': Use GPU if available and main model is on CPU
# 'gpu': Force VAE to run on GPU if CUDA is available
# 'cpu': Force VAE to run on CPU regardless of GPU availability
VAE_DEVICE = 'cpu'  # Changed to CPU to save GPU memory

# Additional memory optimization parameters
ENABLE_MEMORY_TRACKING = True     # Enable to monitor memory usage
AGGRESSIVE_MEMORY_CLEANUP = True  # Enables aggressive memory cleanup

# Gradient Memory Optimization
GRADIENT_CHUNK_SIZE = 500000   # Size of chunks for gradient processing
GRADIENT_AGGREGATION_METHOD = 'mean'
MAX_GRADIENT_NORM = 10.0  # Increased from 5.0 to better detect scaling attacks

# Verbosity settings
VERBOSE = True  # Set to True for detailed debug information, particularly for BatchNorm layer tracking

# Gradient Dimension Reduction for Memory Optimization
ENABLE_DIMENSION_REDUCTION = True     # Enable dimension reduction
DIMENSION_REDUCTION_RATIO = 0.25      # More aggressive reduction (keep 25%)
ENABLE_PARALLEL_PROCESSING = False    # Keep parallel processing disabled

# ======================================
# FEDERATED LEARNING PARAMETERS
# ======================================

# Global parameters
NUM_CLIENTS = 5                    # Increased from 3 to 5 for better testing
FRACTION_MALICIOUS = 0.4           # Increased from 0.2 to 0.4 (40% malicious clients)
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)
BATCH_SIZE = 64                    # Batch size
LR = 0.01                          # Learning rate
LOCAL_EPOCHS_ROOT = 1              # Number of local epochs for root
LOCAL_EPOCHS_CLIENT = 2            # Increased from 1 to 2 for more significant updates
GLOBAL_EPOCHS = 5                  # Increased from 2 to 5 to show more improvement over rounds
CLIENT_SELECTION_RATIO = 1.0       # Fraction of clients to select each round
CLIENT_FRACTION = 1.0              # Fraction of clients to use in each round (selection pool)
LEARNING_RATE = LR                 # Alias for LR used in some methods
MOMENTUM = 0.9                     # Momentum for SGD
WEIGHT_DECAY = 5e-4                # Weight decay for regularization
CLIENT_EPOCHS = LOCAL_EPOCHS_CLIENT

# Learning rate scheduling
LR_DECAY = 0.98                    # Learning rate decay factor
LR_DECAY_EPOCHS = 1                # Apply decay every N epochs
MIN_LR = 0.001                     # Minimum learning rate

# Data loading configuration
NUM_WORKERS = 4                    # Disable worker threads for data loading
PIN_MEMORY = True                 # Disable pin memory to save GPU memory

# ======================================
# MODEL AND DATASET CONFIGURATION
# ======================================

# Dataset selection
DATASET = 'MNIST'                  # Options: 'MNIST', 'ALZHEIMER', 'CIFAR10'

# Model selection
MODEL = 'CNN'                      # Options: 'CNN', 'RESNET18', 'RESNET50'

# ResNet configuration
RESNET50_UNFREEZE_LAYERS = 20      # Number of layers to unfreeze from the end for ResNet50
RESNET18_UNFREEZE_LAYERS = 5       # Number of layers to unfreeze from the end for ResNet18
RESNET_PRETRAINED = True           # Whether to use pretrained weights

# Dataset paths
ALZHEIMER_DATA_ROOT = 'data/alzheimer'
CIFAR_DATA_ROOT = 'data/cifar'
MNIST_DATA_ROOT = 'data/mnist'

# Alzheimer's dataset configuration
ALZHEIMER_DATA_DIR = './data/alzheimer'
ALZHEIMER_IMG_SIZE = 224          # Size to resize images to
ALZHEIMER_CLASSES = 4             # Number of classes

# CIFAR-10 specific settings
CIFAR_IMG_SIZE = 32               # CIFAR-10 images are 32x32
CIFAR_CLASSES = 10                # 10 classes in CIFAR-10

# VAE training configuration
VAE_EPOCHS = 5                  # Number of epochs for VAE training
VAE_BATCH_SIZE = 32              # Batch size for VAE training
VAE_LEARNING_RATE = 0.001        # Learning rate for VAE training
VAE_PROJECTION_DIM = 128         # Much smaller projection dimension
VAE_HIDDEN_DIM = 64              # Hidden dimension for VAE
VAE_LATENT_DIM = 32              # Latent dimension for VAE
GRADIENT_DIMENSION = None        # Placeholder: Will be determined dynamically based on the model

# ======================================
# DATA DISTRIBUTION CONFIGURATION
# ======================================

DATA_DISTRIBUTION = 'label_skew'   # Choose from: 'iid', 'label_skew', 'dirichlet'

# Distribution parameters
Q = 0.5                           # Label skew concentration
DIRICHLET_ALPHA = 0.5             # Dirichlet concentration
NON_IID_LABEL_RATIO = 0.5         # Label ratio for non-IID label split

# Root dataset configuration
ROOT_DATASET_RATIO = 0.1          # Ratio of dataset to use as root dataset
ROOT_DATASET_SIZE = 1000          # Used only if ROOT_DATASET_DYNAMIC_SIZE is False
ROOT_DATASET_DYNAMIC_SIZE = True  # If True, use ROOT_DATASET_RATIO
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1

# ======================================
# GRADIENT COMBINATION METHOD CONFIGURATION
# ======================================

# Available gradient combination methods:
# - 'fedavg': Standard Federated Averaging (McMahan et al.)
# - 'fedprox': Adds proximal term to client optimization (Li et al.)
# - 'fedadmm': Alternating Direction Method of Multipliers for FL (Wang et al.)
# - 'fedbn': Keeps batch normalization parameters local to clients (Li et al.)
# - 'feddwa': Dynamic weighted aggregation based on client performance (Chai et al.)
# - 'fednova': Normalized averaging based on local optimization steps (Wang et al.)
# - 'fedbn_fedprox': Combination of FedBN and FedProx methods
GRADIENT_COMBINATION_METHOD = 'fedbn'     # Using our improved FedBN implementation

# For backward compatibility
AGGREGATION_METHOD = GRADIENT_COMBINATION_METHOD

# FedProx parameters
FEDPROX_MU = 0.1                 # μ coefficient for proximal term

# FedADMM parameters
FEDADMM_RHO = 1.0                # ρ coefficient for ADMM
FEDADMM_SIGMA = 0.1              # σ coefficient for dual update
FEDADMM_ITERATIONS = 3           # Number of iterations for ADMM

# FedDWA parameters
FEDDWA_WEIGHTING = 'accuracy'    # Options: 'accuracy', 'loss', 'gradient_norm'
FEDDWA_HISTORY_FACTOR = 0.2      # Weight for historical values (0-1)

# FedNova parameters
FEDNOVA_NORMALIZE_UPDATES = True # Normalize updates based on local steps

# ======================================
# MALICIOUS CLIENT DETECTION
# ======================================

# Dual Attention parameters
ENABLE_DUAL_ATTENTION = True     # Enable dual attention-based malicious client detection
DUAL_ATTENTION_HIDDEN_SIZE = 64  # Increased from 32 to 64 for better feature processing
DUAL_ATTENTION_HEADS = 4         # Number of attention heads
DUAL_ATTENTION_LAYERS = 2        # Number of transformer layers

# VAE parameters for anomaly detection
ENABLE_VAE = True                # Enable VAE-based anomaly detection

# Shapley value integration
ENABLE_SHAPLEY = True            # Enable Shapley value calculation
SHAPLEY_WEIGHT = 0.5             # Increased from 0.3 to 0.5 for stronger influence on trust scoring
SHAPLEY_NUM_SAMPLES = 5          # Number of samples for Monte Carlo approximation

# Malicious weighting method
# Options: 'binary', 'continuous', 'squared', 'sqrt'
MALICIOUS_WEIGHTING_METHOD = 'continuous'

# Malicious penalty factor (0-1)
# Higher values (closer to 1) give much lower weights to detected malicious clients
# Lower values are more lenient. 0 means no special penalty beyond trust score
MALICIOUS_PENALTY_FACTOR = 0.98  # Strong penalty for detected malicious clients

# Attacker impact weighing
ATTACKER_IMPACT_WEIGHING = True    # Enable attacker impact weighing


# ======================================
# RL-BASED AGGREGATION CONFIGURATION
# ======================================

# Aggregation method selection
# Options: 'dual_attention', 'rl_actor_critic', 'hybrid'
# - 'dual_attention': Use only the dual attention mechanism
# - 'rl_actor_critic': Use only the RL actor-critic approach
# - 'hybrid': Start with dual attention, then gradually transition to RL
RL_AGGREGATION_METHOD = 'hybrid'  # Options: 'dual_attention', 'rl_actor_critic', 'hybrid'

# RL Actor-Critic Parameters
RL_ACTOR_HIDDEN_DIMS = [128, 64]       # Hidden layer dimensions for actor network
RL_CRITIC_HIDDEN_DIMS = [128, 64]      # Hidden layer dimensions for critic network
RL_LEARNING_RATE = 0.001
RL_GAMMA = 0.99
RL_ENTROPY_COEF = 0.01
RL_INITIAL_TEMP = 5.0
RL_MIN_TEMP = 0.5
RL_SKIP_PRETRAINING = True  # Skip RL pretraining by default to speed up execution
RL_WARMUP_ROUNDS = 5
RL_RAMP_UP_ROUNDS = 10
RL_PRETRAINING_EPISODES = 100
RL_VALIDATION_MINIBATCH = 0.2
RL_SAVE_INTERVAL = 50

# ======================================
# ATTACK CONFIGURATION
# ======================================

# Attack simulation parameters
ENABLE_ATTACK_SIMULATION = True  # Enable attack simulation
ATTACK_TYPE = 'partial_scaling_attack'  # Type of attack to simulate

# Available attack types:
# - 'scaling_attack': Scale gradients to increase their impact
# - 'partial_scaling_attack': Scale only a portion of the gradients
# - 'label_flipping': Flip labels during training
# - 'sign_flipping': Flip the sign of the gradients
# - 'noise_injection': Add random noise to gradients
# - 'min_max': Minimize loss for some classes, maximize for others
# - 'min_sum': Minimize loss for targeted samples
# - 'targeted_parameters': Attack specific model parameters

# Attack parameters
SCALING_FACTOR = 20.0            # Increased from 15.0 to 20.0 for more obvious attacks
PARTIAL_SCALING_PERCENT = 0.4    # Increased from 0.3 to 0.4 to affect more gradients
FLIP_PROBABILITY = 0.8           # Probability of flipping a label in label flipping attack
NOISE_FACTOR = 5.0               # Factor for noise magnitude in noise injection attack
TARGETED_CLASS = 1               # Target class for class-targeted attacks
EPSILON_L2 = 1.0                 # L2 constraint for projected gradient attacks

# ======================================
# PRIVACY CONFIGURATION
# ======================================

# Privacy mechanism
PRIVACY_MECHANISM = 'none'           # Options: 'none', 'dp' (differential privacy), 'paillier' (homomorphic encryption)

# Differential Privacy parameters
DP_EPSILON = 2.0                     # Privacy budget ε (lower = more privacy but less accuracy)
DP_DELTA = 1e-5                      # Privacy leakage probability δ
DP_CLIP_NORM = 1.0                   # Gradient clipping norm

# ======================================
# MISCELLANEOUS SETTINGS
# ======================================

# Evaluation and logging
LOGGING_LEVEL = 'INFO'               # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# File storage
SAVE_MODEL = True                    # Save final global model
MODEL_SAVE_PATH = 'model_weights/'   # Path to save model

# Random seed
SEED = 42                            # Random seed for reproducibility

# System-wide random seed for reproducibility
RANDOM_SEED = 42

# Note: Root gradients for VAE and DualAttention training come directly from LOCAL_EPOCHS_ROOT
# Honest client gradients for DualAttention come from BENIGN_DA_EPOCHS
# Malicious client gradients for DualAttention come from MALICIOUS_DA_EPOCHS with 10 attack types each 

# Communication Rounds Configuration
# The number of global training rounds to run
COMMUNICATION_ROUNDS = GLOBAL_EPOCHS  # Restored to original value 

# Training Configuration
CLIENT_LR = LR

# Create model weights directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Federated Learning Configuration
DEVICE = device  # Use the device configured above 

# ======================================
# SHAPLEY VALUE CONFIGURATION
# ======================================

# Enable Shapley value calculation
ENABLE_SHAPLEY = True              # Whether to calculate Shapley values to measure client contributions
SHAPLEY_SAMPLES = 5                # Number of Monte Carlo samples for Shapley estimation (higher = more accurate but slower)
SHAPLEY_WEIGHT = 0.3               # Weight of Shapley value in the final trust score (0.0-1.0)
VALIDATION_RATIO = 0.1             # Ratio of test data to use for validation during Shapley calculation
SHAPLEY_BATCH_SIZE = 64            # Batch size for validation during Shapley calculation

# Performance impact of Shapley calculation
# The Shapley calculation adds computational overhead but provides 
# a principled way to measure each client's contribution to model performance.
# For larger models or with many clients, consider reducing SHAPLEY_SAMPLES
# or setting ENABLE_SHAPLEY=False if computational resources are limited. 

# Add missing configuration variables
NORMALIZE_GRADIENTS = True
GRADIENT_CLIP_VALUE = 10.0
# Make sure AGGREGATION_METHOD is defined if not already
if 'AGGREGATION_METHOD' not in globals():
    AGGREGATION_METHOD = 'fedbn' 

# Distribution settings
ENABLE_NON_IID = True       # Whether to use non-IID distribution for client datasets
DIRICHLET_ALPHA = 0.5       # Lower alpha means more non-IID (only used if ENABLE_NON_IID is True) 

# Malicious clients
MALICIOUS_CLIENT_RATIO = 0.33      # Ratio of malicious clients

# Attack settings
DEFAULT_ATTACK_TYPE = 'partial_scaling'  # Attack type to use if unspecified
SCALING_FRACTION = 0.5              # Fraction of parameters to scale for partial scaling attack 