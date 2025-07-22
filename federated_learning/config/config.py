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
GRADIENT_CHUNK_SIZE = 10000    # Further reduced for memory optimization
GRADIENT_AGGREGATION_METHOD = 'mean'
MAX_GRADIENT_NORM = 10.0  # Increased from 5.0 to better detect scaling attacks

# Verbosity settings
VERBOSE = True  # Set to True for detailed debug information, particularly for BatchNorm layer tracking

# Gradient Dimension Reduction for Memory Optimization
ENABLE_DIMENSION_REDUCTION = True     # Enable dimension reduction
DIMENSION_REDUCTION_RATIO = 0.10      # Very aggressive reduction (keep 10%) for memory
ENABLE_PARALLEL_PROCESSING = False    # Keep parallel processing disabled

# ======================================
# FEDERATED LEARNING PARAMETERS
# ======================================

# Global parameters - MEMORY OPTIMIZED
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)
BATCH_SIZE = 32                    # Optimal batch for MNIST
LR = 0.01                          # Learning rate for MNIST CNN
LOCAL_EPOCHS_ROOT = 12             # Root pretrain epochs
LOCAL_EPOCHS_CLIENT = 4            # Client local epochs
GLOBAL_EPOCHS = 20                 # Global rounds
CLIENT_SELECTION_RATIO = 1.0
CLIENT_FRACTION = 1.0
LEARNING_RATE = LR
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4                # Standard weight decay
CLIENT_EPOCHS = LOCAL_EPOCHS_CLIENT

# Learning rate scheduling - Optimized for long training
LR_DECAY = 0.98                    # Slightly faster decay to match lower LR
LR_DECAY_EPOCHS = 2                # Apply decay every 2 epochs
MIN_LR = 0.0001                    # Minimum learning rate

# Data loading configuration
NUM_WORKERS = 0                    # Disable worker threads for memory optimization
PIN_MEMORY = False                # Disable pin memory to save GPU memory

# ======================================
# MODEL AND DATASET CONFIGURATION
# ======================================

# Dataset and model configuration - Set for MNIST non-IID label skew
DATASET = 'MNIST'
MODEL = 'CNN'
INPUT_CHANNELS = 1  # MNIST is grayscale
NUM_CLASSES = 10

# ResNet configuration
RESNET50_UNFREEZE_LAYERS = 20      # Number of layers to unfreeze from the end for ResNet50
RESNET18_UNFREEZE_LAYERS = 5       # Number of layers to unfreeze from the end for ResNet18
RESNET_PRETRAINED = True           # Whether to use pretrained weights

# Dataset paths - Updated for MNIST
ALZHEIMER_DATA_ROOT = 'data/alzheimer'
CIFAR_DATA_ROOT = 'data/cifar'
MNIST_DATA_ROOT = 'data/mnist'

# MNIST specific settings
MNIST_IMG_SIZE = 28               # MNIST images are 28x28
MNIST_CLASSES = 10                # 10 classes in MNIST (0-9)

# Alzheimer's dataset configuration
ALZHEIMER_DATA_DIR = './data/alzheimer'
ALZHEIMER_IMG_SIZE = 224          # Size to resize images to
ALZHEIMER_CLASSES = 4             # Number of classes

# CIFAR-10 specific settings
CIFAR_IMG_SIZE = 32               # CIFAR-10 images are 32x32
CIFAR_CLASSES = 10                # 10 classes in CIFAR-10

# VAE training configuration - MEMORY OPTIMIZED
VAE_EPOCHS = 15                    # Enough for MNIST
VAE_BATCH_SIZE = 16                # Larger batch for stable VAE training
VAE_LEARNING_RATE = 0.0005         # ✅ Increased for faster learning
VAE_PROJECTION_DIM = 64            # Reduced for memory
VAE_HIDDEN_DIM = 32                # Reduced for memory
VAE_LATENT_DIM = 16                # Reduced for memory
GRADIENT_DIMENSION = None          # Will be set automatically

# ======================================
# DATA DISTRIBUTION CONFIGURATION
# ======================================

# Set to IID for baseline
ENABLE_NON_IID = False
DATA_DISTRIBUTION = 'iid'
DIRICHLET_ALPHA = None
NON_IID_CLASSES_PER_CLIENT = None
LABEL_SKEW_RATIO = None
QUANTITY_SKEW_RATIO = None
NON_IID_TYPE = None
NON_IID_SEVERITY = None

# Root dataset configuration - OPTIMIZED 
ROOT_DATASET_RATIO = 0.18          # ✅ Increased for better training
ROOT_DATASET_SIZE = 4500           # ✅ Increased for more training data       
ROOT_DATASET_DYNAMIC_SIZE = True  
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1

# Distribution settings - تغییر به IID
ENABLE_NON_IID = False             # غیرفعال برای IID

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
GRADIENT_COMBINATION_METHOD = 'fedbn_fedprox'  # Hybrid aggregation

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

# Dual Attention parameters - MEMORY OPTIMIZED
ENABLE_DUAL_ATTENTION = True       # Enable dual attention-based detection
DUAL_ATTENTION_HIDDEN_SIZE = 200   # Reduced for memory
DUAL_ATTENTION_HEADS = 10           # Reduced for memory
DUAL_ATTENTION_LAYERS = 3          # Reduced for memory and performance
DUAL_ATTENTION_EPOCHS = 8          # Reduced for MNIST
DUAL_ATTENTION_BATCH_SIZE = 12     # Larger batch for attention training
DUAL_ATTENTION_LEARNING_RATE = 0.0005  # ✅ Increased for faster learning

# VAE parameters for anomaly detection - Enhanced for research
ENABLE_VAE = True                # Enable VAE-based anomaly detection

# Shapley value integration - MEMORY OPTIMIZED
ENABLE_SHAPLEY = True            # Enable Shapley value calculation
SHAPLEY_SAMPLES = 25               # ok for MNIST
SHAPLEY_WEIGHT = 0.4               # Increased from 0.5 for stronger discrimination
VALIDATION_RATIO = 0.15            # Validation ratio for Shapley calculation
SHAPLEY_BATCH_SIZE = 8             # Reduced for memory optimization

# Malicious weighting method - Research-grade configuration
MALICIOUS_WEIGHTING_METHOD = 'continuous'

# Malicious penalty factor - Optimized for much better precision
MALICIOUS_PENALTY_FACTOR = 0.4     # Balanced penalty (was 0.3)
ZERO_ATTACK_THRESHOLD = 0.01      # More conservative (was 0.001)
HIGH_GRADIENT_STD_MULTIPLIER = 2.5 # Moderate sensitivity (was 3.5)
CONFIDENCE_THRESHOLD = 0.7         # Balanced confidence (was 0.8)
DUAL_ATTENTION_THRESHOLD = 0.65    # Optimized threshold (was 0.75)

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
RL_AGGREGATION_METHOD = 'hybrid'    # Using hybrid approach as intended

# RL Actor-Critic Parameters
RL_ACTOR_HIDDEN_DIMS = [64, 32]        # Balanced for memory and performance
RL_CRITIC_HIDDEN_DIMS = [64, 32]       # Balanced for memory and performance
RL_LEARNING_RATE = 0.001           # Learning rate for RL agent
RL_EPSILON = 0.1                   # Exploration rate for RL
RL_MEMORY_SIZE = 1000              # Memory size for experience replay
RL_BATCH_SIZE = 8                  # Balanced for memory and performance
RL_UPDATE_FREQUENCY = 5            # Update RL agent every N rounds
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

# Attack simulation parameters - Optimized for comprehensive research evaluation
ENABLE_ATTACK_SIMULATION = True  # Enable attack simulation
# ATTACK_TYPE = 'partial_scaling_attack'  # Removed: main.py tests all attacks systematically

# Available attack types:
# - 'scaling_attack': Scale gradients to increase their impact
# - 'partial_scaling_attack': Scale only a portion of the gradients
# - 'label_flipping': Flip labels during training
# - 'sign_flipping_attack': Flip the sign of the gradients  
# - 'noise_attack': Add random noise to gradients
# - 'min_max_attack': Minimize loss for some classes, maximize for others
# - 'min_sum_attack': Minimize loss for targeted samples
# - 'targeted_attack': Attack specific model parameters

# Attack parameters - Conference suitable with improved realism
SCALING_FACTOR = 10.0              # Reduced from 10.0 for more subtle attacks
PARTIAL_SCALING_PERCENT = 0.5      # Reduced from 0.5 for more subtle attacks
NOISE_FACTOR = 5.0                 # Reduced from 5.0 for more realistic noise
FLIP_PROBABILITY = 0.5             # Reduced from 0.8 for more realistic attacks
TARGETED_CLASS = 7                 # Target class for targeted attacks

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

# Random seed for reproducibility - Conference requirement
RANDOM_SEED = 42                   # Fixed seed for reproducible results

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

# Enable Shapley value calculation - Conference settings
ENABLE_SHAPLEY = True              # Enable Shapley value calculation
SHAPLEY_SAMPLES = 25               # Number of Monte Carlo samples (good balance)
SHAPLEY_WEIGHT = 0.4               # Weight of Shapley value in trust score
VALIDATION_RATIO = 0.15            # Validation ratio for Shapley calculation
SHAPLEY_BATCH_SIZE = 16            # Balanced for memory and performance

# Performance impact of Shapley calculation
# The Shapley calculation adds computational overhead but provides 
# a principled way to measure each client's contribution to model performance.
# With higher SHAPLEY_SAMPLES (15), accuracy improves but computation increases.
# For publication-quality results, this trade-off is worthwhile.

# Add missing configuration variables
NORMALIZE_GRADIENTS = True
GRADIENT_CLIP_VALUE = 10.0
# Make sure AGGREGATION_METHOD is defined if not already
if 'AGGREGATION_METHOD' not in globals():
    AGGREGATION_METHOD = 'fedbn_fedprox'  # Updated to match GRADIENT_COMBINATION_METHOD

# Malicious clients - Research-optimized configuration
MALICIOUS_CLIENT_RATIO = 0.3      # 30% malicious clients (standard for research evaluation)

# Attack settings - Enhanced for comprehensive evaluation
SCALING_FRACTION = 0.5              # Increased fraction of parameters to scale for stronger attacks

# Note: Removed DEFAULT_ATTACK_TYPE to avoid redundancy with ATTACK_TYPE parameter
# Use ATTACK_TYPE (defined in Attack Configuration section) to set the attack type

# ======================================
# ENHANCED DETECTION CONFIGURATION
# ======================================

# Detection threshold optimization - FIXED for better accuracy
GRADIENT_NORM_THRESHOLD_FACTOR = 2.5    # Reduced from 3.0 for more sensitive norm detection
GRADIENT_NORM_MIN_THRESHOLD = 0.2       # Reduced from 0.3 for lower floor
ZERO_ATTACK_THRESHOLD = 0.005     # Reduced from 0.03 for less false positives
SUSPICIOUS_CLIENT_PENALTY = 0.7         # Reduced from 0.85 to allow better model learning

# Trust score thresholding - Improved balance
TRUST_SCORE_THRESHOLD = 0.5            # Increased from 0.30 for more specific detection
ADAPTIVE_THRESHOLD = True               # Enable adaptive thresholding based on distribution

# # Model improvement settings - ENHANCED for better learning
# LOCAL_EPOCHS_CLIENT = 10                # Increased from 8 to 10 for stronger local updates
# GLOBAL_EPOCHS = 30                      # Increased back to 30 for more comprehensive training
# LR = 0.01                               # Increased from 0.008 for faster convergence

# Additional detection refinements
FALSE_POSITIVE_MITIGATION = True        # Enable additional checks to reduce false positives
MIN_CLIENTS_FOR_DETECTION = 2          # Require at least 2 suspicious patterns before flagging
GRADIENT_VARIANCE_THRESHOLD = 0.6       # Reduced from 0.8 for more variance sensitivity

# Detection thresholds - Conference optimized  
MALICIOUS_THRESHOLD = 0.55       # Increased from 0.5 for very conservative detection
CONFIDENCE_THRESHOLD = 0.7         # Balanced confidence (was 0.8)
DETECTION_SENSITIVITY = 0.6      # Reduced from 0.9 for less sensitivity (fewer false positives)

# Enhanced confidence thresholds
DUAL_ATTENTION_THRESHOLD = 0.65    # Optimized threshold (was 0.75)

# Note: RL_AGGREGATION_METHOD is already defined above as 'hybrid'
# Removed duplicate RL configuration to avoid conflicts