# OPTIMIZED CONFIGURATION FOR BEST RESULTS
# =======================================

# =============================================================================
# MNIST + CNN OPTIMIZATION (Target: 85%+ accuracy, 70%+ detection precision)
# =============================================================================
MNIST_CONFIG = {
    'DATASET': 'MNIST',
    'MODEL': 'CNN',
    'BATCH_SIZE': 32,  # Increased for better convergence
    'LEARNING_RATE': 0.01,  # Higher LR for faster learning
    'LOCAL_EPOCHS_ROOT': 30,  # More epochs for better root model
    'LOCAL_EPOCHS_CLIENT': 8,  # More client training
    'GLOBAL_EPOCHS': 20,  # Sufficient for convergence
    
    # Detection optimization
    'MALICIOUS_PENALTY_FACTOR': 0.4,  # Balanced penalty
    'ZERO_ATTACK_THRESHOLD': 0.005,  # More conservative
    'HIGH_GRADIENT_STD_MULTIPLIER': 2.5,  # Moderate sensitivity
    'CONFIDENCE_THRESHOLD': 0.7,  # Balanced confidence
    'DUAL_ATTENTION_THRESHOLD': 0.65,  # Optimized threshold
    
    # Enhanced network sizes
    'VAE_PROJECTION_DIM': 128,
    'VAE_HIDDEN_DIM': 64,
    'VAE_LATENT_DIM': 32,
    'DUAL_ATTENTION_HIDDEN_SIZE': 256,
    'DUAL_ATTENTION_HEADS': 8,
    'SHAPLEY_WEIGHT': 0.5,
}

# =============================================================================
# CIFAR-10 + RESNET18 OPTIMIZATION (Target: 87%+ accuracy, 65%+ detection precision)
# =============================================================================
CIFAR10_CONFIG = {
    'DATASET': 'CIFAR10',
    'MODEL': 'RESNET18',
    'BATCH_SIZE': 32,  # Increased for better learning
    'LEARNING_RATE': 0.001,  # Optimal for ResNet18
    'LOCAL_EPOCHS_ROOT': 35,  # More training for complex dataset
    'LOCAL_EPOCHS_CLIENT': 6,  # More client epochs
    'GLOBAL_EPOCHS': 30,  # Extended training
    
    # Advanced detection optimization
    'MALICIOUS_PENALTY_FACTOR': 0.35,  # More conservative penalty
    'ZERO_ATTACK_THRESHOLD': 0.002,  # Very conservative
    'HIGH_GRADIENT_STD_MULTIPLIER': 3.0,  # Higher threshold
    'CONFIDENCE_THRESHOLD': 0.75,  # Higher confidence
    'DUAL_ATTENTION_THRESHOLD': 0.7,  # Optimized threshold
    
    # Enhanced network architecture
    'VAE_PROJECTION_DIM': 256,  # Larger for complex features
    'VAE_HIDDEN_DIM': 128,
    'VAE_LATENT_DIM': 64,
    'DUAL_ATTENTION_HIDDEN_SIZE': 512,  # Larger attention
    'DUAL_ATTENTION_HEADS': 16,
    'DUAL_ATTENTION_LAYERS': 4,
    'SHAPLEY_WEIGHT': 0.7,  # Higher Shapley influence
    'SHAPLEY_SAMPLES': 25,  # More samples for accuracy
}

# =============================================================================
# ALZHEIMER + RESNET18 OPTIMIZATION (Target: 98%+ accuracy, 80%+ detection precision)
# =============================================================================
ALZHEIMER_CONFIG = {
    'DATASET': 'ALZHEIMER',
    'MODEL': 'RESNET18',
    'BATCH_SIZE': 24,  # Optimal for medical images
    'LEARNING_RATE': 0.003,  # Balanced learning rate
    'LOCAL_EPOCHS_ROOT': 40,  # Extensive root training
    'LOCAL_EPOCHS_CLIENT': 7,  # More client training
    'GLOBAL_EPOCHS': 30,  # Extended global training
    
    # Precision-optimized detection
    'MALICIOUS_PENALTY_FACTOR': 0.25,  # Very conservative
    'ZERO_ATTACK_THRESHOLD': 0.001,  # Ultra conservative
    'HIGH_GRADIENT_STD_MULTIPLIER': 4.0,  # High threshold
    'CONFIDENCE_THRESHOLD': 0.85,  # Very high confidence
    'DUAL_ATTENTION_THRESHOLD': 0.8,  # High precision threshold
    
    # Advanced medical imaging optimization
    'VAE_PROJECTION_DIM': 512,  # Large for medical features
    'VAE_HIDDEN_DIM': 256,
    'VAE_LATENT_DIM': 128,
    'DUAL_ATTENTION_HIDDEN_SIZE': 768,  # Very large attention
    'DUAL_ATTENTION_HEADS': 24,
    'DUAL_ATTENTION_LAYERS': 6,
    'SHAPLEY_WEIGHT': 0.8,  # Maximum Shapley influence
    'SHAPLEY_SAMPLES': 30,  # Maximum samples
}

# =============================================================================
# EXPECTED PERFORMANCE WITH OPTIMIZED SETTINGS
# =============================================================================
EXPECTED_RESULTS = {
    'MNIST': {
        'accuracy': '85-88%',
        'detection_precision': '70-75%',
        'detection_recall': '95-100%',
        'f1_score': '80-85%'
    },
    'CIFAR10': {
        'accuracy': '87-90%', 
        'detection_precision': '65-70%',
        'detection_recall': '95-100%',
        'f1_score': '77-82%'
    },
    'ALZHEIMER': {
        'accuracy': '98-99%',
        'detection_precision': '80-85%', 
        'detection_recall': '95-100%',
        'f1_score': '87-92%'
    }
} 

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

# Memory Management Configuration - OPTIMIZED FOR REALISTIC RESULTS
VAE_DEVICE = 'gpu'  # Use GPU for better performance
ENABLE_MEMORY_TRACKING = True
AGGRESSIVE_MEMORY_CLEANUP = True

# Gradient Memory Optimization - IMPROVED
GRADIENT_CHUNK_SIZE = 100000    # Increased for better performance
GRADIENT_AGGREGATION_METHOD = 'mean'
MAX_GRADIENT_NORM = 10.0

# Verbosity settings
VERBOSE = True

# Gradient Dimension Reduction - BALANCED
ENABLE_DIMENSION_REDUCTION = True
DIMENSION_REDUCTION_RATIO = 0.15  # Less aggressive for better accuracy
ENABLE_PARALLEL_PROCESSING = False

# ======================================
# FEDERATED LEARNING PARAMETERS - OPTIMIZED FOR ACCURACY
# ======================================

# Global parameters - IMPROVED FOR REALISTIC RESULTS
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
NUM_MALICIOUS = int(NUM_CLIENTS * FRACTION_MALICIOUS)

# IMPROVED TRAINING PARAMETERS
BATCH_SIZE = 32                    # ✅ Increased from 16 for better training
LR = 0.01                          # ✅ Good learning rate
LOCAL_EPOCHS_ROOT = 15             # ✅ Increased from 5 for better pretraining
LOCAL_EPOCHS_CLIENT = 5            # ✅ Increased from 3 for better local training
GLOBAL_EPOCHS = 25                 # ✅ Increased from 3 for realistic training

CLIENT_SELECTION_RATIO = 1.0
CLIENT_FRACTION = 1.0
LEARNING_RATE = LR
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CLIENT_EPOCHS = LOCAL_EPOCHS_CLIENT

# Learning rate scheduling - OPTIMIZED
LR_DECAY = 0.99                    # More aggressive decay
LR_DECAY_EPOCHS = 3                # Apply decay every 3 epochs
MIN_LR = 0.0001

# Data loading configuration - OPTIMIZED
NUM_WORKERS = 0                    # Keep 0 for stability
PIN_MEMORY = True                  # Enable for better GPU performance

# ======================================
# MODEL AND DATASET CONFIGURATION
# ======================================

# Dataset and model configuration - CIFAR10 with realistic settings
DATASET = 'CIFAR10'
MODEL = 'ResNet18'
INPUT_CHANNELS = 3
NUM_CLASSES = 10

# ResNet configuration - OPTIMIZED
RESNET50_UNFREEZE_LAYERS = 20
RESNET18_UNFREEZE_LAYERS = 8       # ✅ Increased for better learning
RESNET_PRETRAINED = True

# Dataset paths
ALZHEIMER_DATA_ROOT = 'data/alzheimer'
CIFAR_DATA_ROOT = 'data/cifar'
MNIST_DATA_ROOT = 'data/mnist'

# MNIST specific settings
MNIST_IMG_SIZE = 28
MNIST_CLASSES = 10

# Alzheimer's dataset configuration
ALZHEIMER_DATA_DIR = './data/alzheimer'
ALZHEIMER_IMG_SIZE = 224
ALZHEIMER_CLASSES = 4

# CIFAR-10 specific settings
CIFAR_IMG_SIZE = 32
CIFAR_CLASSES = 10

# VAE training configuration - IMPROVED FOR ACCURACY
VAE_EPOCHS = 20                    # ✅ Increased from 12 for better VAE training
VAE_BATCH_SIZE = 16                # ✅ Increased from 6 for better training
VAE_LEARNING_RATE = 0.0005         # ✅ Increased for faster convergence
VAE_PROJECTION_DIM = 128           # ✅ Increased for better representation
VAE_HIDDEN_DIM = 64                # ✅ Increased for better capacity
VAE_LATENT_DIM = 32                # ✅ Increased for better representation
GRADIENT_DIMENSION = None

# ======================================
# DATA DISTRIBUTION CONFIGURATION
# ======================================

# Data distribution - IID for baseline
ENABLE_NON_IID = False
DIRICHLET_ALPHA = None
LABEL_SKEW_RATIO = None
QUANTITY_SKEW_RATIO = None

# Root dataset configuration - IMPROVED
ROOT_DATASET_RATIO = 0.20          # ✅ Increased for better root training
ROOT_DATASET_SIZE = 5000           # ✅ Increased for more training data
ROOT_DATASET_DYNAMIC_SIZE = True
BIAS_PROBABILITY = 0.1
BIAS_CLASS = 1

# ======================================
# GRADIENT COMBINATION METHOD CONFIGURATION
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
# MALICIOUS CLIENT DETECTION - ENHANCED
# ======================================

# Dual Attention parameters - IMPROVED
ENABLE_DUAL_ATTENTION = True
DUAL_ATTENTION_HIDDEN_SIZE = 256   # ✅ Increased for better detection
DUAL_ATTENTION_HEADS = 12          # ✅ Increased for better attention
DUAL_ATTENTION_LAYERS = 4          # ✅ Increased for better representation
DUAL_ATTENTION_EPOCHS = 10         # ✅ Increased from 5 for better training
DUAL_ATTENTION_BATCH_SIZE = 16     # ✅ Increased for better training
DUAL_ATTENTION_LEARNING_RATE = 0.0005

# VAE parameters for anomaly detection
ENABLE_VAE = True

# Shapley value integration - ENHANCED
ENABLE_SHAPLEY = True
SHAPLEY_SAMPLES = 25               # ✅ Increased for better precision
SHAPLEY_WEIGHT = 0.6               # ✅ Increased for stronger discrimination
VALIDATION_RATIO = 0.15
SHAPLEY_BATCH_SIZE = 32            # ✅ Increased for better efficiency

# Malicious weighting method
MALICIOUS_WEIGHTING_METHOD = 'continuous'

# Malicious penalty factor - BALANCED
MALICIOUS_PENALTY_FACTOR = 0.3     # ✅ Balanced penalty
ZERO_ATTACK_THRESHOLD = 0.01       # ✅ More reasonable threshold
HIGH_GRADIENT_STD_MULTIPLIER = 2.5
CONFIDENCE_THRESHOLD = 0.65        # ✅ Balanced confidence
DUAL_ATTENTION_THRESHOLD = 0.65

# Attacker impact weighing
ATTACKER_IMPACT_WEIGHING = True

# ======================================
# RL-BASED AGGREGATION CONFIGURATION
# ======================================

RL_AGGREGATION_METHOD = 'hybrid'

# RL Actor-Critic Parameters - ENHANCED
RL_ACTOR_HIDDEN_DIMS = [128, 64, 32]  # ✅ Increased capacity
RL_CRITIC_HIDDEN_DIMS = [128, 64, 32] # ✅ Increased capacity
RL_LEARNING_RATE = 0.001
RL_EPSILON = 0.1
RL_MEMORY_SIZE = 2000              # ✅ Increased memory
RL_BATCH_SIZE = 16                 # ✅ Increased batch size
RL_UPDATE_FREQUENCY = 5
RL_GAMMA = 0.99
RL_ENTROPY_COEF = 0.01
RL_INITIAL_TEMP = 5.0
RL_MIN_TEMP = 0.5
RL_SKIP_PRETRAINING = True
RL_WARMUP_ROUNDS = 8               # ✅ Increased warmup
RL_RAMP_UP_ROUNDS = 15             # ✅ Increased ramp-up
RL_PRETRAINING_EPISODES = 100
RL_VALIDATION_MINIBATCH = 0.2
RL_SAVE_INTERVAL = 50

# ======================================
# ATTACK CONFIGURATION
# ======================================

ENABLE_ATTACK_SIMULATION = True

# Attack parameters - REALISTIC FOR RESEARCH
SCALING_FACTOR = 3.0               # ✅ More realistic scaling
PARTIAL_SCALING_PERCENT = 0.4      # ✅ Balanced partial scaling
NOISE_FACTOR = 1.5                 # ✅ More realistic noise
FLIP_PROBABILITY = 0.5             # ✅ More realistic probability
TARGETED_CLASS = 7

# ======================================
# PRIVACY CONFIGURATION
# ======================================

PRIVACY_MECHANISM = 'none'

# Differential Privacy parameters
DP_EPSILON = 2.0
DP_DELTA = 1e-5
DP_CLIP_NORM = 1.0

# ======================================
# MISCELLANEOUS SETTINGS
# ======================================

LOGGING_LEVEL = 'INFO'
SAVE_MODEL = True
MODEL_SAVE_PATH = 'model_weights/'
RANDOM_SEED = 42

# Communication Rounds Configuration
COMMUNICATION_ROUNDS = GLOBAL_EPOCHS

# Training Configuration
CLIENT_LR = LR

# Create model weights directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Federated Learning Configuration
DEVICE = device

# ======================================
# ENHANCED DETECTION CONFIGURATION
# ======================================

NORMALIZE_GRADIENTS = True
GRADIENT_CLIP_VALUE = 10.0

# Detection threshold optimization - IMPROVED
GRADIENT_NORM_THRESHOLD_FACTOR = 2.2    # ✅ Balanced sensitivity
GRADIENT_NORM_MIN_THRESHOLD = 0.15      # ✅ Lower threshold for better detection
ZERO_ATTACK_THRESHOLD = 0.015           # ✅ More sensitive
SUSPICIOUS_CLIENT_PENALTY = 0.6         # ✅ Balanced penalty

# Trust score thresholding - ENHANCED
TRUST_SCORE_THRESHOLD = 0.55            # ✅ Balanced threshold
ADAPTIVE_THRESHOLD = True

# Malicious clients - RESEARCH CONFIGURATION
MALICIOUS_CLIENT_RATIO = 0.3

# Attack settings
SCALING_FRACTION = 0.5

# Additional detection refinements
FALSE_POSITIVE_MITIGATION = True
MIN_CLIENTS_FOR_DETECTION = 2
GRADIENT_VARIANCE_THRESHOLD = 0.5       # ✅ More sensitive

# Detection thresholds - BALANCED
MALICIOUS_THRESHOLD = 0.65              # ✅ More balanced detection
CONFIDENCE_THRESHOLD = 0.65             # ✅ Balanced confidence
DETECTION_SENSITIVITY = 0.7             # ✅ Improved sensitivity

# ======================================
# SPECIAL CONFIGURATIONS FOR ACCURACY IMPROVEMENT
# ======================================

# Enable gradient accumulation for larger effective batch sizes
GRADIENT_ACCUMULATION_STEPS = 2         # ✅ Simulate batch_size = 64

# Early stopping for better convergence
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning rate scheduling
ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'step'             # Options: 'step', 'cosine', 'exponential'
LR_STEP_SIZE = 8                       # Reduce LR every 8 epochs
LR_GAMMA = 0.5                         # Multiply LR by 0.5

# Model checkpoint saving
SAVE_BEST_MODEL = True
MODEL_CHECKPOINT_INTERVAL = 5

# Evaluation frequency
EVAL_FREQUENCY = 1                     # Evaluate every epoch

print("✅ OPTIMIZED CONFIG LOADED - TARGETING 85%+ CIFAR-10 ACCURACY")
print(f"📊 Key improvements:")
print(f"   - GLOBAL_EPOCHS: {GLOBAL_EPOCHS} (was 3)")
print(f"   - BATCH_SIZE: {BATCH_SIZE} (was 16)")
print(f"   - LOCAL_EPOCHS_ROOT: {LOCAL_EPOCHS_ROOT} (was 5)")
print(f"   - VAE_EPOCHS: {VAE_EPOCHS} (was 12)")
print(f"   - DUAL_ATTENTION_HIDDEN_SIZE: {DUAL_ATTENTION_HIDDEN_SIZE} (was 128)")
print(f"   - ROOT_DATASET_SIZE: {ROOT_DATASET_SIZE} (was 3500)") 