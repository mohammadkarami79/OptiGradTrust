"""
🚀 NON-IID CONFIGURATION - PHASE 2
=====================================

Non-IID federated learning configuration optimized for
comparative analysis with IID results.

Author: Research Team
Date: 2025-01-27
Target: Generate comprehensive IID vs Non-IID comparison
"""

import os
import torch

# =============================================================================
# NON-IID DATA DISTRIBUTION CONFIGURATION
# =============================================================================

# Data heterogeneity settings
NON_IID_ALPHA = 0.1  # Dirichlet alpha (lower = more heterogeneous)
NON_IID_CLASSES_PER_CLIENT = 2  # Limit classes per client
NON_IID_IMBALANCE_RATIO = 0.7  # Data imbalance level (0.5 = balanced, 1.0 = extreme)

# Distribution strategy
DATA_DISTRIBUTION = 'non_iid'  # 'iid' or 'non_iid'
HETEROGENEITY_LEVEL = 0.7  # 0.0 = IID, 1.0 = extreme Non-IID

# =============================================================================
# OPTIMIZED TRAINING PARAMETERS (Based on IID success)
# =============================================================================

# Core FL parameters - balanced for Non-IID
GLOBAL_EPOCHS = 15  # Reduced from IID 20 (Non-IID needs more careful convergence)
LOCAL_EPOCHS_ROOT = 10  # Reduced from IID 12 
LOCAL_EPOCHS_CLIENT = 5  # Increased from IID 4 (clients need more local training)

# Batch and data parameters
BATCH_SIZE = 32  # Same as IID optimized
ROOT_DATASET_SIZE = 4500  # Same as IID optimized
LEARNING_RATE = 0.008  # Slightly reduced for Non-IID stability

# Clients configuration
NUM_CLIENTS = 10
NUM_ROOT_CLIENTS = 1  # Root server client
FRACTION_MALICIOUS = 0.3  # 30% malicious clients

# =============================================================================
# DETECTION SYSTEM OPTIMIZATION FOR NON-IID
# =============================================================================

# VAE parameters - enhanced for Non-IID complexity
VAE_EPOCHS = 18  # Increased from IID 15 (more epochs for diverse data)
VAE_BATCH_SIZE = 10  # Reduced from IID 12 (smaller batches for stability)
VAE_LEARNING_RATE = 0.0004  # Reduced from IID 0.0005 (more conservative)
VAE_PROJECTION_DIM = 128  # Same as IID optimized
VAE_HIDDEN_DIM = 64  # Same as IID optimized  
VAE_LATENT_DIM = 32  # Same as IID optimized

# Dual Attention parameters - adapted for Non-IID
DUAL_ATTENTION_HIDDEN_SIZE = 180  # Reduced from IID 200 (prevent overfitting)
DUAL_ATTENTION_HEADS = 8  # Reduced from IID 10 (more robust)
DUAL_ATTENTION_EPOCHS = 10  # Increased from IID 8 (more training needed)
DUAL_ATTENTION_BATCH_SIZE = 10  # Reduced from IID 12 (stability)
DUAL_ATTENTION_LAYERS = 3  # Standard depth

# Detection thresholds - relaxed for Non-IID diversity
GRADIENT_NORM_THRESHOLD_FACTOR = 2.5  # Increased from IID 2.0 (more tolerance)
TRUST_SCORE_THRESHOLD = 0.55  # Reduced from IID 0.6 (more tolerance)
MALICIOUS_THRESHOLD = 0.6  # Reduced from IID 0.65 (more tolerance)
ZERO_ATTACK_THRESHOLD = 0.015  # Increased from IID 0.01 (less sensitive)

# Shapley parameters
SHAPLEY_SAMPLES = 20  # Reduced from IID 25 (computational efficiency)
SHAPLEY_WEIGHT = 0.35  # Reduced from IID 0.4 (balance with other features)

# =============================================================================
# DATASET AND MODEL CONFIGURATION
# =============================================================================

# Priority datasets for Non-IID comparison
DATASETS_PRIORITY = ['MNIST', 'ALZHEIMER', 'CIFAR10']  # Order of testing

# Dataset-specific configurations
MNIST_CONFIG = {
    'DATASET': 'MNIST',
    'MODEL': 'CNN',
    'NON_IID_CLASSES_PER_CLIENT': 2,  # Each client gets 2 out of 10 classes
    'EXPECTED_ACCURACY': 97.0,  # Reduced from IID 99.41%
    'EXPECTED_DETECTION': 45.0,  # Reduced from IID 69.23%
}

ALZHEIMER_CONFIG = {
    'DATASET': 'ALZHEIMER', 
    'MODEL': 'RESNET18',
    'NON_IID_CLASSES_PER_CLIENT': 1,  # Each client gets 1 out of 4 classes
    'EXPECTED_ACCURACY': 94.0,  # Reduced from IID 97.24%
    'EXPECTED_DETECTION': 60.0,  # Reduced from IID 75.00%
}

CIFAR10_CONFIG = {
    'DATASET': 'CIFAR10',
    'MODEL': 'RESNET18', 
    'NON_IID_CLASSES_PER_CLIENT': 2,  # Each client gets 2 out of 10 classes
    'EXPECTED_ACCURACY': 78.0,  # Reduced from IID 85.20%
    'EXPECTED_DETECTION': 65.0,  # Maintained high (complex attacks easier to detect)
}

# =============================================================================
# ATTACK CONFIGURATION FOR NON-IID
# =============================================================================

# Priority attacks for comparison
PRIORITY_ATTACKS = [
    'partial_scaling_attack',  # Usually best performing
    'noise_attack',           # Good baseline
    'sign_flipping_attack',   # Standard test
]

# Attack parameters - same as IID for fair comparison
SCALING_FACTOR = 10.0
PARTIAL_SCALING_PERCENT = 0.5
NOISE_FACTOR = 5.0
FLIP_PROBABILITY = 0.5

# =============================================================================
# EXPERIMENTAL PROTOCOL
# =============================================================================

# Testing phases
PHASE_2_TIMELINE = {
    'MNIST_NON_IID': '30 minutes',
    'ALZHEIMER_NON_IID': '45 minutes', 
    'CIFAR10_NON_IID': '60 minutes',
    'COMPARISON_ANALYSIS': '15 minutes',
    'TOTAL_ESTIMATED': '2.5 hours'
}

# Success criteria for Non-IID
SUCCESS_CRITERIA = {
    'min_accuracy_retention': 0.85,  # At least 85% of IID accuracy
    'min_detection_retention': 0.70,  # At least 70% of IID detection
    'max_performance_drop': 0.20,    # Maximum 20% performance drop
}

# =============================================================================
# HARDWARE AND SYSTEM CONFIGURATION
# =============================================================================

# Memory optimization for Non-IID (more complex)
MEMORY_OPTIMIZATION = True
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = device

# Logging and output
LOGGING_LEVEL = 'INFO'
SAVE_MODEL = True
MODEL_SAVE_PATH = 'model_weights/non_iid/'
RESULTS_PATH = 'results/non_iid_experiments/'

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# =============================================================================
# COMPARISON FRAMEWORK
# =============================================================================

# IID results for comparison (from Phase 1)
IID_BASELINE = {
    'MNIST': {'accuracy': 99.41, 'detection': 69.23},
    'ALZHEIMER': {'accuracy': 97.24, 'detection': 75.00},
    'CIFAR10': {'accuracy': 85.20, 'detection': 45.00},
}

# Expected Non-IID performance drops
EXPECTED_DROPS = {
    'MNIST': {'accuracy': -2.4, 'detection': -24.2},      # 97.0%, 45.0%
    'ALZHEIMER': {'accuracy': -3.2, 'detection': -15.0},  # 94.0%, 60.0%
    'CIFAR10': {'accuracy': -7.2, 'detection': +20.0},    # 78.0%, 65.0%
}

# =============================================================================
# AUTO-CONFIGURATION SELECTION
# =============================================================================

def auto_select_config(dataset_name):
    """Auto-select configuration based on dataset"""
    configs = {
        'MNIST': MNIST_CONFIG,
        'ALZHEIMER': ALZHEIMER_CONFIG, 
        'CIFAR10': CIFAR10_CONFIG,
    }
    return configs.get(dataset_name, MNIST_CONFIG)

def get_non_iid_timeline():
    """Get estimated timeline for Non-IID experiments"""
    return PHASE_2_TIMELINE

def validate_non_iid_config():
    """Validate Non-IID configuration"""
    checks = {
        'data_distribution': DATA_DISTRIBUTION == 'non_iid',
        'heterogeneity_level': 0.5 <= HETEROGENEITY_LEVEL <= 1.0,
        'classes_per_client': NON_IID_CLASSES_PER_CLIENT >= 1,
        'detection_thresholds': TRUST_SCORE_THRESHOLD < 0.7,
    }
    
    all_valid = all(checks.values())
    if all_valid:
        print("✅ Non-IID configuration validated successfully")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"❌ Configuration validation failed: {failed}")
    
    return all_valid

# =============================================================================
# READY STATUS
# =============================================================================

NON_IID_READY = True
PHASE_2_PREPARED = True

print("🚀 NON-IID CONFIGURATION LOADED")
print(f"📊 Data Distribution: {DATA_DISTRIBUTION}")
print(f"🎯 Heterogeneity Level: {HETEROGENEITY_LEVEL}")
print(f"⏱️ Estimated Timeline: {PHASE_2_TIMELINE['TOTAL_ESTIMATED']}")
print("✅ Ready to proceed with Phase 2: Non-IID Experiments") 