# OptiGradTrust: Byzantine-Robust Federated Learning with Multi-Feature Gradient Analysis and Reinforcement Learning-Based Trust Weighting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

**OptiGradTrust** is a novel Byzantine-robust federated learning framework that addresses the critical challenges of malicious client detection and data heterogeneity in distributed machine learning environments. Our comprehensive defense system evaluates gradient updates through a novel **six-dimensional fingerprint** and employs a **hybrid RL-attention module** for adaptive trust scoring, while introducing **FedBN-Prox (FedBN-P)** to optimize convergence under non-IID conditions.

### Abstract

Federated Learning (FL) enables collaborative model training across distributed medical institutions while preserving patient privacy, but remains vulnerable to Byzantine attacks and statistical heterogeneity. We present **OptiGradTrust**, a comprehensive defense framework that evaluates gradient updates through a novel six-dimensional fingerprint including VAE reconstruction error, cosine similarity metrics, L2 norm, sign-consistency ratio, and Monte Carlo Shapley value, which drive a hybrid RL-attention module for adaptive trust scoring. To address convergence challenges under data heterogeneity, we develop **FedBN-Prox (FedBN-P)**, combining Federated Batch Normalization with proximal regularization for optimal accuracy-convergence trade-offs. Extensive evaluation across MNIST, CIFAR-10, and Alzheimer's MRI datasets under various Byzantine attack scenarios demonstrates significant improvements over state-of-the-art defenses, achieving up to **+1.6 percentage points** over FLGuard under non-IID conditions while maintaining robust performance against diverse attack patterns through our adaptive learning approach.

## Key Features

### 🛡️ **OptiGradTrust Core Innovations**

#### **Six-Dimensional Gradient Fingerprinting**
- **VAE Reconstruction Error**: Advanced anomaly detection using variational autoencoders
- **Cosine Similarity Metrics**: Multi-level similarity analysis (root-client, client-client)
- **L2 Norm Analysis**: Gradient magnitude consistency evaluation
- **Sign-Consistency Ratio**: Directional gradient alignment assessment
- **Monte Carlo Shapley Value**: Game-theoretic client contribution analysis
- **Gradient Norm Analysis**: Comprehensive magnitude-based detection

#### **Hybrid RL-Attention Module**
- **Reinforcement Learning Integration**: Adaptive trust weighting with actor-critic architecture
- **Transformer-Based Attention**: Multi-head attention for complex pattern recognition
- **Dynamic Trust Scoring**: Real-time assessment of client trustworthiness
- **Adaptive Thresholding**: Self-adjusting detection sensitivity

#### **FedBN-Prox (FedBN-P) Optimization**
- **Federated Batch Normalization**: Statistical independence for non-IID robustness
- **Proximal Regularization**: Enhanced convergence stability
- **Optimal Trade-offs**: Balanced accuracy-convergence performance
- **Heterogeneity Resilience**: Superior performance under data distribution skew

### 🔬 **Comprehensive Byzantine Attack Defense**
- **5 Primary Attack Types**: Scaling, partial scaling, sign flipping, Gaussian noise, label flipping
- **Advanced Attack Simulation**: Sophisticated threat modeling for realistic evaluation
- **Attack Impact Quantification**: Detailed analysis of attack effectiveness and mitigation
- **Progressive Learning**: Adaptive improvement in detection capabilities over training rounds

### 🔄 **Multi-Domain Aggregation Framework**
- **FedAvg**: Standard federated averaging baseline
- **FedBN**: Batch normalization parameter isolation
- **FedProx**: Proximal term for convergence stability
- **FedBN-Prox (FedBN-P)**: Our novel hybrid optimization method
- **RL-Guided Aggregation**: Reinforcement learning-based adaptive weighting
- **Hybrid Methods**: Intelligent combination of multiple aggregation strategies

### 📊 **Comprehensive Multi-Domain Evaluation**
- **Detection Performance**: Precision, Recall, F1-Score across all Byzantine attack scenarios
- **Model Accuracy**: Performance tracking across MNIST, CIFAR-10, and Alzheimer's MRI datasets
- **Statistical Validation**: p-value analysis, Cohen's d effect sizes, confidence intervals
- **Progressive Learning**: Documented improvement in detection rates across training rounds
- **Comparative Analysis**: State-of-the-art comparisons with FLTrust, FLGuard, and baseline methods
- **Results Management**: Comprehensive experiment tracking and reproducibility tools

### 🏥 **Medical AI Applications**
- **Alzheimer's Disease Classification**: MRI-based diagnosis with privacy preservation
- **Progressive Learning Phenomenon**: First documented adaptive improvement in medical FL
- **Detection Rate Improvement**: 42.86% → 75.00% (+32.14pp) across training rounds
- **Privacy-Preserving**: Secure collaboration across medical institutions

## OptiGradTrust Architecture

```
OptiGradTrust Framework
├── 📊 Data Distribution & Management
│   ├── Root Dataset (Server-side)
│   ├── Client Datasets (IID/Non-IID)
│   ├── Label Skew Distribution (70%, 90%)
│   └── Dirichlet Distribution (α=0.5, α=0.1)
├── 🔍 Six-Dimensional Gradient Fingerprinting
│   ├── VAE Reconstruction Error Analysis
│   ├── Cosine Similarity Computation (Root & Client)
│   ├── L2 Norm Gradient Analysis
│   ├── Sign-Consistency Ratio Evaluation
│   ├── Monte Carlo Shapley Value Computation
│   └── Multi-dimensional Feature Vector Generation
├── 🧠 Hybrid RL-Attention Module
│   ├── Transformer-Based Attention Mechanism
│   ├── Reinforcement Learning Actor-Critic
│   ├── Dynamic Trust Score Computation
│   ├── Adaptive Threshold Detection
│   └── Progressive Learning Enhancement
├── ⚡ FedBN-Prox (FedBN-P) Optimization
│   ├── Federated Batch Normalization
│   ├── Proximal Regularization Integration
│   ├── Convergence Stability Enhancement
│   └── Non-IID Performance Optimization
├── 🔐 Byzantine-Robust Aggregation
│   ├── Trust-Weighted Parameter Averaging
│   ├── Malicious Client Penalty System
│   ├── Multi-Method Aggregation Support
│   └── Attack-Resilient Model Updates
└── 📈 Comprehensive Analysis & Evaluation
    ├── Multi-Domain Performance Metrics
    ├── Statistical Significance Testing
    ├── Progressive Learning Documentation
    ├── Comparative State-of-the-Art Analysis
    └── Medical AI Application Results
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone the OptiGradTrust repository
git clone <repository-url>
cd new_paper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation and core functionality
python main.py --test-mode

# Run comprehensive system validation
python -m federated_learning.test_basic
```

## Quick Start

### Basic Usage
```bash
# Run OptiGradTrust with default configuration (MNIST + CNN + FedBN-P)
python main.py

# Run with different datasets and configurations
python main.py --dataset CIFAR10 --model RESNET18 --aggregation fedbn_prox
python main.py --dataset ALZHEIMER --model CNN --attack_type scaling

# Run comprehensive OptiGradTrust evaluation
python run_fltrust_experiments.py

# Generate comparison plots and analysis
python create_plots.py
```

### Configuration Examples

#### OptiGradTrust High Security Setup
```python
# config/config_optimized.py
DATASET = 'MNIST'
MODEL = 'CNN'
AGGREGATION_METHOD = 'fedbn_prox'  # FedBN-Prox optimization
FRACTION_MALICIOUS = 0.4
MALICIOUS_PENALTY_FACTOR = 0.98
ENABLE_DUAL_ATTENTION = True
ENABLE_RL_ATTENTION = True  # Hybrid RL-Attention module
ATTACK_TYPE = 'partial_scaling_attack'
SHAPLEY_SAMPLES = 10  # Enhanced Shapley value computation
```

#### Medical AI Research Setup (Alzheimer's)
```python
# config/config_noniid_alzheimer.py
DATASET = 'ALZHEIMER'
MODEL = 'RESNET18'
AGGREGATION_METHOD = 'fedbn_prox'  # FedBN-P for medical data
RL_AGGREGATION_METHOD = 'hybrid'
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
GLOBAL_EPOCHS = 15
NON_IID_ALPHA = 0.5  # Dirichlet distribution
ENABLE_PROGRESSIVE_LEARNING = True
```

#### State-of-the-Art Comparison Setup
```python
# config/config_comparison.py
DATASET = 'CIFAR10'
MODEL = 'RESNET18'
AGGREGATION_METHOD = 'fedbn_prox'
COMPARISON_METHODS = ['FLTrust', 'FLGuard', 'FedAvg', 'FedProx']
NUM_EXPERIMENTS = 5  # Statistical significance
ENABLE_STATISTICAL_ANALYSIS = True
```

## OptiGradTrust System Components

### 1. **Multi-Domain Data Management**
- **Dataset Support**: MNIST (vision), CIFAR-10 (computer vision), Alzheimer's MRI (medical)
- **Distribution Scenarios**: 
  - IID (Independent and Identically Distributed)
  - Label Skew (70%, 90% non-uniformity)
  - Dirichlet Distribution (α=0.5, α=0.1 for varying heterogeneity)
- **Medical Data Handling**: Privacy-preserving Alzheimer's MRI classification
- **Dynamic Partitioning**: Configurable client data distribution with realistic medical constraints

### 2. **Advanced Model Architecture**
- **CNN**: Optimized convolutional neural network for medical imaging
- **ResNet18**: Deep residual network for complex pattern recognition
- **VAE**: Variational autoencoder for sophisticated gradient anomaly detection
- **Hybrid RL-Attention**: Novel transformer-based detection with reinforcement learning integration
- **FedBN-Prox**: Our innovative federated batch normalization with proximal regularization

### 3. **Byzantine Attack Framework**
```python
# OptiGradTrust Evaluated Attack Types
ATTACK_TYPES = [
    'scaling',               # Gradient scaling attack (multiplicative)
    'partial_scaling',       # Selective gradient component scaling
    'sign_flipping',         # Gradient direction reversal
    'gaussian_noise',        # Additive Gaussian noise injection
    'label_flipping'         # Training label manipulation
]

# Advanced Attack Configurations
ATTACK_INTENSITIES = {
    'scaling': [5.0, 10.0, 20.0],           # Scaling factors
    'partial_scaling': [0.2, 0.4, 0.6],     # Fraction of gradients
    'gaussian_noise': [0.1, 0.5, 1.0],      # Noise variance
    'label_flipping': [0.1, 0.2, 0.4]       # Flip probability
}
```

### 4. **OptiGradTrust Detection Pipeline**

#### Six-Dimensional Gradient Fingerprinting
1. **VAE Reconstruction Error**: Advanced anomaly detection using variational autoencoders trained on root gradients
2. **Root Similarity**: Cosine similarity between client gradients and server root dataset gradients
3. **Client Similarity**: Inter-client gradient cosine similarity for collective behavior analysis
4. **L2 Norm Analysis**: Gradient magnitude consistency evaluation across training rounds
5. **Sign-Consistency Ratio**: Proportion of gradient components with consistent directional alignment
6. **Monte Carlo Shapley Value**: Game-theoretic client contribution analysis with statistical sampling

#### Hybrid RL-Attention Trust Computation
```python
# OptiGradTrust Hybrid RL-Attention Pipeline
# Step 1: Multi-head attention processing
attention_scores = multi_head_attention(six_dim_features)

# Step 2: Reinforcement learning actor-critic evaluation
rl_trust_weights = actor_critic_network(attention_scores, historical_performance)

# Step 3: Hybrid trust score integration
trust_scores = hybrid_integration(attention_scores, rl_trust_weights)

# Step 4: Adaptive threshold with progressive learning
threshold = adaptive_threshold(trust_scores, round_num, historical_stats)
detected_malicious = trust_scores < threshold
```

#### FedBN-Prox Aggregation Framework
```python
# OptiGradTrust FedBN-Prox Aggregation
# Step 1: Trust-weighted parameter updates
weighted_updates = trust_scores * client_updates

# Step 2: Federated Batch Normalization separation
bn_params = separate_batch_norm_params(weighted_updates)
model_params = extract_model_params(weighted_updates)

# Step 3: Proximal regularization for stability
prox_term = FEDPROX_MU * ||client_params - global_params||²

# Step 4: Combined FedBN-P update
global_model = fedbn_prox_aggregation(model_params, bn_params, prox_term)
```

## OptiGradTrust Evaluation Metrics

### **Byzantine Detection Performance**
- **Precision**: TP / (TP + FP) - Accuracy of malicious client identification
- **Recall**: TP / (TP + FN) - Coverage of actual Byzantine clients
- **F1-Score**: Harmonic mean of precision and recall for balanced evaluation
- **Progressive Learning Rate**: Improvement in detection over training rounds
- **Trust Score Distribution**: Statistical separation between honest and malicious clients

### **Multi-Domain Model Performance**
- **Global Accuracy**: Test dataset performance across MNIST, CIFAR-10, Alzheimer's
- **Convergence Stability**: FedBN-Prox optimization effectiveness
- **Non-IID Robustness**: Performance under label skew and Dirichlet distributions
- **Medical AI Accuracy**: Alzheimer's disease classification precision

### **Comparative Analysis**
- **State-of-the-Art Comparison**: Performance vs. FLTrust, FLGuard, baseline methods
- **Statistical Significance**: p-value analysis with Cohen's d effect sizes
- **Attack Resilience**: Performance degradation under various Byzantine scenarios
- **Computational Efficiency**: Overhead analysis of six-dimensional fingerprinting

### **Medical AI Specific Metrics**
- **Progressive Learning Phenomenon**: 42.86% → 75.00% detection rate improvement
- **Privacy Preservation**: Differential privacy guarantees in medical federated learning
- **Clinical Relevance**: Real-world applicability in healthcare scenarios

## OptiGradTrust Configuration Guide

### Core Parameters
```python
# OptiGradTrust Client Configuration
NUM_CLIENTS = 10                   # Federated learning participants
FRACTION_MALICIOUS = 0.3           # Byzantine client proportion
CLIENT_SELECTION_RATIO = 1.0       # Clients selected per round

# Training & Optimization Parameters
GLOBAL_EPOCHS = 15                 # Federated learning rounds
LOCAL_EPOCHS_CLIENT = 5            # Local training epochs
BATCH_SIZE = 64                    # Training batch size
LR = 0.01                         # Learning rate
FEDPROX_MU = 0.1                  # FedBN-Prox proximal term

# OptiGradTrust Security Framework
AGGREGATION_METHOD = 'fedbn_prox'  # FedBN-Prox optimization
ENABLE_DUAL_ATTENTION = True       # Hybrid RL-Attention module
ENABLE_RL_ATTENTION = True         # Reinforcement learning integration
SHAPLEY_NUM_SAMPLES = 10          # Monte Carlo Shapley sampling
MALICIOUS_PENALTY_FACTOR = 0.98   # Trust-based penalty strength

# Six-Dimensional Fingerprinting
VAE_LATENT_DIM = 64               # VAE reconstruction dimension
ATTENTION_HEADS = 8               # Multi-head attention
RL_WARMUP_ROUNDS = 3              # RL training initialization
```

### Advanced OptiGradTrust Configuration
```python
# Hybrid RL-Attention Architecture
RL_AGGREGATION_METHOD = 'hybrid'   # OptiGradTrust hybrid approach
RL_WARMUP_ROUNDS = 3              # RL system initialization
RL_RAMP_UP_ROUNDS = 8             # Progressive learning integration
ACTOR_CRITIC_LR = 0.001           # RL learning rate

# Enhanced VAE Configuration
VAE_EPOCHS = 10                   # Extended VAE training
VAE_LATENT_DIM = 64               # Enhanced latent representation
VAE_PROJECTION_DIM = 256          # High-dimensional gradient projection
VAE_BETA = 1.0                    # Beta-VAE regularization

# Multi-Head Attention Architecture
ATTENTION_HIDDEN_SIZE = 128       # Enhanced hidden layer capacity
ATTENTION_HEADS = 8               # Multi-head attention mechanism
ATTENTION_LAYERS = 4              # Deep transformer architecture
ATTENTION_DROPOUT = 0.1           # Regularization for generalization

# FedBN-Prox Optimization
FEDBN_MOMENTUM = 0.9              # Batch normalization momentum
FEDPROX_MU = 0.1                  # Proximal regularization strength
CONVERGENCE_TOLERANCE = 1e-6      # Optimization convergence criteria
```

## Testing Framework

### Quick Validation
```bash
# Test basic trust score logic
python quick_test.py

# Expected output:
# ✅ PASS: Honest clients have higher trust scores
# ✅ PASS: Honest clients have higher aggregation weights
# ✅ OVERALL: Trust score logic is working correctly!
```

### Comprehensive Testing
```bash
# Test all configurations and attack types
python test_comprehensive_fix.py

# Tests include:
# - Trust score logic validation
# - Model configuration testing (CNN, ResNet)
# - Aggregation method testing (FedAvg, FedBN, FedProx)
# - Attack detection across all attack types
# - Client configuration variations
```

### Experiment Runner
```bash
# Run predefined experiment configurations
python run_experiments.py

# Custom experiment configuration
python run_experiments.py --config custom_config.json --output results/custom/
```

## Results Organization

```
results/
├── experiments/          # Individual experiment results
│   ├── mnist_cnn_fedbn_partial_scaling_20241203_140530/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   ├── model.pth
│   │   └── plots/
├── metrics/             # Aggregated metrics
├── models/              # Saved model checkpoints
├── plots/               # Visualization outputs
└── configs/             # Configuration files
```

### Results Analysis
```python
# Load experiment results
with open('results/experiments/exp_name/metrics.json') as f:
    metrics = json.load(f)

# Key metrics available:
metrics['detection_precision']     # Malicious client detection accuracy
metrics['detection_recall']        # Coverage of malicious clients
metrics['detection_f1_score']     # Overall detection performance
metrics['final_accuracy']         # Model performance
metrics['trust_scores']           # Per-client trust scores
```

## Research Applications

### Security Research
- **Attack Development**: Framework for testing new attack vectors
- **Defense Evaluation**: Comprehensive testing of detection mechanisms
- **Robustness Analysis**: Performance under various threat models

### Federated Learning Research
- **Aggregation Algorithms**: Testing ground for new aggregation methods
- **Non-IID Scenarios**: Evaluation under realistic data distributions
- **Scalability Studies**: Performance analysis with varying client numbers

### Machine Learning Research
- **Attention Mechanisms**: Novel application of transformers to FL security
- **Multi-modal Analysis**: Integration of multiple detection signals
- **Shapley Value Applications**: Game-theoretic approach to client evaluation

## Troubleshooting

### Common Issues

#### GPU Memory Issues
```bash
# Reduce batch size in config
BATCH_SIZE = 32
VAE_BATCH_SIZE = 16

# Enable memory optimization
ENABLE_MEMORY_OPTIMIZATION = True
```

#### Detection Performance Issues
```python
# Increase training data diversity
VAE_EPOCHS = 10
DUAL_ATTENTION_EPOCHS = 10

# Adjust detection sensitivity
MALICIOUS_PENALTY_FACTOR = 0.95  # More lenient
MALICIOUS_PENALTY_FACTOR = 0.99  # More strict
```

#### Convergence Issues
```python
# Adjust learning parameters
LR = 0.001                    # Lower learning rate
LR_DECAY = 0.99              # More aggressive decay
FEDPROX_MU = 0.1             # Increase proximal term
```

### Debugging Tools
```bash
# Verbose output with detailed logging
python main.py --verbose

# Debug specific components
python -c "
from federated_learning.models.attention import DualAttention
model = DualAttention(feature_dim=6)
print('Model loaded successfully')
"

# Memory profiling
python -m memory_profiler main.py
```

## Citation

If you use OptiGradTrust in your research, please cite:

```bibtex
@article{optigradtrust2024,
  title={OptiGradTrust: Byzantine-Robust Federated Learning with Multi-Feature Gradient Analysis and Reinforcement Learning-Based Trust Weighting},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]},
  url={[Repository URL]},
  keywords={Federated Learning, Byzantine Attacks, Reinforcement Learning, Non-IID Distribution, Medical Applications, Gradient Fingerprinting, Trust Weighting, Robust Aggregation}
}
```

### Key Research Contributions

**OptiGradTrust** introduces several novel contributions to Byzantine-robust federated learning:

1. **Six-Dimensional Gradient Fingerprinting**: First comprehensive gradient analysis framework combining VAE reconstruction, cosine similarity, L2 norm, sign-consistency, and Monte Carlo Shapley values.

2. **Hybrid RL-Attention Module**: Novel integration of reinforcement learning with transformer-based attention for adaptive trust scoring.

3. **FedBN-Prox (FedBN-P)**: Innovative combination of Federated Batch Normalization with proximal regularization for optimal convergence under data heterogeneity.

4. **Progressive Learning in Medical FL**: First documented phenomenon of adaptive improvement in Byzantine detection capabilities specifically in medical federated learning scenarios.

5. **Multi-Domain Evaluation**: Comprehensive assessment across vision (MNIST), computer vision (CIFAR-10), and medical imaging (Alzheimer's MRI) domains.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Federated learning research community
- Contributors to the attention mechanism literature

## Contact

For questions, issues, or collaboration opportunities regarding OptiGradTrust:
- **GitHub Issues**: Create an issue for technical questions and bug reports
- **Research Collaboration**: [mohammad.karami79@ut.ac.ir]
- **Medical AI Applications**: For healthcare and medical federated learning applications

---

## Status & Acknowledgments

**Status**: ✅ **OptiGradTrust fully implemented and tested** across multiple domains and attack scenarios.

**Performance Achievements**:
- 🎯 **+1.6pp improvement** over FLGuard under non-IID conditions
- 🏥 **Progressive learning** documented in medical federated learning
- 🛡️ **Superior Byzantine robustness** compared to state-of-the-art methods
- 📊 **Multi-domain validation** across vision, computer vision, and medical imaging

**Last Updated**: August 2024  
**Version**: 3.0.0 - OptiGradTrust Release  
**Paper Status**: Ready for Journal Submission 
