# Federated Learning with Dual Attention-Based Malicious Client Detection

## Overview

This repository implements a comprehensive federated learning system with advanced malicious client detection using a **dual attention mechanism**. The system combines multiple techniques including VAE-based gradient analysis, Shapley value computation, and transformer-based attention models to identify and mitigate the impact of malicious clients in federated learning environments.

## Key Features

### 🛡️ Advanced Malicious Client Detection
- **Dual Attention Mechanism**: Transformer-based architecture for analyzing client behavior
- **Multi-Feature Analysis**: 6-dimensional feature extraction including:
  - VAE reconstruction error
  - Root gradient similarity 
  - Client gradient similarity
  - Gradient norm analysis
  - Sign consistency checking
  - Shapley value computation
- **Adaptive Thresholding**: Dynamic detection thresholds based on score distributions
- **Real-time Trust Scoring**: Continuous assessment of client trustworthiness

### 🔬 Comprehensive Attack Simulation
- **7 Attack Types**: scaling, partial_scaling, sign_flipping, noise_injection, min_max, min_sum, targeted
- **Realistic Attack Patterns**: Sophisticated attack simulation for robust testing
- **Attack Impact Analysis**: Detailed metrics on attack effectiveness and detection

### 🔄 Multiple Aggregation Methods
- **FedAvg**: Standard federated averaging
- **FedBN**: Batch normalization parameter separation
- **FedProx**: Proximal term for stable convergence
- **FedADMM**: Alternating Direction Method of Multipliers
- **RL-based**: Reinforcement learning guided aggregation
- **Hybrid**: Adaptive combination of multiple methods

### 📊 Comprehensive Evaluation
- **Detection Metrics**: Precision, Recall, F1-Score for malicious client detection
- **Model Performance**: Accuracy tracking across federated rounds
- **Trust Score Analysis**: Detailed trust score distributions and patterns
- **Results Management**: Organized experiment storage and comparison tools

## Architecture

```
Federated Learning System
├── Data Collection & Distribution
│   ├── Root Dataset (Server)
│   └── Client Datasets (Non-IID)
├── Feature Extraction Pipeline
│   ├── VAE Training (Root Gradients)
│   ├── Gradient Feature Extraction
│   ├── Shapley Value Computation
│   └── Multi-dimensional Feature Vector
├── Dual Attention Detection
│   ├── Transformer Architecture
│   ├── Trust Score Computation
│   └── Adaptive Threshold Detection
├── Secure Aggregation
│   ├── Trust-weighted Averaging
│   ├── Malicious Client Penalties
│   └── Multiple Aggregation Algorithms
└── Results & Analysis
    ├── Detection Performance Metrics
    ├── Model Accuracy Tracking
    └── Comprehensive Reporting
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd federated_learning_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python quick_test.py
```

## Quick Start

### Basic Usage
```bash
# Run with default configuration (MNIST + CNN + FedBN)
python main.py

# Run comprehensive tests
python test_comprehensive_fix.py

# Run experiments with different configurations
python run_experiments.py
```

### Configuration Examples

#### High Security Setup
```python
# config/config.py
DATASET = 'MNIST'
MODEL = 'CNN'
AGGREGATION_METHOD = 'fedbn'
FRACTION_MALICIOUS = 0.4
MALICIOUS_PENALTY_FACTOR = 0.98
ENABLE_DUAL_ATTENTION = True
ATTACK_TYPE = 'partial_scaling_attack'
```

#### Research Evaluation Setup
```python
# config/config.py
DATASET = 'ALZHEIMER'
MODEL = 'RESNET18'
AGGREGATION_METHOD = 'hybrid'
RL_AGGREGATION_METHOD = 'hybrid'
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.3
GLOBAL_EPOCHS = 10
```

## System Components

### 1. Data Management
- **Multi-dataset Support**: MNIST, CIFAR-10, Alzheimer's disease classification
- **Non-IID Distribution**: Label skew and Dirichlet distribution
- **Dynamic Dataset Sizing**: Configurable client data distribution

### 2. Model Architecture
- **CNN**: Lightweight convolutional neural network
- **ResNet18/50**: Pre-trained models with configurable fine-tuning
- **VAE**: Variational autoencoder for gradient analysis
- **Dual Attention**: Transformer-based malicious client detector

### 3. Attack Framework
```python
# Available attack types
ATTACK_TYPES = [
    'scaling_attack',        # Scale gradients by factor
    'partial_scaling_attack', # Scale subset of gradients  
    'sign_flipping',         # Flip gradient signs
    'noise_injection',       # Add Gaussian noise
    'min_max',              # Minimize/maximize different classes
    'min_sum',              # Minimize total loss
    'targeted'              # Target specific parameters
]
```

### 4. Detection Pipeline

#### Feature Extraction (6D Vector)
1. **VAE Reconstruction Error**: Measures gradient anomaly
2. **Root Similarity**: Cosine similarity to server gradients
3. **Client Similarity**: Average similarity to other clients
4. **Gradient Norm**: L2 norm of gradient vector
5. **Sign Consistency**: Proportion of consistent gradient signs
6. **Shapley Value**: Contribution-based client valuation

#### Trust Score Computation
```python
# Dual attention forward pass
malicious_scores, confidence = dual_attention(features)
trust_scores = 1.0 - malicious_scores

# Adaptive threshold detection
threshold = np.clip(mean + 0.5 * std, 0.4, 0.8)
detected_malicious = malicious_scores > threshold
```

#### Aggregation Weighting
```python
# Trust-weighted aggregation
weights = trust_scores.clone()
for malicious_client in detected_malicious:
    penalty = MALICIOUS_PENALTY_FACTOR * severity
    weights[malicious_client] *= (1 - penalty)

# Normalize weights
weights = weights / weights.sum()
```

## Evaluation Metrics

### Detection Performance
- **Precision**: TP / (TP + FP) - Accuracy of malicious detection
- **Recall**: TP / (TP + FN) - Coverage of actual malicious clients  
- **F1-Score**: Harmonic mean of precision and recall
- **Trust Score Analysis**: Distribution patterns and separation

### Model Performance  
- **Global Accuracy**: Performance on test dataset
- **Convergence Rate**: Rounds to reach target accuracy
- **Robustness**: Performance degradation under attacks

### System Analysis
- **Attack Impact**: Effectiveness of different attack types
- **Detection Latency**: Time to identify malicious clients
- **Computational Overhead**: Additional cost of security measures

## Configuration Guide

### Core Parameters
```python
# Client Setup
NUM_CLIENTS = 5                    # Total number of clients
FRACTION_MALICIOUS = 0.4           # Proportion of malicious clients
CLIENT_SELECTION_RATIO = 1.0       # Fraction selected per round

# Training Parameters  
GLOBAL_EPOCHS = 10                 # Federated learning rounds
LOCAL_EPOCHS_CLIENT = 2            # Local training epochs
BATCH_SIZE = 64                    # Training batch size
LR = 0.01                         # Learning rate

# Security Parameters
MALICIOUS_PENALTY_FACTOR = 0.98    # Penalty strength (0-1)
ENABLE_DUAL_ATTENTION = True       # Enable detection system
SHAPLEY_NUM_SAMPLES = 5           # Shapley value samples

# Attack Configuration
ATTACK_TYPE = 'partial_scaling_attack'
SCALING_FACTOR = 20.0             # Attack intensity
PARTIAL_SCALING_PERCENT = 0.4     # Fraction of gradients to attack
```

### Advanced Configuration
```python
# RL-based Aggregation
RL_AGGREGATION_METHOD = 'hybrid'   # 'dual_attention', 'rl_actor_critic', 'hybrid'
RL_WARMUP_ROUNDS = 5              # Rounds before RL activation
RL_RAMP_UP_ROUNDS = 10            # Gradual transition period

# VAE Parameters
VAE_EPOCHS = 5                    # VAE training epochs
VAE_LATENT_DIM = 32              # Latent space dimension
VAE_PROJECTION_DIM = 128         # Gradient projection size

# Dual Attention Architecture
DUAL_ATTENTION_HIDDEN_SIZE = 64   # Hidden layer size
DUAL_ATTENTION_HEADS = 4          # Number of attention heads
DUAL_ATTENTION_LAYERS = 2         # Transformer layers
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

If you use this code in your research, please cite:

```bibtex
@misc{federated_dual_attention_2024,
  title={Federated Learning with Dual Attention-Based Malicious Client Detection},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

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

For questions, issues, or collaboration opportunities:
- Create an issue on GitHub
- Contact: [Your Email]
- Documentation: [Wiki/Docs URL]

---

**Status**: ✅ All systems functional and tested across multiple configurations.
**Last Updated**: December 2024
**Version**: 2.0.0 