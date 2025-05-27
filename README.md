# Federated Learning with Dual Attention and Reinforcement Learning for Robust Aggregation

## Abstract

This project implements a novel federated learning system that combines dual attention mechanisms and reinforcement learning to create a robust defense against malicious clients while maintaining high performance on non-IID data. The system features advanced aggregation methods, comprehensive attack detection, and efficient memory management for large-scale federated learning.

## Key Innovations

1. **Hybrid Aggregation Architecture**
   - Dual attention mechanism for trust scoring
   - Reinforcement learning for adaptive aggregation
   - Smooth transition between methods during training

2. **Advanced Defense Mechanisms**
   - VAE-based gradient anomaly detection
   - Feature-rich gradient analysis (6 key features)
   - Shapley value integration for contribution measurement

3. **Memory-Efficient Implementation**
   - Gradient chunking for large models
   - Dynamic dimension reduction
   - Optimized tensor management

4. **Multi-Model Support**
   - CNN for lightweight tasks
   - ResNet18/50 for complex tasks
   - BatchNorm optimization for federated setting

## System Architecture

### Core Components

1. **Server**
   - Global model management
   - Client selection and aggregation
   - Trust evaluation and defense mechanisms

2. **Clients**
   - Local training
   - Gradient computation
   - Attack simulation (for malicious clients)

3. **Defense Mechanisms**
   - VAE for anomaly detection
   - Dual attention for trust scoring
   - RL actor-critic for adaptive aggregation

### Aggregation Methods

1. **FedAvg**: Standard federated averaging
2. **FedBN**: BatchNorm-aware aggregation
3. **FedProx**: Proximal term regularization
4. **FedADMM**: ADMM-based optimization
5. **Hybrid**: RL-based adaptive aggregation

## Features

### Attack Types Supported

1. **Gradient-Based Attacks**
   - Scaling attacks (full/partial)
   - Sign flipping attacks
   - Noise injection attacks

2. **Sophisticated Attacks**
   - Min-max attacks
   - Targeted model poisoning
   - Backdoor attacks

3. **Data-Based Attacks**
   - Label flipping
   - Data poisoning

### Defense Mechanisms

1. **Gradient Feature Analysis**
   - VAE reconstruction error
   - Root gradient similarity
   - Client similarity
   - Gradient norm analysis
   - Pattern consistency
   - Shapley value (optional)

2. **Trust Scoring**
   - Dual attention mechanism
   - Historical performance tracking
   - Adaptive thresholding

3. **Aggregation Defense**
   - Trust-weighted aggregation
   - Dynamic client filtering
   - Adaptive learning rates

## Datasets Supported

1. **MNIST**
   - 60,000 training samples
   - 10,000 test samples
   - 10 classes
   - Single channel images

2. **ALZHEIMER**
   - Brain MRI scans
   - 4 impairment classes
   - Multi-channel images
   - High-resolution data

3. **CIFAR10**
   - 50,000 training samples
   - 10,000 test samples
   - 10 classes
   - RGB images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-learning.git
cd federated-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from federated_learning.main import main

# Run with default configuration
main()
```

### Custom Configuration

```python
# In config.py
DATASET = 'ALZHEIMER'
MODEL = 'RESNET18'
NUM_CLIENTS = 10
FRACTION_MALICIOUS = 0.2
ATTACK_TYPE = 'scaling_attack'
AGGREGATION_METHOD = 'fedbn'
```

### Running Experiments

```bash
# Run basic training
python main.py

# Run with custom parameters
python main.py --dataset ALZHEIMER --model RESNET18 --num_clients 10

# Run comprehensive validation
python test_complete_validation.py
```

## Performance Metrics

1. **Model Performance**
   - Test accuracy
   - Convergence rate
   - Client contribution analysis

2. **Defense Effectiveness**
   - Attack detection rate
   - False positive rate
   - Trust score distribution

3. **System Efficiency**
   - Memory usage
   - Communication overhead
   - Computation time

## Research Applications

This implementation is particularly suited for research in:

1. **Federated Learning Security**
   - Attack detection mechanisms
   - Defense strategy evaluation
   - Trust system development

2. **Non-IID Data Handling**
   - BatchNorm optimization
   - Client drift management
   - Aggregation method comparison

3. **Medical Image Analysis**
   - Alzheimer's detection
   - Privacy-preserving collaboration
   - Model performance optimization

## Future Directions

1. **Enhanced Privacy**
   - Differential privacy integration
   - Homomorphic encryption support
   - Secure aggregation protocols

2. **Scalability**
   - Cross-silo federation
   - Hierarchical aggregation
   - Dynamic client management

3. **Model Support**
   - Vision Transformers
   - Graph Neural Networks
   - Custom architecture support

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{federated_learning_dual_attention,
  title={Federated Learning with Dual Attention and RL for Robust Aggregation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/federated-learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implementation inspired by various federated learning papers
- Dual attention mechanism based on transformer architectures
- VAE implementation adapted from PyTorch examples 