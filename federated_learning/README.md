# Federated Learning with Dual Attention and VAE-based Defense

This project implements a federated learning system with defense mechanisms against malicious clients using Dual Attention and VAE-based anomaly detection.

## Project Structure

```
federated_learning/
├── config/
│   └── config.py           # Configuration parameters
├── models/
│   ├── cnn.py              # CNN model for MNIST
│   ├── resnet.py           # ResNet models for ALZHEIMER
│   ├── vae.py              # Gradient VAE model
│   ├── attention.py        # Dual Attention mechanism
│   └── dimension_reducer.py # Gradient dimensionality reduction
├── data/
│   └── dataset.py          # Dataset handling and preprocessing
├── attacks/
│   └── attack_utils.py     # Attack simulation utilities
├── privacy/
│   ├── __init__.py         # Privacy module initialization
│   ├── privacy_utils.py    # General privacy utilities
│   ├── differential_privacy.py # Differential privacy implementation
│   └── homomorphic_encryption.py # Homomorphic encryption implementation
├── utils/
│   ├── gradient_features.py # Gradient feature extraction utilities
│   ├── model_utils.py       # Model update utilities
│   └── shapley_utils.py     # Shapley value calculation utilities
├── training/
│   ├── client.py           # Client implementation
│   ├── server.py           # Server implementation
│   ├── aggregators.py      # Aggregation methods implementation
│   └── training_utils.py   # Training utilities
└── main.py                 # Main script
```

## Modules

### config/
Contains all configuration parameters for the federated learning system, including:
- Number of clients
- Learning rates
- Epochs
- Attack types
- Dataset parameters
- Privacy settings
- Memory management settings

### models/
Contains the neural network models:
- `cnn.py`: CNN model for MNIST classification
- `resnet.py`: ResNet models for Alzheimer's disease classification
- `vae.py`: Gradient VAE for anomaly detection
- `attention.py`: Dual Attention mechanism for trust scoring
- `dimension_reducer.py`: PCA-based gradient dimensionality reduction

### data/
Handles data loading and preprocessing:
- Dataset loading
- Non-IID data splitting
- Root dataset creation
- Attack dataset implementations

### attacks/
Contains attack simulation utilities:
- Label flipping
- Scaling attacks
- Partial scaling attacks
- Backdoor attacks
- Adaptive attacks
- Min-max and min-sum attacks
- Alternating, targeted, and gradient inversion attacks

### privacy/
Implements privacy-preserving mechanisms:
- `privacy_utils.py`: Framework for applying privacy mechanisms
- `differential_privacy.py`: Implementation of DP-SGD with gradient clipping and noise addition
- `homomorphic_encryption.py`: Paillier cryptosystem for homomorphic encryption of gradients

### utils/
Contains utility functions for the federated learning system:
- `gradient_features.py`: Standardized feature extraction from gradients
- `model_utils.py`: Model update functionality including FedProx implementation
- `shapley_utils.py`: Shapley value calculation for measuring client contributions
- `README_SHAPLEY.md`: Detailed documentation of the Shapley value implementation

### training/
Contains the core federated learning implementation:
- `client.py`: Client-side training and gradient computation
- `server.py`: Server-side aggregation and defense mechanisms
- `aggregators.py`: Various aggregation methods including FedAvg, FedProx, and FedBN
- `training_utils.py`: Utility functions for training

## Malicious Client Detection Architecture

### VAE-based Anomaly Detection

The system employs a Variational Autoencoder (VAE) to detect anomalous gradients that may indicate malicious activity:

1. **Memory-Efficient Architecture**:
   - Uses HashProjection for handling large gradient dimensions
   - Automatically scales projection dimensions based on model size
   - Employs BatchNorm and dropout for regularization

2. **Training Process**:
   - Trained on trusted gradients from root and benign clients
   - Uses memory-optimized batch processing with early stopping
   - Monitors reconstruction errors to identify outliers

3. **Anomaly Detection**:
   - Computes reconstruction error for client gradients
   - High reconstruction error indicates potential malicious updates
   - Used as a feature for the Dual Attention model

### Dual Attention Mechanism

The Dual Attention model classifies clients as honest or malicious based on extracted gradient features:

1. **Gradient Feature Extraction**:
   - Reconstruction error from VAE
   - Mean cosine similarity with root gradients
   - Mean cosine similarity with neighbor gradients
   - Norm of the raw gradient
   - Pattern consistency with root gradients
   - Shapley value (contribution to model performance) [if enabled]

2. **Architecture**:
   - Sequential and context encoders for feature processing
   - Self-attention mechanism to focus on important features
   - Prior knowledge embedding for known attack patterns
   - Comparison layers for decision making

3. **Trust Scoring**:
   - Outputs a trust score (0-1) for each client
   - Higher scores indicate higher likelihood of being honest
   - Scores are used to weight client contributions
   - Can be blended with Shapley values for contribution-aware weighting

### Shapley Value Client Contribution Measurement

The system uses Shapley values to measure each client's contribution to the global model:

1. **Implementation**:
   - Monte Carlo-based approximation of Shapley values
   - Measures marginal contribution of each client to model performance
   - Provides a principled, game-theoretic approach to credit assignment

2. **Integration with Defense Mechanisms**:
   - Added as a 6th feature for the Dual Attention mechanism
   - Used to enhance client weighting during gradient aggregation
   - Helps identify clients with low contributions (potentially malicious)
   
3. **Configuration**:
   - Can be enabled/disabled via the `ENABLE_SHAPLEY` parameter
   - Sample size controlled by `SHAPLEY_SAMPLES` parameter
   - Influence on final weights controlled by `SHAPLEY_WEIGHT`

## Memory Management

The system implements several memory optimization techniques:

1. **Gradient Chunking**:
   - Processes gradients in small chunks to reduce memory usage
   - Aggregates chunks using configurable methods (mean, sum, or last)

2. **Device Management**:
   - Configurable VAE device placement (GPU, CPU, or auto)
   - Optimizes tensor placement based on available resources

3. **Dimension Reduction**:
   - PCA-based gradient dimensionality reduction
   - Configurable reduction ratio (e.g., reducing to 10% of original size)
   - Automatically fits to the first gradient and transforms subsequent ones

4. **Memory Cleanup**:
   - Aggressive memory cleanup with `torch.cuda.empty_cache()`
   - Process memory tracking for monitoring resource usage
   - Explicit deletion of tensors after use

## Privacy Mechanisms

The system supports three privacy mechanisms:

1. **None**: No privacy mechanism applied
2. **Differential Privacy (DP)**:
   - Gradient clipping to bound sensitivity
   - Gaussian noise addition calibrated to (ε, δ)-DP
   - Configurable privacy parameters (epsilon, delta, clip norm)

3. **Homomorphic Encryption (Paillier)**:
   - Public-key cryptography for encrypted gradient aggregation
   - Server computes on encrypted gradients without decryption
   - Clients encrypt gradients; server aggregates; client decrypts result

## Federated Learning Workflow

1. **Initialization**:
   - Server creates global model and initializes VAE and Dual Attention models
   - Clients are assigned datasets (some designated as malicious)
   - Root gradients are collected from trusted data

2. **Defense Model Training**:
   - VAE is trained on trusted gradients from root and benign clients
   - Dual Attention is trained on features from benign and simulated malicious gradients
   - Various attack types are simulated to create diverse training data

3. **Federated Learning Process**:
   - For each communication round:
     - Server selects a subset of clients for training
     - Selected clients train on local data and send gradient updates
     - Server extracts features from client gradients
     - Dual Attention evaluates trust scores for each client
     - Updates are weighted based on trust scores
     - Weighted updates are aggregated and applied to the global model

4. **Evaluation and Metrics**:
   - The system tracks:
     - Test error on the global model
     - True positive rate for malicious client detection
     - False positive rate for honest clients
     - Precision of malicious client detection

## Usage

1. Install dependencies:
```bash
pip install torch torchvision numpy phe psutil
```

2. Run the federated learning system:
```bash
python main.py
```

## Configuration

Modify parameters in `config/config.py` to adjust:
- Number of clients
- Fraction of malicious clients
- Learning rates
- Number of epochs
- Attack type
- Dataset parameters
- Privacy settings
- Memory management settings

## Training Configuration

The training process uses the following epoch configurations:

- **COMMUNICATION_ROUNDS**: Number of global training rounds (default: 15)
- **LOCAL_EPOCHS_CLIENT**: Number of epochs for training all clients (benign and malicious) in the regular federated training (default: 5)
- **LOCAL_EPOCHS_ROOT**: Number of epochs for training the trusted root client (default: 10)
- **MALICIOUS_DA_EPOCHS**: Number of epochs used ONLY for collecting malicious gradients during Dual Attention training (default: 30). This is not used in regular federated updates.
- **BENIGN_DA_EPOCHS**: Number of epochs used ONLY for collecting benign gradients during Dual Attention training (default: 20). This is not used in regular federated updates.

During the training of the detection models:
- The VAE is trained on both root gradients and benign client gradients
- The Dual Attention model is trained on:
  - Root client gradients (labeled as benign)
  - Benign client gradients
  - Simulated malicious gradients with various attack types

All these parameters can be configured in `federated_learning/config/config.py`.

## Development

The modular structure allows for easy development and testing of individual components:

1. Models can be modified independently in the `models/` directory
2. New attack types can be added in `attacks/attack_utils.py`
3. Data handling can be modified in `data/dataset.py`
4. Training procedures can be adjusted in the `training/` directory
5. Privacy mechanisms can be extended in the `privacy/` directory

## Team Collaboration

The modular structure enables parallel development:
1. Model developers can work on the `models/` directory
2. Security researchers can focus on the `attacks/` directory
3. Data engineers can work on the `data/` directory
4. Training algorithm developers can work on the `training/` directory
5. Privacy specialists can work on the `privacy/` directory

Each team can work independently while maintaining a consistent interface between modules.

## Attack Types

This implementation supports the following attack types:

1. **None**: No attack (honest client)
2. **Label Flipping**: Inverts gradient direction by flipping labels during training
3. **Scaling Attack**: Scales gradient by number of clients to dominate aggregation
4. **Partial Scaling Attack**: Scales a subset (~66%) of gradient elements
5. **Backdoor Attack**: Adds a constant value to the gradient
6. **Adaptive Attack**: Adds random noise to gradient proportional to its norm
7. **Min-Max Attack**: Normalizes and inverts gradient, scales by client count
8. **Min-Sum Attack**: Minimizes cosine similarity with benign gradients
9. **Alternating Attack**: Creates oscillating pattern of positive/negative values
10. **Targeted Attack**: Targets specific subset of parameters
11. **Gradient Inversion Attack**: Applies different inversion strategies to different parts of the gradient

The attack type can be configured in the `federated_learning/config/config.py` file by setting the `ATTACK_TYPE` variable.

## Aggregation Methods

The system supports several aggregation methods:

1. **FedAvg**: Standard federated averaging of client updates
2. **FedProx**: Adds a proximal term to client optimization to limit client drift
3. **FedBN**: Keeps batch normalization parameters local to clients
4. **FedBN_FedProx**: Combines FedBN and FedProx approaches

The aggregation method can be configured by setting the `AGGREGATION_METHOD` variable in `config.py`.

## Update Note

This version has been updated with:
1. Privacy mechanisms (Differential Privacy and Homomorphic Encryption)
2. Support for multiple aggregation methods (FedAvg, FedProx, FedBN, FedBN_FedProx)
3. Memory management optimizations for handling large models
4. Improved malicious client detection with VAE and Dual Attention
5. Comprehensive gradient feature extraction for trust evaluation
6. Shapley value implementation for measuring client contributions to model performance 