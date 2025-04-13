# Federated Learning with Dual Attention and VAE-based Defense

This project implements a federated learning system with defense mechanisms against malicious clients using Dual Attention and VAE-based anomaly detection.

## Project Structure

```
federated_learning/
├── config/
│   └── config.py           # Configuration parameters
├── models/
│   ├── cnn.py             # CNN model for MNIST
│   ├── vae.py             # Gradient VAE model
│   └── attention.py       # Dual Attention mechanism
├── data/
│   └── dataset.py         # Dataset handling and preprocessing
├── attacks/
│   └── attack_utils.py    # Attack simulation utilities
├── training/
│   ├── client.py          # Client implementation
│   ├── server.py          # Server implementation
│   └── training_utils.py  # Training utilities
└── main.py                # Main script
```

## Modules

### config/
Contains all configuration parameters for the federated learning system, including:
- Number of clients
- Learning rates
- Epochs
- Attack types
- Dataset parameters

### models/
Contains the neural network models:
- `cnn.py`: CNN model for MNIST classification
- `vae.py`: Gradient VAE for anomaly detection
- `attention.py`: Dual Attention mechanism for trust scoring

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

### training/
Contains the core federated learning implementation:
- `client.py`: Client-side training and gradient computation
- `server.py`: Server-side aggregation and defense mechanisms
- `training_utils.py`: Utility functions for training

## Usage

1. Install dependencies:
```bash
pip install torch torchvision numpy
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

## Training Configuration

The training process uses the following epoch configurations:

- **LOCAL_EPOCHS_CLIENT**: Number of epochs for training all clients (benign and malicious) in the regular federated training (default: 10)
- **LOCAL_EPOCHS_ROOT**: Number of epochs for training the trusted root client (default: 200)
- **MALICIOUS_EPOCHS**: Number of epochs used ONLY for collecting malicious gradients during Dual Attention training (default: 30). This is not used in regular federated updates.
- **BENIGN_DA_EPOCHS**: Number of epochs used ONLY for collecting benign gradients during Dual Attention training (default: 20). This is not used in regular federated updates.

During the training of the detection models:
- The VAE is trained on both root gradients (200) and benign client gradients (80), totaling 280 gradients
- The Dual Attention model is trained on:
  - 200 root client gradients (labeled as benign)
  - 160 benign client gradients (BENIGN_DA_EPOCHS × 8 clients)
  - 600 malicious gradients (MALICIOUS_EPOCHS × 2 clients × 10 attack types)

All these parameters can be configured in `federated_learning/config/config.py`.

## Development

The modular structure allows for easy development and testing of individual components:

1. Models can be modified independently in the `models/` directory
2. New attack types can be added in `attacks/attack_utils.py`
3. Data handling can be modified in `data/dataset.py`
4. Training procedures can be adjusted in the `training/` directory

## Team Collaboration

The modular structure enables parallel development:
1. Model developers can work on the `models/` directory
2. Security researchers can focus on the `attacks/` directory
3. Data engineers can work on the `data/` directory
4. Training algorithm developers can work on the `training/` directory

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

## Update Note

This version has been updated to use the `partial_scaling_attack` as the default attack type, which scales only a subset of gradient values (66% of values). This makes the attack more sophisticated and harder to detect compared to the basic scaling attack. 