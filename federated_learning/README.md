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

This project supports several attack types:

- **None**: No attack, clients behave honestly.
- **Label Flipping**: Malicious clients flip the labels of their training data.
- **Scaling Attack**: Malicious clients scale their gradients by a large factor (NUM_CLIENTS).
- **Partial Scaling Attack**: Malicious clients scale only a subset of their gradient values (66% of values).
- **Backdoor Attack**: Malicious clients add a constant to their gradients.
- **Adaptive Attack**: Malicious clients add random noise to their gradients.

The current attack type is set to `partial_scaling_attack`.

## Update Note

This version has been updated to use the `partial_scaling_attack` as the default attack type, which scales only a subset of gradient values (66% of values). This makes the attack more sophisticated and harder to detect compared to the basic scaling attack. 