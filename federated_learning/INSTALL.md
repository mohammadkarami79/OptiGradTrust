# Installation Guide

This document provides detailed instructions for setting up the Federated Learning project environment.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- Git

## Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/federated-learning.git
cd federated-learning
```

2. **Create a Virtual Environment (Recommended)**

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
```

## GPU Support (Optional)

To check if PyTorch can detect your GPU:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If you have a CUDA-capable GPU but PyTorch doesn't detect it:
1. Uninstall PyTorch: `pip uninstall torch torchvision`
2. Visit [PyTorch's website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system
3. Install the CUDA version of PyTorch using the command from the website

## Common Issues

1. **Out of Memory During Installation**
   - Try installing packages one by one
   - Use `pip install --no-cache-dir` to avoid caching
   - Clear pip cache: `pip cache purge`

2. **CUDA Not Found**
   - Ensure NVIDIA drivers are installed
   - Install CUDA Toolkit from NVIDIA website
   - Match PyTorch version with your CUDA version

3. **Import Errors**
   - Ensure you're in the virtual environment
   - Verify all dependencies are installed: `pip list`
   - Try reinstalling the problematic package

## Project Structure

After installation, your project structure should look like this:
```
federated_learning/
├── config/
│   └── config.py           # Configuration parameters
├── models/
│   ├── cnn.py             # CNN model for MNIST
│   ├── vae.py             # Gradient VAE model
│   └── attention.py       # Dual Attention mechanism
├── data/
│   └── dataset.py         # Dataset handling
├── attacks/
│   └── attack_utils.py    # Attack simulation
├── training/
│   ├── client.py          # Client implementation
│   ├── server.py          # Server implementation
│   └── training_utils.py  # Training utilities
├── requirements.txt       # Dependencies
└── main.py               # Main script
```

## Running the Project

After installation, you can run the project with:
```bash
python main.py
```

## Development Setup

For development, you might want to install additional packages:
```bash
pip install black flake8 pytest  # Code formatting, linting, and testing
```

## Support

If you encounter any issues during installation:
1. Check the [Common Issues](#common-issues) section
2. Open an issue on GitHub
3. Contact the maintainers 