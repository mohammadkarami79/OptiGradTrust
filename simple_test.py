import sys
import os
import torch

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

print("\nCurrent working directory:", os.getcwd())
print("Directory contents:", os.listdir('.'))

print("\nTest completed successfully") 