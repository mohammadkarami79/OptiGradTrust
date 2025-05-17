import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    # Try to import key modules
    import federated_learning
    print("✓ Successfully imported federated_learning module")
    
    from federated_learning.config.config import *
    print("✓ Successfully imported config")
    
    from federated_learning.models.attention import DualAttention
    print("✓ Successfully imported DualAttention model")
    
    from federated_learning.training.server import Server
    print("✓ Successfully imported Server")
    
    from federated_learning.utils.model_utils import update_model_with_gradient
    print("✓ Successfully imported update_model_with_gradient")
    
    print("\nAll imports successful!")
except Exception as e:
    print(f"Error during import: {str(e)}")
    import traceback
    traceback.print_exc() 