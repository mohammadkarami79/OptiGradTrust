import torch
from federated_learning.training.server import Server

def test_server_init():
    print("Testing server initialization...")
    server = Server()
    
    # Check if global_model is initialized
    if hasattr(server, 'global_model'):
        print("SUCCESS: global_model is initialized")
        print(f"Model type: {type(server.global_model)}")
    else:
        print("ERROR: global_model is not initialized")
    
    # Check if model is initialized
    if hasattr(server, 'model'):
        print("NOTE: server also has 'model' attribute")
        print(f"Model type: {type(server.model)}")
    else:
        print("NOTE: server does not have 'model' attribute")
    
    # Try to get the device from model parameters
    try:
        device = next(server.global_model.parameters()).device
        print(f"SUCCESS: Device is {device}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
    
    print("Test completed")

if __name__ == "__main__":
    test_server_init() 