import torch
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from federated_learning.utils.model_utils import update_model_with_gradient
    
    def test_model_update():
        print("Testing model update functionality...")
        
        # Create a simple model and gradient
        model = torch.nn.Linear(10, 5)
        gradient = torch.ones(55)  # 5*10 weights + 5 bias
        lr = 0.1
        
        # Print initial weight
        initial_weight = model.weight[0, 0].item()
        print(f"Initial weight: {initial_weight:.6f}")
        
        # Apply the update
        model, total_change, avg_change = update_model_with_gradient(model, gradient, lr)
        
        # Print final weight and change
        final_weight = model.weight[0, 0].item()
        print(f"Final weight: {final_weight:.6f}")
        print(f"Weight difference: {initial_weight - final_weight:.6f}")
        print(f"Expected change: {lr:.6f}")
        print(f"Total change: {total_change:.6f}")
        print(f"Average change: {avg_change:.6f}")
        
        # Verify the change is as expected
        if abs((initial_weight - final_weight) - lr) < 1e-6:
            print("✅ Update successful! Weight changed as expected.")
        else:
            print("❌ Update failed! Weight did not change as expected.")
            
    if __name__ == "__main__":
        test_model_update()
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc() 